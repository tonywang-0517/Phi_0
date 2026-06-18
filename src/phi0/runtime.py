"""Runtime helpers: model creation, training loop, batch preparation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from phi0.data.cosmos_video_size import cosmos_video_size_from_cfg
from phi0.data.processor import Phi0MixedDataset, Phi0Processor, build_overfit_datasets
from phi0.data.sequence import SequenceDataset, sequence_dataset_from_cfg
from phi0.data.action_stats import (
    compute_action_stats_from_datasets,
    load_action_stats,
    resolve_action_stats_path,
    save_action_stats,
)
from phi0.checkpoint_utils import checkpoint_paths, extract_action_expert_state_dict
from phi0.inference.session import PromptEmbedCache
from phi0.models.cosmos.loader import configure_hf_env
from phi0.models.phi0 import Phi0
from phi0.schema.draw_schema import D_RAW

logger = logging.getLogger(__name__)


def _mixed_precision_to_dtype(mp: str) -> torch.dtype:
    key = str(mp).strip().lower()
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    return torch.float32


def cuda_device_index(device: str) -> int:
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    return 0


def list_cuda_memory() -> list[tuple[int, int, int]]:
    """Return [(device_index, free_bytes, total_bytes), ...] for visible CUDA devices."""
    if not torch.cuda.is_available():
        return []
    out: list[tuple[int, int, int]] = []
    for idx in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
        out.append((idx, int(free_bytes), int(total_bytes)))
    return out


def pick_free_cuda_device(min_free_gb: float = 18.0) -> str:
    """Pick the visible GPU with the most free VRAM (must exceed ``min_free_gb``)."""
    min_free_bytes = int(min_free_gb * (1024**3))
    candidates = list_cuda_memory()
    if not candidates:
        raise RuntimeError("No CUDA devices visible to PyTorch.")
    idx, free_bytes, total_bytes = max(candidates, key=lambda x: x[1])
    if free_bytes < min_free_bytes:
        summary = ", ".join(
            f"cuda:{i} {free / (1024**3):.1f}/{total / (1024**3):.1f} GiB free"
            for i, free, total in candidates
        )
        raise RuntimeError(
            f"No GPU has >={min_free_gb:.0f} GiB free VRAM. Visible: {summary}"
        )
    logger.info(
        "Auto-selected cuda:%d (%.1f / %.1f GiB free)",
        idx,
        free_bytes / (1024**3),
        total_bytes / (1024**3),
    )
    return f"cuda:{idx}"


def resolve_inference_device(device: str, *, min_free_gb: float = 18.0) -> str:
    """Resolve inference device. ``cuda`` / ``auto`` → GPU with most free memory."""
    key = str(device).strip().lower()
    if key == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        if key in {"cuda", "auto", "cuda:auto"} or key.startswith("cuda"):
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return device
    if key in {"cuda", "auto", "cuda:auto"}:
        return pick_free_cuda_device(min_free_gb=min_free_gb)
    if key.startswith("cuda"):
        resolved = device if ":" in device else "cuda:0"
        idx = cuda_device_index(resolved)
        free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
        logger.info(
            "Using %s (%.1f / %.1f GiB free)",
            resolved,
            free_bytes / (1024**3),
            total_bytes / (1024**3),
        )
        return resolved
    return device


def activate_cuda_device(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.set_device(cuda_device_index(device))


def load_checkpoint_if_configured(
    model: Phi0,
    cfg: DictConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    """Load optional legacy FastWAM or Phi_0 resume checkpoint. Returns step to continue from."""
    fastwam_ckpt = cfg.get("fastwam_ckpt")
    if fastwam_ckpt is not None and str(fastwam_ckpt).lower() not in {"", "null", "none"}:
        path = Path(str(fastwam_ckpt))
        if not path.is_file():
            raise FileNotFoundError(f"fastwam_ckpt not found: {path}")
        logger.info("Loading fastwam_ckpt (action expert only): %s", path)
        model.load_checkpoint(str(path), optimizer=optimizer)
        return 0
    resume_ckpt = cfg.get("resume_ckpt")
    if resume_ckpt is not None and str(resume_ckpt).lower() not in {"", "null", "none"}:
        path = Path(str(resume_ckpt))
        if not path.is_file():
            raise FileNotFoundError(f"resume_ckpt not found: {path}")
        logger.info("Resuming from resume_ckpt: %s", path)
        payload = model.load_checkpoint(str(path), optimizer=optimizer)
        start = int(payload.get("step", 0)) if isinstance(payload, dict) else 0
        logger.info("Resume start step=%d", start)
        return start
    return 0


def build_optimizer(model: Phi0, cfg: DictConfig) -> torch.optim.AdamW:
    """DiT4DiT-style param groups: Cosmos transformer vs action expert."""
    lr_backbone = float(cfg.get("learning_rate_backbone", cfg.learning_rate))
    lr_action = float(cfg.get("learning_rate_action", cfg.learning_rate))

    backbone_params = []
    action_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("video_tower.transformer."):
            backbone_params.append(param)
        else:
            action_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})
    if action_params:
        param_groups.append({"params": action_params, "lr": lr_action})
    if not param_groups:
        raise ValueError("No trainable parameters found; check freeze_* settings.")

    return torch.optim.AdamW(param_groups, weight_decay=float(cfg.get("weight_decay", 0.0)))


def build_vggt_tower(cfg: DictConfig, *, device: str, torch_dtype: torch.dtype):
    """Optional frozen VGGT-Omega tower for dual cross-attention."""
    vggt_cfg = cfg.model.get("vggt")
    if vggt_cfg is None or not bool(vggt_cfg.get("enabled", False)):
        return None
    checkpoint = vggt_cfg.get("checkpoint_path")
    if checkpoint is None or str(checkpoint).lower() in {"", "null", "none"}:
        raise ValueError("vggt.enabled=true requires vggt.checkpoint_path")
    path = Path(str(checkpoint))
    if not path.is_file():
        raise FileNotFoundError(f"VGGT checkpoint not found: {path}")
    from phi0.models.vggt.tower import VGGTOmegaTower

    return VGGTOmegaTower(
        checkpoint_path=str(path),
        device=device,
        torch_dtype=torch_dtype,
        image_resolution=int(vggt_cfg.get("image_resolution", 512)),
        freeze=bool(vggt_cfg.get("freeze", True)),
    )


def create_phi0(cfg: DictConfig, smoke: bool = False) -> Phi0:
    if smoke or bool(cfg.get("smoke_action_only", False)):
        from phi0.models.factory_smoke import create_phi0_action_only_smoke

        dtype = _mixed_precision_to_dtype(cfg.get("mixed_precision", "bf16"))
        device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        model_cfg = cfg.model
        action_dit = OmegaConf.to_container(model_cfg.action_dit_config, resolve=True)
        return create_phi0_action_only_smoke(
            device=device,
            torch_dtype=dtype,
            hidden_dim=int(action_dit.get("hidden_dim", 1024)),
            num_layers=int(action_dit.get("num_layers", 2)),
            text_dim=int(action_dit.get("text_dim", 512)),
            action_head=str(model_cfg.get("action_head", "fm")),
        )

    ckpt_root = configure_hf_env(cfg.get("checkpoints_dir"))
    model_cfg = cfg.model
    dtype = _mixed_precision_to_dtype(cfg.get("mixed_precision", "bf16"))
    device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    action_dit = OmegaConf.to_container(model_cfg.action_dit_config, resolve=True)
    action_fm = OmegaConf.to_container(model_cfg.get("action_fm", {}), resolve=True)
    lambda_video = float(model_cfg.loss.get("lambda_video", 0.0))
    freeze_transformer = bool(model_cfg.get("freeze_transformer", False))
    if lambda_video <= 0 and not freeze_transformer:
        logger.warning(
            "lambda_video=0 but freeze_transformer=false: Cosmos DiT still runs each step for hook "
            "context without video-loss gradients. Set freeze_transformer=true for action-only training."
        )
    model = Phi0.from_cosmos_pretrained(
        device=device,
        torch_dtype=dtype,
        base_model=model_cfg.get("base_model"),
        revision=str(model_cfg.get("revision", "diffusers/base/post-trained")),
        checkpoints_dir=str(ckpt_root),
        load_text_encoder=bool(model_cfg.get("load_text_encoder", True)),
        load_transformer=bool(model_cfg.get("load_transformer", True)),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
        action_dit_config=action_dit,
        action_head=str(model_cfg.get("action_head", "fm")),
        action_fm_config=action_fm,
        extract_layer=int(model_cfg.get("extract_layer", 17)),
        num_context_tokens=int(model_cfg.get("num_context_tokens", 64)),
        raw_action_dim=int(model_cfg.get("raw_action_dim", D_RAW)),
        loss_lambda_video=float(model_cfg.loss.lambda_video),
        loss_lambda_action=float(model_cfg.loss.lambda_action),
        loss_lambda_bone=float(model_cfg.loss.get("lambda_bone", 0.0)),
        loss_lambda_bone_hand=float(model_cfg.loss.get("lambda_bone_hand", 0.0)),
        loss_lambda_bone_dir=float(model_cfg.loss.get("lambda_bone_dir", 0.0)),
        loss_lambda_hand_mse=float(model_cfg.loss.get("lambda_hand_mse", 0.0)),
        freeze_text_encoder=bool(model_cfg.get("freeze_text_encoder", True)),
        freeze_vae=bool(model_cfg.get("freeze_vae", True)),
        freeze_transformer=bool(model_cfg.get("freeze_transformer", False)),
        freeze_video_tower=(
            bool(model_cfg.freeze_video_tower)
            if model_cfg.get("freeze_video_tower") is not None
            else None
        ),
        detach_action_context=bool(model_cfg.get("detach_action_context", True)),
        action_context_mode=str(model_cfg.get("action_context_mode", "full_clip")),
        capture_stochastic=bool(model_cfg.get("capture_stochastic", False)),
        vae_sample=bool(model_cfg.get("vae_sample", False)),
        conditional_frame_timestep=float(model_cfg.get("conditional_frame_timestep", 0.0001)),
        enable_cosmos_gradient_checkpointing=bool(
            model_cfg.get("enable_cosmos_gradient_checkpointing", False)
        ),
        cosmos_hook_early_exit=bool(model_cfg.get("cosmos_hook_early_exit", True)),
        infer_generate_video=bool(model_cfg.get("inference", {}).get("generate_video", False)),
        video_fm_config=OmegaConf.to_container(model_cfg.get("video_fm", {}), resolve=True),
        prompt_max_length=int(model_cfg.get("prompt_max_length", 512)),
        past_action_window_size=int(model_cfg.get("past_action_window_size", 4)),
        action_history_window=(
            int(model_cfg["action_history_window"])
            if model_cfg.get("action_history_window") is not None
            else None
        ),
        action_future_horizon=(
            int(model_cfg["action_future_horizon"])
            if model_cfg.get("action_future_horizon") is not None
            else None
        ),
        vggt_use_full_video=bool(model_cfg.get("vggt_use_full_video", True)),
        vggt_tower=build_vggt_tower(cfg, device=device, torch_dtype=dtype),
    )
    # Importing diffusers (Cosmos pipeline) can leave grad disabled globally.
    torch.set_grad_enabled(True)
    return model


def build_dataloader(cfg: DictConfig) -> DataLoader:
    data_cfg = cfg.data
    video_path = data_cfg.get("xperience_video")
    if video_path is not None and str(video_path).lower() in {"", "null", "none"}:
        video_path = None
    cache_video = bool(data_cfg.get("cache_video", True))
    image_size = cosmos_video_size_from_cfg(data_cfg)
    mixed = build_overfit_datasets(
        xperience_max_frames=int(data_cfg.get("xperience_max_frames", 32)),
        egodex_max_frames=int(data_cfg.get("egodex_max_frames", 32)),
        xperience_video=video_path,
        cache_video=cache_video,
        image_size=image_size,
    )
    seq = sequence_dataset_from_cfg(mixed, data_cfg)
    num_workers = int(cfg.get("num_workers", 0))
    use_cuda = str(cfg.get("device", "cuda")).startswith("cuda") and torch.cuda.is_available()
    loader_kwargs: Dict[str, Any] = {
        "batch_size": int(cfg.batch_size),
        "shuffle": True,
        "num_workers": num_workers,
        "collate_fn": SequenceDataset.collate_fn,
        "pin_memory": use_cuda,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
    return DataLoader(seq, **loader_kwargs)


def _datasets_for_stats(cfg: DictConfig):
    data_cfg = cfg.data
    video_path = data_cfg.get("xperience_video")
    if video_path is not None and str(video_path).lower() in {"", "null", "none"}:
        video_path = None
    image_size = cosmos_video_size_from_cfg(data_cfg)
    mixed = build_overfit_datasets(
        xperience_max_frames=int(data_cfg.get("xperience_max_frames", 32)),
        egodex_max_frames=int(data_cfg.get("egodex_max_frames", 32)),
        xperience_video=video_path,
        cache_video=False,
        image_size=image_size,
    )
    return list(mixed.datasets)


def ensure_action_stats(cfg: DictConfig) -> Dict[str, Any]:
    """Load or compute z-score stats; write to ``action_stats_path`` if missing."""
    data_cfg = cfg.data
    out_dir = Path(str(cfg.output_dir))
    stats_path = resolve_action_stats_path(data_cfg, output_dir=out_dir)
    auto_compute = bool(data_cfg.get("auto_compute_action_stats", True))

    if stats_path is not None and stats_path.is_file():
        stats = load_action_stats(stats_path)
        logger.info("Loaded action stats from %s (%d frames)", stats_path, stats.get("num_frames", -1))
        return stats

    if not auto_compute:
        logger.warning("No action stats at %s; using identity normalize (mean=0 std=1)", stats_path)
        return {}

    logger.info("Computing action stats from training datasets...")
    stats = compute_action_stats_from_datasets(_datasets_for_stats(cfg), show_progress=True)
    if stats_path is not None:
        save_action_stats(stats, stats_path)
        logger.info("Wrote action stats: %s (%d frames)", stats_path, stats["num_frames"])
    return stats


def build_processor(cfg: DictConfig) -> Phi0Processor:
    data_cfg = cfg.data
    crop_scale = data_cfg.get("cosmos_video_crop_scale")
    if crop_scale is not None:
        crop_scale = float(crop_scale)
    processor = Phi0Processor(
        normalize=bool(data_cfg.get("normalize", True)),
        cosmos_video_size=cosmos_video_size_from_cfg(data_cfg),
        cosmos_video_crop_scale=crop_scale,
    )
    if not processor.normalize:
        return processor.train()
    stats = ensure_action_stats(cfg)
    if stats:
        processor.register_stats_from_dict(stats)
    return processor.train()


def sync_model_action_norm(model: Phi0, processor: Phi0Processor) -> None:
    if hasattr(model, "set_action_norm_stats"):
        model.set_action_norm_stats(processor.mean, processor.std)


def apply_processor_stats_from_checkpoint(
    processor: Phi0Processor,
    payload: dict,
    cfg: DictConfig,
) -> None:
    """Load z-score stats from checkpoint payload, stats file, or compute."""
    if not processor.normalize:
        return
    if isinstance(payload, dict) and payload.get("action_norm_stats"):
        processor.register_stats_from_dict(payload["action_norm_stats"])
        return
    stats_path = resolve_action_stats_path(cfg.data, output_dir=Path(str(cfg.get("output_dir", "."))))
    if stats_path is not None and stats_path.is_file():
        processor.load_stats_path(stats_path)
        return
    stats = ensure_action_stats(cfg)
    if stats:
        processor.register_stats_from_dict(stats)


def prepare_model_batch(
    model: Phi0,
    processor: Phi0Processor,
    batch: Dict[str, Any],
    *,
    prompt_cache: Optional[Any] = None,
) -> Dict[str, Any]:
    sample = processor.preprocess(batch)
    pixel = sample["pixel_values"]
    pixel_native = sample.get("pixel_values_native", pixel)
    # Mono camera only: [B, T, C, H, W]
    if pixel.ndim == 6:
        pixel = pixel[:, 0]
    if pixel_native.ndim == 6:
        pixel_native = pixel_native[:, 0]
    b, t, c, h, w = pixel.shape
    video = pixel.permute(0, 2, 1, 3, 4).contiguous()
    video = video * 2.0 - 1.0
    vggt_video = pixel_native.permute(0, 2, 1, 3, 4).contiguous()
    vggt_video = vggt_video * 2.0 - 1.0

    if getattr(model.video_tower, "text_encoder", None) is not None:
        if prompt_cache is not None:
            context, context_mask = prompt_cache.get_batch(model, sample["instruction"])
        else:
            context, context_mask = model.encode_prompt(sample["instruction"])
    else:
        dim = model.text_dim
        context = torch.zeros((b, 4, dim), device=model.device, dtype=model.torch_dtype)
        context_mask = torch.ones((b, 4), device=model.device, dtype=torch.bool)

    out = {
        "video": video,
        "vggt_video": vggt_video,
        "context": context,
        "context_mask": context_mask,
        "action": sample["action"],
        "action_is_pad": sample["action_is_pad"],
        "action_dim_is_pad": sample["action_dim_is_pad"],
        "image_is_pad": sample["image_is_pad"],
    }
    if batch.get("input_latents") is not None:
        out["input_latents"] = batch["input_latents"].to(device=model.device, dtype=model.torch_dtype)

    device = model.device
    dtype = model.torch_dtype
    video = video.to(device=device, dtype=dtype, non_blocking=True)
    vggt_video = vggt_video.to(device=device, dtype=dtype, non_blocking=True)
    context = context.to(device=device, dtype=dtype, non_blocking=True)
    context_mask = context_mask.to(device=device, non_blocking=True)
    out["video"] = video
    out["vggt_video"] = vggt_video
    out["context"] = context
    out["context_mask"] = context_mask

    # Frozen towers: precompute context once under inference_mode (no autograd graph).
    with torch.inference_mode():
        if float(getattr(model, "loss_lambda_video", 0.0)) <= 0 and float(
            getattr(model, "loss_lambda_action", 0.0)
        ) > 0:
            hook_video = video[:, :, -1:, :, :]
            _, action_ctx, action_ctx_mask = model.video_tower.forward_joint_step(
                hook_video,
                context,
                compute_video_loss=False,
            )
            out["action_ctx"] = action_ctx.detach().clone()
            out["action_ctx_mask"] = action_ctx_mask.detach().clone()

        if model.uses_dual_vggt_cross_attn() and model.vggt_tower is not None:
            vggt_ctx, vggt_ctx_mask = model.vggt_tower.extract_register_context(vggt_video)
            out["vggt_ctx"] = vggt_ctx.detach()
            out["vggt_ctx_mask"] = vggt_ctx_mask

    return out


def save_training_checkpoint(
    model: Phi0,
    cfg: DictConfig,
    step: int,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Path:
    """Save training checkpoint; overwrite mode keeps a single weights file."""
    out_dir = Path(str(cfg.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = str(cfg.get("checkpoint_name", "phi0"))
    step_path, latest_path, legacy_alias = checkpoint_paths(out_dir, ckpt_name, step)
    overwrite = bool(cfg.get("checkpoint_overwrite", False))
    save_action_only = bool(
        cfg.get(
            "save_action_expert_only",
            float(cfg.model.loss.get("lambda_video", 1.0)) <= 0,
        )
    )
    payload: Dict[str, Any] = {
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "step": step,
    }
    stats_path = resolve_action_stats_path(cfg.data, output_dir=Path(str(cfg.output_dir)))
    if stats_path is not None and stats_path.is_file():
        payload["action_norm_stats"] = load_action_stats(stats_path)
    if save_action_only:
        payload["action_expert"] = extract_action_expert_state_dict(model)
        payload["checkpoint_kind"] = "action_expert_only"
        logger.info("Saving action_expert-only checkpoint (%d tensors)", len(payload["action_expert"]))
    else:
        payload["model"] = model.state_dict()
        payload["checkpoint_kind"] = "full"
    if optimizer is not None and bool(cfg.get("save_optimizer", False)):
        payload["optimizer"] = optimizer.state_dict()
        logger.info("Including optimizer state in checkpoint")

    if overwrite:
        target = latest_path
        torch.save(payload, target)
        logger.info("Saved checkpoint (overwrite): %s [step=%d]", target, step)
        return target

    torch.save(payload, step_path)
    torch.save(payload, latest_path)
    torch.save(payload, legacy_alias)
    logger.info("Saved checkpoints: %s, %s (legacy alias %s)", step_path, latest_path, legacy_alias)
    return latest_path


def run_training(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    processor = build_processor(cfg)
    model = create_phi0(cfg, smoke=bool(cfg.get("smoke_action_only", False)))
    sync_model_action_norm(model, processor)
    model.repeated_action_steps = max(1, int(cfg.get("repeated_action_steps", 1)))
    loader = build_dataloader(cfg)
    optim = build_optimizer(model, cfg)
    start_step = load_checkpoint_if_configured(model, cfg, optimizer=optim)
    device = model.device
    max_steps = int(cfg.get("max_steps", 2))
    if start_step >= max_steps:
        logger.warning(
            "start_step=%d >= max_steps=%d; saving checkpoint and exiting",
            start_step,
            max_steps,
        )
        save_training_checkpoint(model, cfg, start_step, optimizer=optim)
        return
    mp = str(cfg.get("mixed_precision", "bf16")).lower()
    use_amp = mp in {"bf16", "fp16"} and device.type == "cuda"
    autocast_dtype = _mixed_precision_to_dtype(mp)
    save_every = int(cfg.get("save_every_steps", 0))
    prompt_cache = PromptEmbedCache()
    step = start_step
    logger.info("Training steps %d -> %d", start_step, max_steps)
    model.train()
    model.set_frozen_towers_eval()
    while step < max_steps:
        for batch in loader:
            model.set_frozen_towers_eval()
            model_batch = prepare_model_batch(model, processor, batch, prompt_cache=prompt_cache)
            model_batch = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in model_batch.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=use_amp,
            ):
                loss, loss_dict = model.training_loss(model_batch)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            max_norm = cfg.get("gradient_clipping")
            if max_norm is not None and float(max_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_norm))
            optim.step()
            logger.info("step=%d loss=%.4f %s", step, float(loss.item()), loss_dict)
            step += 1
            if save_every > 0 and step % save_every == 0:
                save_training_checkpoint(model, cfg, step, optimizer=optim)
            if step >= max_steps:
                break

    if save_every <= 0 or step % save_every != 0:
        save_training_checkpoint(model, cfg, step, optimizer=optim)
