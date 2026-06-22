"""Runtime helpers: model creation, training loop, batch preparation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from phi0.data.dataloader_policy import log_dataloader_settings, resolve_dataloader_settings
from phi0.data.libero_rlds import LiberoRldsFrameDataset
from phi0.data.processor import Phi0MixedDataset, Phi0Processor, build_overfit_datasets
from phi0.data.sequence import SequenceDataset, sequence_dataset_from_cfg
from phi0.data.action_stats import (
    compute_action_stats_for_data_cfg,
    load_action_stats,
    load_or_validate_stats,
    resolve_action_stats_path,
    resolve_proprio_stats_path,
    save_action_stats,
)
from phi0.data.robot_action_norm import normalize_robot7d, stats_view_for_robot7d
from phi0.checkpoint_utils import (
    checkpoint_paths,
    extract_action_expert_state_dict,
    resolve_resume_checkpoint,
    unwrap_compiled_module,
)
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


def maybe_compile_action_expert(model: Phi0, cfg: DictConfig) -> None:
    """Optionally ``torch.compile`` the trainable action head (after checkpoint load)."""
    if not bool(cfg.get("compile_action_expert", False)):
        return
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile unavailable; skipping compile_action_expert")
        return
    expert = getattr(model, "action_expert", None)
    if expert is None:
        return
    expert = unwrap_compiled_module(expert)
    mode = str(cfg.get("compile_action_expert_mode", "default")).strip() or "default"
    try:
        logger.info("torch.compile(action_expert, mode=%s)", mode)
        model.action_expert = torch.compile(expert, mode=mode)
    except Exception as exc:
        logger.warning(
            "torch.compile(action_expert) failed (%s); continuing without compile.",
            exc,
        )


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
    resume_path = resolve_resume_checkpoint(cfg)
    if resume_path is not None:
        path = Path(str(resume_path))
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        logger.info("Resuming training from %s", path)
        payload = model.load_checkpoint(str(path), optimizer=optimizer)
        start = int(payload.get("step", 0)) if isinstance(payload, dict) else 0
        logger.info("Resume start step=%d", start)
        return start
    return 0


def build_optimizer(model: Phi0, cfg: DictConfig) -> torch.optim.AdamW:
    """Param groups: optional VLM backbone vs action expert."""
    from phi0.training.lr_schedule import scaled_action_learning_rate

    lr_backbone = float(cfg.get("learning_rate_backbone", cfg.learning_rate))
    scale_mode = str(cfg.get("lr_scale", "none")).strip().lower()
    if scale_mode in {"sqrt", "linear"}:
        lr_action = scaled_action_learning_rate(
            per_device_batch=int(cfg.batch_size),
            reference_batch=int(cfg.get("reference_batch_size", 16)),
            reference_lr=float(cfg.get("reference_lr_action", 1.5e-4)),
            scale=scale_mode,  # type: ignore[arg-type]
            explicit_lr=float(cfg.get("learning_rate_action", 1.5e-4)),
        )
    else:
        lr_action = float(cfg.get("learning_rate_action", cfg.learning_rate))

    backbone_params = []
    action_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("vlm_tower.") or name.startswith("video_tower."):
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

    betas = (
        float(cfg.get("adam_beta1", 0.9)),
        float(cfg.get("adam_beta2", 0.999)),
    )
    return torch.optim.AdamW(
        param_groups,
        betas=betas,
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        eps=float(cfg.get("adam_eps", 1e-8)),
    )


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

    tower = VGGTOmegaTower(
        checkpoint_path=str(path),
        device=device,
        torch_dtype=torch_dtype,
        image_resolution=int(vggt_cfg.get("image_resolution", 512)),
        freeze=bool(vggt_cfg.get("freeze", True)),
    )
    if tower.freeze:
        for param in tower.parameters():
            param.requires_grad = False
    total_params = sum(p.numel() for p in tower.parameters())
    logger.info(
        "Loaded VGGT-Omega from %s (%.2fM params, freeze=%s)",
        path,
        total_params / 1e6,
        tower.freeze,
    )
    return tower


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
            past_action_window_size=int(model_cfg.get("past_action_window_size", 5)),
        )

    ckpt_root = str(cfg.get("checkpoints_dir", "./checkpoints"))
    model_cfg = cfg.model
    dtype = _mixed_precision_to_dtype(cfg.get("mixed_precision", "bf16"))
    device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    action_dit = OmegaConf.to_container(model_cfg.action_dit_config, resolve=True)
    action_fm = OmegaConf.to_container(model_cfg.get("action_fm", {}), resolve=True)
    vlm_cfg = OmegaConf.to_container(model_cfg.get("vlm", {}), resolve=True) or {}
    vlm_enabled = bool(vlm_cfg.get("enabled", True))
    if not vlm_enabled:
        model = Phi0.from_action_only(
            device=device,
            torch_dtype=dtype,
            action_dit_config=action_dit,
            action_head=str(model_cfg.get("action_head", "fm")),
            action_fm_config=action_fm,
            raw_action_dim=(
                7
                if int(model_cfg.get("robot_action_dim", 0)) == 7
                else int(model_cfg.get("raw_action_dim", D_RAW))
            ),
            loss_lambda_action=float(model_cfg.loss.lambda_action),
            loss_lambda_bone=float(model_cfg.loss.get("lambda_bone", 0.0)),
            loss_lambda_bone_hand=float(model_cfg.loss.get("lambda_bone_hand", 0.0)),
            loss_lambda_bone_dir=float(model_cfg.loss.get("lambda_bone_dir", 0.0)),
            loss_lambda_hand_mse=float(model_cfg.loss.get("lambda_hand_mse", 0.0)),
            prompt_max_length=int(model_cfg.get("prompt_max_length", 512)),
            past_action_window_size=int(model_cfg.get("past_action_window_size", 1)),
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
            vggt_use_full_video=bool(model_cfg.get("vggt_use_full_video", False)),
            vggt_tower=build_vggt_tower(cfg, device=device, torch_dtype=dtype),
        )
        if int(model_cfg.get("robot_action_dim", 0)) == 7:
            model.robot_action_loss_type = str(
                model_cfg.get("robot_action_loss_type", "l1")
            ).strip().lower()
        torch.set_grad_enabled(True)
        _log_trainable_scope(model)
        return model

    vlm_path = vlm_cfg.get("model_path") or model_cfg.get("vlm_model_path") or model_cfg.get("base_model")
    model = Phi0.from_vlm_pretrained(
        device=device,
        torch_dtype=dtype,
        vlm_model_path=vlm_path,
        checkpoints_dir=ckpt_root,
        local_files_only=bool(vlm_cfg.get("local_files_only", True)),
        freeze_vlm=bool(vlm_cfg.get("freeze", model_cfg.get("freeze_vlm", True))),
        attn_implementation=str(vlm_cfg.get("attn_implementation", "flash_attention_2")),
        action_dit_config=action_dit,
        action_head=str(model_cfg.get("action_head", "fm")),
        action_fm_config=action_fm,
        raw_action_dim=(
            7
            if int(model_cfg.get("robot_action_dim", 0)) == 7
            else int(model_cfg.get("raw_action_dim", D_RAW))
        ),
        loss_lambda_action=float(model_cfg.loss.lambda_action),
        loss_lambda_bone=float(model_cfg.loss.get("lambda_bone", 0.0)),
        loss_lambda_bone_hand=float(model_cfg.loss.get("lambda_bone_hand", 0.0)),
        loss_lambda_bone_dir=float(model_cfg.loss.get("lambda_bone_dir", 0.0)),
        loss_lambda_hand_mse=float(model_cfg.loss.get("lambda_hand_mse", 0.0)),
        prompt_max_length=int(model_cfg.get("prompt_max_length", 512)),
        past_action_window_size=int(model_cfg.get("past_action_window_size", 1)),
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
        vggt_use_full_video=bool(model_cfg.get("vggt_use_full_video", False)),
        vggt_tower=build_vggt_tower(cfg, device=device, torch_dtype=dtype),
    )
    if int(model_cfg.get("robot_action_dim", 0)) == 7:
        model.robot_action_loss_type = str(
            model_cfg.get("robot_action_loss_type", "l1")
        ).strip().lower()
    # Importing diffusers can leave grad disabled globally.
    torch.set_grad_enabled(True)
    _log_trainable_scope(model)
    return model


def _log_trainable_scope(model: Phi0) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    groups = {"vlm_tower": 0, "vggt_tower": 0, "action_expert": 0, "other": 0}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        key = "other"
        if name.startswith("action_expert."):
            key = "action_expert"
        elif name.startswith("vlm_tower.") or name.startswith("video_tower."):
            key = "vlm_tower"
        elif name.startswith("vggt_tower."):
            key = "vggt_tower"
        groups[key] += param.numel()
    logger.info(
        "Trainable %.2fM / %.2fM total (action_expert=%.2fM, vlm=%.2fM, vggt=%.2fM)",
        trainable / 1e6,
        total / 1e6,
        groups["action_expert"] / 1e6,
        groups["vlm_tower"] / 1e6,
        groups["vggt_tower"] / 1e6,
    )


def build_base_dataset(
    cfg: DictConfig,
    *,
    dist_ctx=None,
    loader_settings=None,
):
    """Frame-level dataset before SequenceDataset clipping."""
    data_cfg = cfg.data
    dataset_name = str(data_cfg.get("dataset", "xperience")).strip().lower()
    if dataset_name in {"libero_spatial", "libero", "libero_rlds"}:
        if loader_settings is None:
            loader_settings = resolve_dataloader_settings(cfg, dist_ctx=dist_ctx)
        suite = str(data_cfg.get("libero_suite", "libero_spatial"))
        max_eps = data_cfg.get("libero_max_episodes")
        max_shards = data_cfg.get("libero_max_shards")
        return LiberoRldsFrameDataset(
            suite=suite,
            rlds_root=data_cfg.get("libero_rlds_root"),
            image_size=(256, 256),
            max_episodes=int(max_eps) if max_eps is not None else None,
            max_shards=int(max_shards) if max_shards is not None else None,
            libero_delta_eef=bool(data_cfg.get("libero_delta_eef", True)),
            defer_cosmos_resize=True,
            cache_native_frames=loader_settings.cache_native_frames,
            mono_camera=bool(data_cfg.get("mono_camera", True)),
        )

    video_path = data_cfg.get("xperience_video")
    if video_path is not None and str(video_path).lower() in {"", "null", "none"}:
        video_path = None
    cache_video = bool(data_cfg.get("cache_video", True))
    return build_overfit_datasets(
        xperience_max_frames=int(data_cfg.get("xperience_max_frames", 32)),
        egodex_max_frames=int(data_cfg.get("egodex_max_frames", 32)),
        xperience_video=video_path,
        cache_video=cache_video,
    )


def build_dataloader(cfg: DictConfig, *, dist_ctx=None) -> DataLoader:
    data_cfg = cfg.data
    loader_settings = resolve_dataloader_settings(cfg, dist_ctx=dist_ctx)
    log_dataloader_settings(loader_settings, logger=logger, dist_ctx=dist_ctx)
    base = build_base_dataset(cfg, dist_ctx=dist_ctx, loader_settings=loader_settings)
    seq = sequence_dataset_from_cfg(base, data_cfg)
    num_workers = loader_settings.num_workers
    use_cuda = str(cfg.get("device", "cuda")).startswith("cuda") and torch.cuda.is_available()
    loader_kwargs: Dict[str, Any] = {
        "batch_size": int(cfg.batch_size),
        "num_workers": num_workers,
        "collate_fn": SequenceDataset.collate_fn,
        "pin_memory": use_cuda,
    }
    if dist_ctx is not None and dist_ctx.enabled:
        from torch.utils.data.distributed import DistributedSampler

        loader_kwargs["sampler"] = DistributedSampler(
            seq,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=True,
            drop_last=bool(cfg.get("ddp_drop_last", True)),
        )
        loader_kwargs["drop_last"] = bool(cfg.get("ddp_drop_last", True))
    else:
        loader_kwargs["shuffle"] = True
        if bool(cfg.get("train_drop_last", False)):
            loader_kwargs["drop_last"] = True
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = loader_settings.prefetch_factor
    return DataLoader(seq, **loader_kwargs)


def _datasets_for_stats(cfg: DictConfig):
    base = build_base_dataset(cfg)
    if isinstance(base, Phi0MixedDataset):
        return list(base.datasets)
    return [base]


def ensure_action_stats(
    cfg: DictConfig,
    *,
    proprio: bool = False,
    allow_compute: bool = True,
) -> Dict[str, Any]:
    """Load or compute action/proprio stats; recompute when semantics or norm_mode mismatch."""
    data_cfg = cfg.data
    out_dir = Path(str(cfg.output_dir))
    stats_path = (
        resolve_proprio_stats_path(data_cfg, output_dir=out_dir)
        if proprio
        else resolve_action_stats_path(data_cfg, output_dir=out_dir)
    )
    auto_compute = bool(data_cfg.get("auto_compute_action_stats", True)) and allow_compute

    if stats_path is not None and stats_path.is_file():
        try:
            stats = load_or_validate_stats(stats_path, data_cfg, proprio=proprio)
            if stats is not None:
                label = "proprio" if proprio else "action"
                logger.info(
                    "Loaded %s stats from %s (%d frames, semantics=%s, norm=%s)",
                    label,
                    stats_path,
                    stats.get("num_frames", -1),
                    stats.get("robot_action_semantics", "?"),
                    stats.get("norm_mode", "?"),
                )
                return stats
        except ValueError as exc:
            if not auto_compute:
                raise
            logger.warning("Invalid stats at %s (%s); recomputing.", stats_path, exc)

    if not auto_compute:
        logger.warning("No valid stats at %s; using identity normalize (mean=0 std=1)", stats_path)
        return {}

    label = "proprio" if proprio else "action"
    logger.info("Computing %s stats from training datasets...", label)
    stats = compute_action_stats_for_data_cfg(
        _datasets_for_stats(cfg),
        data_cfg,
        proprio=proprio,
        show_progress=True,
    )
    if stats_path is not None:
        save_action_stats(stats, stats_path)
        logger.info("Wrote %s stats: %s (%d frames)", label, stats_path, stats["num_frames"])
    return stats


def build_processor(cfg: DictConfig, *, dist_ctx=None) -> Phi0Processor:
    data_cfg = cfg.data
    model_cfg = cfg.model
    vlm_cfg = model_cfg.get("vlm") or {}
    vlm_size = vlm_cfg.get("image_size", [180, 320])
    processor = Phi0Processor(
        normalize=bool(data_cfg.get("normalize", True)),
        vlm_image_size=(int(vlm_size[0]), int(vlm_size[1])),
        vlm_img_aug=bool(vlm_cfg.get("img_aug", False)),
        use_wrist_view=not bool(data_cfg.get("mono_camera", True)),
    )
    if not processor.normalize:
        return processor.train()
    allow_compute = dist_ctx is None or not dist_ctx.enabled or dist_ctx.is_main
    if dist_ctx is not None and dist_ctx.enabled and not dist_ctx.is_main:
        from phi0.distributed import barrier

        barrier(dist_ctx)
    stats = ensure_action_stats(cfg, allow_compute=allow_compute)
    if stats:
        processor.register_stats_from_dict(stats)
    if bool(data_cfg.get("libero_delta_eef", False)):
        proprio_stats = ensure_action_stats(cfg, proprio=True, allow_compute=allow_compute)
        if proprio_stats:
            processor.register_proprio_stats_from_dict(proprio_stats)
        else:
            processor.proprio_mean = processor.mean.clone()
            processor.proprio_std = processor.std.clone()
            processor.proprio_q01 = processor.action_q01.clone()
            processor.proprio_q99 = processor.action_q99.clone()
    if dist_ctx is not None and dist_ctx.enabled and dist_ctx.is_main:
        from phi0.distributed import barrier

        barrier(dist_ctx)
    return processor.train()


def sync_model_action_norm(model: Phi0, processor: Phi0Processor) -> None:
    if hasattr(model, "set_action_norm_stats"):
        q01 = getattr(processor, "action_q01", processor.mean)
        q99 = getattr(processor, "action_q99", processor.mean)
        model.set_action_norm_stats(
            processor.mean,
            processor.std,
            q01=q01,
            q99=q99,
            norm_mode=getattr(processor, "action_norm_mode", "z-score"),
            normalize_gripper=getattr(processor, "normalize_gripper", True),
        )


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
    else:
        stats_path = resolve_action_stats_path(cfg.data, output_dir=Path(str(cfg.get("output_dir", "."))))
        if stats_path is not None and stats_path.is_file():
            processor.load_stats_path(stats_path)
        else:
            stats = ensure_action_stats(cfg)
            if stats:
                processor.register_stats_from_dict(stats)
    if bool(cfg.data.get("libero_delta_eef", False)):
        proprio_stats = ensure_action_stats(cfg, proprio=True)
        if proprio_stats:
            processor.register_proprio_stats_from_dict(proprio_stats)
        else:
            processor.proprio_mean = processor.mean.clone()
            processor.proprio_std = processor.std.clone()
            processor.proprio_q01 = processor.action_q01.clone()
            processor.proprio_q99 = processor.action_q99.clone()


def _normalize_robot7d_chunk(
    processor: Phi0Processor,
    action_7d: torch.Tensor,
    *,
    proprio: bool = False,
    normalize_gripper: Optional[bool] = None,
) -> torch.Tensor:
    stats = stats_view_for_robot7d(processor, proprio=proprio)
    grip = processor.normalize_gripper if normalize_gripper is None else normalize_gripper
    if proprio:
        grip = True
    return normalize_robot7d(action_7d.float(), stats, normalize_gripper=grip)


def _normalize_libero_proprio_delta_batch(
    processor: Phi0Processor,
    batch: Dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize proprio (absolute) + future (delta); return (normed, merged_7d, future_7d)."""
    proprio = batch["robot_proprio_7d"].float()
    future = batch["robot_future_delta_7d"].float()
    norm_proprio = _normalize_robot7d_chunk(processor, proprio, proprio=True)
    norm_future = _normalize_robot7d_chunk(
        processor, future, proprio=False, normalize_gripper=False
    )
    normed = torch.cat([norm_proprio, norm_future], dim=1)
    merged_7d = torch.cat([proprio, future], dim=1)
    return normed, merged_7d, future


def _normalize_libero_absolute_batch(
    processor: Phi0Processor,
    batch: Dict[str, Any],
) -> torch.Tensor:
    return _normalize_robot7d_chunk(processor, batch["robot_action_7d"].float(), proprio=False)


def prepare_model_batch_cpu(
    model: Phi0,
    processor: Phi0Processor,
    batch: Dict[str, Any],
) -> Dict[str, Any]:
    """CPU-side batch prep: robot encode, processor, VLM tokenization (no GPU towers)."""
    from phi0.checkpoint_utils import unwrap_training_module
    from phi0.models.vlm.preprocess import build_training_vlm_inputs_from_pixels

    model = unwrap_training_module(model)
    batch_work = batch
    normalized_robot_action = None
    merged_robot_7d = None
    future_delta_7d = None
    if "robot_proprio_7d" in batch and "robot_future_delta_7d" in batch:
        normalized_robot_action, merged_robot_7d, future_delta_7d = (
            _normalize_libero_proprio_delta_batch(processor, batch)
        )
    elif "robot_action_7d" in batch and model.uses_robot7d_action():
        normalized_robot_action = _normalize_libero_absolute_batch(processor, batch)
    sample = processor.preprocess(batch_work)
    if normalized_robot_action is not None:
        sample["action"] = normalized_robot_action

    payload: Dict[str, Any] = {
        "sample": sample,
        "batch": batch,
        "batch_work": batch_work,
        "merged_robot_7d": merged_robot_7d,
        "future_delta_7d": future_delta_7d,
    }
    if not model.uses_vlm_tower():
        return payload

    pixel = sample["pixel_values"]
    wrist_pixel = None
    if pixel.ndim == 6:
        if processor.use_wrist_view:
            if pixel.shape[1] < 2:
                raise ValueError(
                    f"use_wrist_view=True but pixel_values has shape {tuple(pixel.shape)}"
                )
            wrist_pixel = pixel[:, 1]
        pixel = pixel[:, 0]
    frame_idx = _resolve_observation_frame_index(model, pixel, batch)
    obs_pixel = pixel[:, frame_idx : frame_idx + 1]
    obs_wrist_pixel = None
    if wrist_pixel is not None:
        obs_wrist_pixel = wrist_pixel[:, frame_idx : frame_idx + 1]

    processor_obj = getattr(model.vlm_tower, "processor", None)
    if processor_obj is None:
        raise RuntimeError("VLM tower missing HuggingFace processor.")
    vlm_inputs = build_training_vlm_inputs_from_pixels(
        processor_obj,
        processor,
        obs_pixel,
        sample["instruction"],
        model_max_length=int(getattr(model, "prompt_max_length", 512)),
        wrist_pixel=obs_wrist_pixel,
    )

    payload.update(
        {
            "obs_pixel": obs_pixel,
            "vlm_inputs": vlm_inputs,
        }
    )
    return payload


def _resolve_observation_frame_index(
    model: Phi0,
    pixel: torch.Tensor,
    batch: Dict[str, Any],
) -> int:
    """Subsampled pixel timeline index aligned with proprio-prefix current step."""
    from phi0.data.temporal_align import (
        observation_subsampled_frame_index,
        video_sample_control_indices,
    )

    past_w = int(getattr(model, "past_action_window_size", 1))
    video_ctrl = batch.get("video_control_indices")
    if video_ctrl is not None:
        return observation_subsampled_frame_index(past_w, video_ctrl)
    ratio = int(batch.get("action_video_freq_ratio") or 2)
    seq_len = int(batch["action_is_pad"].shape[1])
    subsampled = video_sample_control_indices(seq_len, ratio)
    return observation_subsampled_frame_index(past_w, subsampled)


_TOWER_PREP_STREAMS: Dict[int, "TowerPrepStreams"] = {}


def get_tower_prep_streams(device: torch.device):
    from phi0.data.train_prefetch import TowerPrepStreams

    if device.type != "cuda":
        return None
    idx = device.index if device.index is not None else 0
    streams = _TOWER_PREP_STREAMS.get(idx)
    if streams is None:
        streams = TowerPrepStreams.for_device(device)
        _TOWER_PREP_STREAMS[idx] = streams
    return streams


def prepare_model_batch_gpu(
    model: Phi0,
    processor: Phi0Processor,
    cpu_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """GPU-side batch prep: H2D transfer + frozen VLM/VGGT towers."""
    from phi0.checkpoint_utils import unwrap_training_module

    model = unwrap_training_module(model)
    sample = cpu_payload["sample"]
    batch = cpu_payload["batch"]
    batch_work = cpu_payload["batch_work"]
    merged_robot_7d = cpu_payload.get("merged_robot_7d")
    future_delta_7d = cpu_payload.get("future_delta_7d")

    device = model.device
    out: Dict[str, Any] = {
        "action": sample["action"].to(device=device, non_blocking=True),
        "action_is_pad": sample["action_is_pad"].to(device=device, non_blocking=True),
        "action_dim_is_pad": sample["action_dim_is_pad"].to(device=device, non_blocking=True),
    }
    if "image_is_pad" in sample:
        image_is_pad = sample["image_is_pad"]
        if model.uses_vlm_tower():
            obs_pixel = cpu_payload["obs_pixel"]
            frame_idx = _resolve_observation_frame_index(model, obs_pixel, batch)
            if image_is_pad.ndim == 2 and image_is_pad.shape[1] > 1:
                image_is_pad = image_is_pad[:, frame_idx : frame_idx + 1]
        out["image_is_pad"] = image_is_pad.to(device=device, non_blocking=True)

    if model.uses_vlm_tower():
        vlm_inputs = dict(cpu_payload["vlm_inputs"])
        obs_pixel = cpu_payload["obs_pixel"]
        for key, value in vlm_inputs.items():
            vlm_inputs[key] = value.to(device=device, non_blocking=True)

        obs_pixel_gpu = obs_pixel.float().to(device=device, non_blocking=True)
        vggt_video = obs_pixel_gpu.permute(0, 2, 1, 3, 4).contiguous().mul_(2.0).sub_(1.0)
        out.update(vlm_inputs)
        if model.uses_cross_attn_context():
            out["vggt_video"] = vggt_video
    if merged_robot_7d is not None:
        out["robot_action_7d"] = merged_robot_7d.to(device=device, dtype=torch.float32)
    elif "robot_action_7d" in batch_work:
        out["robot_action_7d"] = batch_work["robot_action_7d"].to(
            device=device, dtype=torch.float32
        )
    if future_delta_7d is not None:
        out["robot_future_delta_7d"] = future_delta_7d.to(
            device=device, dtype=torch.float32
        )
    elif batch_work.get("robot_future_delta_7d") is not None:
        out["robot_future_delta_7d"] = batch_work["robot_future_delta_7d"].to(
            device=device, dtype=torch.float32
        )

    tower_streams = get_tower_prep_streams(device)
    vlm_stream = tower_streams.vlm if tower_streams is not None else None
    vggt_stream = tower_streams.vggt if tower_streams is not None else None

    with torch.inference_mode():
        need_vlm = (
            model.uses_vlm_tower()
            and model.uses_cross_attn_context()
            and float(getattr(model, "loss_lambda_action", 0.0)) > 0
        )
        need_vggt = model.uses_dual_vggt_cross_attn() and model.vggt_tower is not None

        if need_vlm:
            if vlm_stream is not None:
                with torch.cuda.stream(vlm_stream):
                    action_ctx, action_ctx_mask = model.vlm_tower.extract_action_context(
                        out["input_ids"],
                        out["attention_mask"],
                        out["pixel_values"],
                        out["image_grid_thw"],
                        out.get("mm_token_type_ids"),
                    )
            else:
                action_ctx, action_ctx_mask = model.vlm_tower.extract_action_context(
                    out["input_ids"],
                    out["attention_mask"],
                    out["pixel_values"],
                    out["image_grid_thw"],
                    out.get("mm_token_type_ids"),
                )
            out["action_ctx"] = action_ctx.detach()
            out["action_ctx_mask"] = action_ctx_mask.detach()

        if need_vggt:
            if vggt_stream is not None:
                with torch.cuda.stream(vggt_stream):
                    vggt_ctx, vggt_ctx_mask = model._resolve_vggt_context(
                        vggt_video, inputs={"vggt_video": vggt_video}
                    )
            else:
                vggt_ctx, vggt_ctx_mask = model._resolve_vggt_context(
                    vggt_video, inputs={"vggt_video": vggt_video}
                )
            out["vggt_ctx"] = vggt_ctx.detach()
            out["vggt_ctx_mask"] = vggt_ctx_mask

        default_stream = torch.cuda.current_stream(device)
        if vlm_stream is not None:
            default_stream.wait_stream(vlm_stream)
        if vggt_stream is not None:
            default_stream.wait_stream(vggt_stream)

    return out


def prepare_model_batch(
    model: Phi0,
    processor: Phi0Processor,
    batch: Dict[str, Any],
) -> Dict[str, Any]:
    cpu_payload = prepare_model_batch_cpu(model, processor, batch)
    return prepare_model_batch_gpu(model, processor, cpu_payload)


def save_training_checkpoint(
    model: Phi0,
    cfg: DictConfig,
    step: int,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    dist_ctx=None,
) -> Optional[Path]:
    """Save training checkpoint; overwrite mode keeps a single weights file."""
    from phi0.checkpoint_utils import unwrap_training_module

    if dist_ctx is not None and dist_ctx.enabled and not dist_ctx.is_main:
        from phi0.distributed import barrier

        barrier(dist_ctx)
        return None

    model = unwrap_training_module(model)
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
        for stale in out_dir.glob(f"{ckpt_name}_step*.pt"):
            if stale != target:
                try:
                    stale.unlink()
                except OSError as exc:
                    logger.warning("Could not remove stale checkpoint %s: %s", stale, exc)
        legacy = out_dir / "phi0_smoke.pt"
        if legacy.is_file() and legacy != target:
            try:
                legacy.unlink()
            except OSError:
                pass
        logger.info("Saved checkpoint (overwrite): %s [step=%d]", target, step)
        if dist_ctx is not None and dist_ctx.enabled:
            from phi0.distributed import barrier

            barrier(dist_ctx)
        return target

    torch.save(payload, step_path)
    torch.save(payload, latest_path)
    torch.save(payload, legacy_alias)
    logger.info("Saved checkpoints: %s, %s (legacy alias %s)", step_path, latest_path, legacy_alias)
    if dist_ctx is not None and dist_ctx.enabled:
        from phi0.distributed import barrier

        barrier(dist_ctx)
    return latest_path


def run_training(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    from phi0.distributed import (
        DistributedContext,
        barrier,
        configure_logging_for_rank,
        distributed_context_from_cfg,
        setup_process_device,
        wrap_ddp,
    )

    from phi0.training.lr_schedule import apply_warmup_lr, build_lr_scheduler

    dist_ctx: DistributedContext = distributed_context_from_cfg(cfg)
    configure_logging_for_rank(dist_ctx)
    setup_process_device(dist_ctx)
    if dist_ctx.enabled:
        from omegaconf import open_dict

        with open_dict(cfg):
            cfg.device = dist_ctx.device
        if dist_ctx.is_main:
            logger.info(
                "Distributed training (VLA-Adapter-style torchrun+DDP): "
                "rank=%d local_rank=%d world_size=%d device=%s",
                dist_ctx.rank,
                dist_ctx.local_rank,
                dist_ctx.world_size,
                dist_ctx.device,
            )

    data_cfg = cfg.data
    model_cfg = cfg.model
    past_w = int(model_cfg.get("past_action_window_size", 1))
    seq_len = int(data_cfg.get("seq_len", 24))
    if seq_len <= past_w:
        raise ValueError(f"seq_len={seq_len} must exceed proprio prefix size={past_w}.")
    processor = build_processor(cfg, dist_ctx=dist_ctx)
    model = create_phi0(cfg, smoke=bool(cfg.get("smoke_action_only", False)))
    sync_model_action_norm(model, processor)
    barrier(dist_ctx)
    model.repeated_action_steps = max(1, int(cfg.get("repeated_action_steps", 1)))
    loader = build_dataloader(cfg, dist_ctx=dist_ctx)
    optim = build_optimizer(model, cfg)
    if dist_ctx.is_main:
        action_lrs = [g["lr"] for g in optim.param_groups if g["lr"] > 0]
        logger.info("Optimizer action LR(s): %s", action_lrs)
    start_step = load_checkpoint_if_configured(model, cfg, optimizer=optim)
    barrier(dist_ctx)
    if dist_ctx.enabled and bool(cfg.get("compile_action_expert", False)):
        if dist_ctx.is_main:
            logger.warning(
                "Skipping torch.compile(action_expert) under DDP; set compile_action_expert=false."
            )
    elif not dist_ctx.enabled:
        maybe_compile_action_expert(model, cfg)
    find_unused = bool(cfg.get("ddp_find_unused_parameters", True))
    train_model = (
        wrap_ddp(model, device_id=dist_ctx.local_rank, find_unused=find_unused)
        if dist_ctx.enabled
        else model
    )
    barrier(dist_ctx)
    device = model.device
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", cuda_device_index(str(cfg.get("device", "cuda:0"))))
    max_steps = int(cfg.get("max_steps", 2))
    if start_step >= max_steps:
        logger.warning(
            "start_step=%d >= max_steps=%d; saving checkpoint and exiting",
            start_step,
            max_steps,
        )
        save_training_checkpoint(model, cfg, start_step, optimizer=optim, dist_ctx=dist_ctx)
        return
    mp = str(cfg.get("mixed_precision", "bf16")).lower()
    use_amp = mp in {"bf16", "fp16"} and device.type == "cuda"
    autocast_dtype = _mixed_precision_to_dtype(mp)
    save_every = int(cfg.get("save_every_steps", 0))
    step = start_step
    per_device_batch = int(cfg.batch_size)
    effective_batch = per_device_batch * max(1, dist_ctx.world_size)
    logger.info(
        "Training steps %d -> %d (per_device_batch=%d effective_batch=%d)",
        start_step,
        max_steps,
        per_device_batch,
        effective_batch,
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    model.train()
    model.set_frozen_towers_eval()
    sampler = getattr(loader, "sampler", None)
    cpu_prefetch = int(cfg.get("train_cpu_prefetch", 0))
    gpu_pipeline = bool(cfg.get("train_gpu_pipeline", False))
    gpu_pipeline_depth = int(cfg.get("train_gpu_pipeline_depth", 2))
    warmup_steps = int(cfg.get("learning_rate_warmup_steps", 0))
    base_lrs = [float(group["lr"]) for group in optim.param_groups]
    scheduler_type = str(cfg.get("learning_rate_scheduler", "none")).strip().lower()
    lr_scheduler = build_lr_scheduler(
        optim,
        scheduler_type=scheduler_type,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        min_lr=float(cfg.get("learning_rate_min", 0.0)),
    )
    tower_streams = get_tower_prep_streams(device)
    if dist_ctx.is_main:
        logger.info(
            "Train pipeline cpu_prefetch=%d gpu_pipeline=%s gpu_depth=%d "
            "warmup_steps=%d scheduler=%s base_lrs=%s compile=%s",
            cpu_prefetch,
            gpu_pipeline,
            gpu_pipeline_depth,
            warmup_steps,
            scheduler_type if lr_scheduler is not None else "manual_warmup_only",
            base_lrs,
            bool(cfg.get("compile_action_expert", False)),
        )

    def _prepare_cpu(batch: Dict[str, Any]) -> Dict[str, Any]:
        return prepare_model_batch_cpu(model, processor, batch)

    def _prepare_gpu(cpu_payload: Dict[str, Any]) -> Dict[str, Any]:
        return prepare_model_batch_gpu(model, processor, cpu_payload)

    from phi0.data.train_prefetch import TrainingBatchIterator

    batch_pipeline = TrainingBatchIterator(
        loader,
        sampler=sampler,
        prepare_cpu=_prepare_cpu,
        prepare_gpu=_prepare_gpu,
        device=device,
        cpu_prefetch=cpu_prefetch,
        gpu_pipeline=gpu_pipeline,
        prep_stream=tower_streams.prep if tower_streams is not None else None,
        gpu_pipeline_depth=gpu_pipeline_depth,
    )

    log_every = int(cfg.get("train_log_every_steps", 1))
    batch_iter = iter(batch_pipeline)
    while step < max_steps:
        try:
            model_batch = next(batch_iter)
        except StopIteration:
            break
        apply_warmup_lr(optim, step, warmup_steps, base_lrs) if lr_scheduler is None else None
        step_t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        step_t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        if step_t0 is not None:
            step_t0.record()
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=use_amp,
        ):
            loss, loss_dict = train_model(model_batch)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        max_norm = cfg.get("gradient_clipping")
        if max_norm is not None and float(max_norm) > 0:
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), float(max_norm))
        optim.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if step_t0 is not None and step_t1 is not None:
            step_t1.record()
            step_t1.synchronize()
            step_ms = step_t0.elapsed_time(step_t1)
        else:
            step_ms = None
        if dist_ctx.is_main and (step % log_every == 0):
            if step_ms is not None:
                logger.info(
                    "step=%d loss=%.4f step_ms=%.0f %s",
                    step,
                    float(loss.item()),
                    step_ms,
                    loss_dict,
                )
            else:
                logger.info("step=%d loss=%.4f %s", step, float(loss.item()), loss_dict)
        del model_batch, loss, loss_dict
        step += 1
        if save_every > 0 and step % save_every == 0:
            save_training_checkpoint(model, cfg, step, optimizer=optim, dist_ctx=dist_ctx)

    if save_every <= 0 or step % save_every != 0:
        save_training_checkpoint(model, cfg, step, optimizer=optim, dist_ctx=dist_ctx)
