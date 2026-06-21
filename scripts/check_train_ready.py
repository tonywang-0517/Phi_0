#!/usr/bin/env python3
"""Pre-flight checks before LIBERO VLM + dual-tower Phi_0 training."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from phi0.data.sequence import sequence_dataset_from_cfg
from phi0.runtime import build_base_dataset, build_processor, create_phi0

logger = logging.getLogger(__name__)


def _resolve_vlm_path(model_cfg, *, root: Path) -> Path:
    vlm_cfg = model_cfg.get("vlm") or {}
    raw = vlm_cfg.get("model_path") or model_cfg.get("vlm_model_path")
    if raw is None:
        raise ValueError("model.vlm.model_path is required.")
    path = Path(str(raw))
    if not path.is_absolute():
        path = (root / path).resolve()
    return path


def _check_vlm_checkpoint(model_cfg, *, root: Path) -> None:
    path = _resolve_vlm_path(model_cfg, root=root)
    if not path.is_dir():
        raise FileNotFoundError(f"VLM checkpoint dir not found: {path}")
    weights = path / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"VLM weights missing: {weights}")
    for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"VLM checkpoint incomplete (missing {name}): {path}")
    vlm_cfg = model_cfg.get("vlm") or {}
    size = list(vlm_cfg.get("image_size", [180, 320]))
    if size != [180, 320]:
        logger.warning(
            "VLM image_size=%s; Psi0 finetune-simple uses [180, 320] (H×W).",
            size,
        )
    logger.info("VLM checkpoint OK: %s", path)


def _check_loss_and_freeze(model_cfg) -> None:
    lam_a = float(model_cfg.loss.get("lambda_action", 0.0))
    if lam_a <= 0:
        raise ValueError("lambda_action must be > 0 for action training.")
    if not bool(model_cfg.get("vlm", {}).get("freeze", model_cfg.get("freeze_vlm", True))):
        raise ValueError("Expected frozen VLM (vlm.freeze=true) for action-head-only training.")
    if bool(model_cfg.get("vggt", {}).get("enabled", False)):
        if not bool(model_cfg.get("vggt", {}).get("freeze", True)):
            raise ValueError("Expected frozen VGGT (vggt.freeze=true).")


def _check_seq_layout(data_cfg, *, past_w: int) -> None:
    seq_len = int(data_cfg.get("seq_len", 0))
    future = int(data_cfg.get("future_action_steps", 0) or max(0, seq_len - past_w))
    if seq_len != past_w + future:
        raise ValueError(
            f"seq_len={seq_len} should equal past_action_window_size({past_w}) + "
            f"future_action_steps({future})."
        )
    logger.info(
        "Sequence layout OK: seq_len=%d = proprio(%d) + future(%d)",
        seq_len,
        past_w,
        future,
    )


def _check_sample_clip(cfg: DictConfig, processor, *, past_w: int) -> None:
    base = build_base_dataset(cfg)
    seq = sequence_dataset_from_cfg(base, cfg.data)
    if len(seq) == 0:
        raise RuntimeError("SequenceDataset is empty; check RLDS path / libero_max_* limits.")
    batch = seq.collate_fn([seq[0]])
    sample = processor.preprocess(batch)
    action_t = int(sample["action"].shape[1])
    if action_t != int(cfg.data.get("seq_len", action_t)):
        raise ValueError(f"preprocessed action T={action_t} != data.seq_len")
    if action_t <= past_w:
        raise ValueError(f"clip action T={action_t} must exceed proprio prefix={past_w}.")
    logger.info(
        "Sample clip OK: action T=%d (prefix=%d, future=%d), vlm_size=%s, img_aug=%s",
        action_t,
        past_w,
        action_t - past_w,
        processor.vlm_image_size,
        processor.vlm_img_aug,
    )


def _check_trainable_scope(model) -> None:
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen_prefixes = ("vlm_tower.", "vggt_tower.")
    for name in trainable:
        if name.startswith(frozen_prefixes):
            raise ValueError(f"Frozen tower param is trainable: {name}")
    if not any(n.startswith("action_expert.") for n in trainable):
        raise ValueError("No action_expert parameters are trainable.")
    logger.info(
        "Trainable scope OK: %d tensors under action_expert (VLM+VGGT frozen).",
        len(trainable),
    )


def run_checks(cfg: DictConfig, *, smoke_model: bool = True, load_vlm: bool = False) -> None:
    data_cfg = cfg.data
    model_cfg = cfg.model
    root = Path(str(cfg.get("checkpoints_dir", ROOT / "checkpoints")))
    if not root.is_absolute():
        root = (ROOT / root).resolve()

    past_w = int(model_cfg.get("past_action_window_size", 1))
    _check_vlm_checkpoint(model_cfg, root=ROOT)
    _check_loss_and_freeze(model_cfg)
    _check_seq_layout(data_cfg, past_w=past_w)

    processor = build_processor(cfg)
    _check_sample_clip(cfg, processor, past_w=past_w)

    noise_std = float(model_cfg.get("action_dit_config", {}).get("future_placeholder_noise_std", 0.02))
    logger.info("ACT future placeholder: inference=zeros, training=zeros+N(0, %.4f)", noise_std)

    if smoke_model and not load_vlm:
        model = create_phi0(cfg, smoke=True)
    else:
        model = create_phi0(cfg, smoke=False)
        _check_trainable_scope(model)

    if int(getattr(model, "past_action_window_size", 0)) != past_w:
        raise ValueError("Model past_action_window_size mismatch vs config.")
    expert = model.action_expert
    if int(getattr(expert, "proprio_window", 0)) != past_w:
        raise ValueError(
            f"action_expert.proprio_window={getattr(expert, 'proprio_window', None)} "
            f"!= past_action_window_size={past_w}"
        )
    logger.info("Model build OK (past_action_window_size=%d).", past_w)


@hydra.main(config_path="../configs", config_name="train_libero_spatial_act_delta_300", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    load_vlm = bool(cfg.get("check_load_vlm", True))
    smoke = bool(cfg.get("check_smoke_only", False))
    run_checks(cfg, smoke_model=smoke, load_vlm=load_vlm)
    logger.info("All pre-flight checks passed.")


if __name__ == "__main__":
    main()
