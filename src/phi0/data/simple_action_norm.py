"""G1 SIMPLE whole-body action normalize/denormalize (36-dim bounds)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

SIMPLE_G1_DIM = 36
STATS_SEMANTICS_SIMPLE_G1 = "simple_g1_wholebody_36d"


def load_simple_stats_json(path: str | Path) -> dict[str, Any]:
    """Load Psi0/SIMPLE LeRobot stats (meta/stats_psi0.json)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    action = raw.get("action") or raw.get("actions") or {}
    state = raw.get("states") or raw.get("state") or {}
    norm_mode = "bounds"
    if "q01" in action and "q99" in action:
        norm_mode = "bounds_q99"
    return {
        "version": 2,
        "robot_action_semantics": STATS_SEMANTICS_SIMPLE_G1,
        "norm_mode": norm_mode,
        "normalize_gripper": True,
        "robot_dim": SIMPLE_G1_DIM,
        "action_dim": SIMPLE_G1_DIM,
        "num_frames": int(raw.get("num_frames", 0)),
        "mean": action.get("mean", [0.0] * SIMPLE_G1_DIM),
        "std": action.get("std", [1.0] * SIMPLE_G1_DIM),
        "q01": action.get("q01", action.get("min", [0.0] * SIMPLE_G1_DIM)),
        "q99": action.get("q99", action.get("max", [1.0] * SIMPLE_G1_DIM)),
        "state_mean": state.get("mean", [0.0] * SIMPLE_G1_DIM),
        "state_std": state.get("std", [1.0] * SIMPLE_G1_DIM),
        "state_q01": state.get("q01", state.get("min", [0.0] * SIMPLE_G1_DIM)),
        "state_q99": state.get("q99", state.get("max", [1.0] * SIMPLE_G1_DIM)),
    }


def _stats_vectors(
    stats: Mapping[str, Any],
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    proprio: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prefix = "state_" if proprio else ""
    mean = torch.tensor(stats.get(f"{prefix}mean", stats["mean"]), device=device, dtype=dtype)
    std = torch.tensor(stats.get(f"{prefix}std", stats["std"]), device=device, dtype=dtype).clamp(
        min=1e-6
    )
    q01 = torch.tensor(stats.get(f"{prefix}q01", stats.get("q01", mean)), device=device, dtype=dtype)
    q99 = torch.tensor(stats.get(f"{prefix}q99", stats.get("q99", mean)), device=device, dtype=dtype)
    return mean[:dim], std[:dim], q01[:dim], q99[:dim]


def normalize_robot_nd(
    action: torch.Tensor,
    stats: Mapping[str, Any],
    *,
    dim: int = SIMPLE_G1_DIM,
    proprio: bool = False,
) -> torch.Tensor:
    if action.shape[-1] != dim:
        raise ValueError(f"expected last dim {dim}, got {action.shape[-1]}")
    norm_mode = str(stats.get("norm_mode", "bounds")).strip().lower()
    mean, std, q01, q99 = _stats_vectors(stats, dim, action.device, action.dtype, proprio=proprio)
    if norm_mode in {"bounds", "bounds_q99"}:
        span = (q99 - q01).clamp(min=1e-6)
        return (2.0 * (action - q01) / span - 1.0).clamp(-1.0, 1.0)
    return (action - mean) / std


def denormalize_robot_nd(
    action_norm: torch.Tensor,
    stats: Mapping[str, Any],
    *,
    dim: int = SIMPLE_G1_DIM,
    proprio: bool = False,
) -> torch.Tensor:
    if action_norm.shape[-1] != dim:
        raise ValueError(f"expected last dim {dim}, got {action_norm.shape[-1]}")
    norm_mode = str(stats.get("norm_mode", "bounds")).strip().lower()
    mean, std, q01, q99 = _stats_vectors(
        stats, dim, action_norm.device, action_norm.dtype, proprio=proprio
    )
    if norm_mode in {"bounds", "bounds_q99"}:
        span = (q99 - q01).clamp(min=1e-6)
        return 0.5 * (action_norm + 1.0) * span + q01
    return action_norm * std + mean


def stats_view_for_robot_nd(processor, *, proprio: bool = False, dim: int = SIMPLE_G1_DIM) -> dict[str, Any]:
    if proprio and hasattr(processor, "proprio_mean"):
        mean, std = processor.proprio_mean, processor.proprio_std
        q01 = getattr(processor, "proprio_q01", mean)
        q99 = getattr(processor, "proprio_q99", mean)
        norm_mode = getattr(processor, "proprio_norm_mode", "bounds")
    else:
        mean, std = processor.mean, processor.std
        q01 = getattr(processor, "action_q01", mean)
        q99 = getattr(processor, "action_q99", mean)
        norm_mode = getattr(processor, "action_norm_mode", "bounds")
    return {
        "norm_mode": norm_mode,
        "mean": mean.detach().cpu().tolist()[:dim],
        "std": std.detach().cpu().tolist()[:dim],
        "q01": q01.detach().cpu().tolist()[:dim],
        "q99": q99.detach().cpu().tolist()[:dim],
    }
