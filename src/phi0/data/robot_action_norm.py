"""7D robot action normalize/denormalize (VLA-Adapter LIBERO conventions)."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

ROBOT_DIM = 7
GRIPPER_DIM = 6
DELTA_TRANSLATE_ROT_DIMS = 6
STATS_SEMANTICS_DELTA = "libero_delta_eef_6d"
STATS_SEMANTICS_ABSOLUTE = "libero_absolute_eef_7d"
STATS_SEMANTICS_PROPRIO = "libero_proprio_absolute_eef_7d"


def _cfg_get(data_cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(data_cfg, Mapping):
        return data_cfg.get(key, default)
    return getattr(data_cfg, key, default)


def stats_semantics_for_cfg(data_cfg: Mapping[str, Any], *, proprio: bool = False) -> str:
    if proprio:
        return STATS_SEMANTICS_PROPRIO
    if bool(_cfg_get(data_cfg, "libero_delta_eef", False)):
        return STATS_SEMANTICS_DELTA
    if bool(_cfg_get(data_cfg, "libero_absolute_eef", True)):
        return STATS_SEMANTICS_ABSOLUTE
    return STATS_SEMANTICS_ABSOLUTE


def validate_stats_for_cfg(data_cfg: Mapping[str, Any], stats: Mapping[str, Any], *, proprio: bool = False) -> None:
    version = int(stats.get("version", 1))
    semantics = str(stats.get("robot_action_semantics", "")).strip()
    if version < 2 or not semantics:
        legacy_ok = (
            proprio
            and version == 1
            and str(stats.get("norm_mode", "z-score")).strip().lower() == "z-score"
        )
        if not legacy_ok:
            raise ValueError(
                "Action stats file is legacy (missing version>=2 or robot_action_semantics). "
                "Delete and recompute for delta-EEF training."
            )
    expected = stats_semantics_for_cfg(data_cfg, proprio=proprio)
    if semantics and semantics != expected:
        raise ValueError(
            f"Action stats semantics mismatch: expected {expected!r}, got {semantics!r}. "
            "Delete the stats file and recompute (delta requires libero_spatial_delta_action_stats.json)."
        )
    norm_mode = str(stats.get("norm_mode", "z-score")).strip().lower()
    if proprio:
        expected_mode = str(_cfg_get(data_cfg, "proprio_norm_mode", "z-score")).strip().lower()
    elif bool(_cfg_get(data_cfg, "libero_delta_eef", False)):
        expected_mode = str(_cfg_get(data_cfg, "action_norm_mode", "bounds_q99")).strip().lower()
    else:
        expected_mode = str(_cfg_get(data_cfg, "action_norm_mode", "z-score")).strip().lower()
    if norm_mode != expected_mode:
        raise ValueError(
            f"Action stats norm_mode mismatch: expected {expected_mode!r}, got {norm_mode!r}."
        )


def _stats_vectors(stats: Mapping[str, Any], device: torch.device, dtype: torch.dtype) -> tuple:
    mean = torch.tensor(stats["mean"], device=device, dtype=dtype)
    std = torch.tensor(stats["std"], device=device, dtype=dtype).clamp(min=1e-6)
    q01 = torch.tensor(stats.get("q01", stats["mean"]), device=device, dtype=dtype)
    q99 = torch.tensor(stats.get("q99", stats["mean"]), device=device, dtype=dtype)
    return mean, std, q01, q99


def normalize_robot7d(
    action_7d: torch.Tensor,
    stats: Mapping[str, Any],
    *,
    normalize_gripper: bool = True,
    normalize_dims: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalize ``[..., 7]`` robot controls."""
    if action_7d.shape[-1] != ROBOT_DIM:
        raise ValueError(f"expected last dim {ROBOT_DIM}, got {action_7d.shape[-1]}")
    norm_mode = str(stats.get("norm_mode", "z-score")).strip().lower()
    mean, std, q01, q99 = _stats_vectors(stats, action_7d.device, action_7d.dtype)
    m = mean[:ROBOT_DIM]
    s = std[:ROBOT_DIM]
    lo = q01[:ROBOT_DIM]
    hi = q99[:ROBOT_DIM]

    if norm_mode == "bounds_q99":
        span = (hi - lo).clamp(min=1e-6)
        out = (2.0 * (action_7d - lo) / span - 1.0).clamp(-1.0, 1.0)
    else:
        out = (action_7d - m) / s

    if normalize_dims is not None:
        mask = normalize_dims.to(device=out.device, dtype=torch.bool)
        if mask.shape[-1] != ROBOT_DIM:
            raise ValueError(f"normalize_dims last dim must be {ROBOT_DIM}")
        keep = (~mask).to(dtype=out.dtype)
        out = out * mask.to(dtype=out.dtype) + action_7d * keep

    if not normalize_gripper:
        out = out.clone()
        out[..., GRIPPER_DIM] = action_7d[..., GRIPPER_DIM]
    return out


def denormalize_robot7d(
    action_7d_norm: torch.Tensor,
    stats: Mapping[str, Any],
    *,
    denormalize_gripper: bool = True,
    normalize_dims: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Inverse of :func:`normalize_robot7d` for ``[..., 7]``."""
    if action_7d_norm.shape[-1] != ROBOT_DIM:
        raise ValueError(f"expected last dim {ROBOT_DIM}, got {action_7d_norm.shape[-1]}")
    norm_mode = str(stats.get("norm_mode", "z-score")).strip().lower()
    mean, std, q01, q99 = _stats_vectors(stats, action_7d_norm.device, action_7d_norm.dtype)
    m = mean[:ROBOT_DIM]
    s = std[:ROBOT_DIM]
    lo = q01[:ROBOT_DIM]
    hi = q99[:ROBOT_DIM]

    if norm_mode == "bounds_q99":
        span = (hi - lo).clamp(min=1e-6)
        out = 0.5 * (action_7d_norm + 1.0) * span + lo
    else:
        out = action_7d_norm * s + m

    if normalize_dims is not None:
        mask = normalize_dims.to(device=out.device, dtype=torch.bool)
        keep = (~mask).to(dtype=out.dtype)
        out = out * mask.to(dtype=out.dtype) + action_7d_norm * keep

    if not denormalize_gripper:
        out = out.clone()
        out[..., GRIPPER_DIM] = action_7d_norm[..., GRIPPER_DIM]
    return out


def stats_dict_from_tensors(
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    robot_action_semantics: str,
    norm_mode: str,
    num_frames: int,
    q01: Optional[torch.Tensor] = None,
    q99: Optional[torch.Tensor] = None,
    normalize_gripper: bool = True,
    supervised_mask: Optional[list[bool]] = None,
) -> dict[str, Any]:
    from phi0.schema.draw_schema import D_RAW

    if supervised_mask is None:
        supervised = (std > 1e-8).tolist()
    else:
        supervised = supervised_mask
    out: dict[str, Any] = {
        "version": 2,
        "robot_action_semantics": robot_action_semantics,
        "norm_mode": norm_mode,
        "normalize_gripper": normalize_gripper,
        "robot_dim": ROBOT_DIM,
        "action_dim": D_RAW,
        "num_frames": num_frames,
        "supervised_mask": supervised,
        "mean": mean.detach().cpu().float().tolist(),
        "std": std.detach().cpu().float().clamp(min=1e-6).tolist(),
    }
    if q01 is not None and q99 is not None:
        out["q01"] = q01.detach().cpu().float().tolist()
        out["q99"] = q99.detach().cpu().float().tolist()
    return out


def processor_stats_dict(processor) -> dict[str, Any]:
    """Serialize processor mean/std (+ optional q01/q99) for checkpointing."""
    from phi0.schema.draw_schema import D_RAW

    out: dict[str, Any] = {
        "version": 2,
        "action_dim": D_RAW,
        "norm_mode": getattr(processor, "action_norm_mode", "z-score"),
        "mean": processor.mean.detach().cpu().tolist(),
        "std": processor.std.detach().cpu().tolist(),
    }
    if hasattr(processor, "action_q01"):
        out["q01"] = processor.action_q01.detach().cpu().tolist()
        out["q99"] = processor.action_q99.detach().cpu().tolist()
    if hasattr(processor, "robot_action_semantics"):
        out["robot_action_semantics"] = processor.robot_action_semantics
    return out


def stats_view_for_robot7d(processor, *, proprio: bool = False) -> dict[str, Any]:
    """Build a stats mapping for :func:`normalize_robot7d` from a processor."""
    if proprio and hasattr(processor, "proprio_mean"):
        mean, std = processor.proprio_mean, processor.proprio_std
        q01 = getattr(processor, "proprio_q01", mean)
        q99 = getattr(processor, "proprio_q99", mean)
        norm_mode = getattr(processor, "proprio_norm_mode", "z-score")
    else:
        mean, std = processor.mean, processor.std
        q01 = getattr(processor, "action_q01", mean)
        q99 = getattr(processor, "action_q99", mean)
        norm_mode = getattr(processor, "action_norm_mode", "z-score")
    return {
        "norm_mode": norm_mode,
        "mean": mean.detach().cpu().tolist(),
        "std": std.detach().cpu().tolist(),
        "q01": q01.detach().cpu().tolist(),
        "q99": q99.detach().cpu().tolist(),
    }
