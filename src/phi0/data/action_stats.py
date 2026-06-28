"""Action normalization statistics (z-score or VLA-Adapter bounds_q99 over D_raw)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from phi0.data.robot_action_norm import (
    DELTA_TRANSLATE_ROT_DIMS,
    GRIPPER_DIM,
    ROBOT_DIM,
    STATS_SEMANTICS_ABSOLUTE,
    STATS_SEMANTICS_DELTA,
    STATS_SEMANTICS_PROPRIO,
    stats_dict_from_tensors,
    stats_semantics_for_cfg,
    validate_stats_for_cfg,
)
from phi0.schema.draw_schema import D_RAW
from phi0.schema.unified_action_schema import D_UNIFIED, dim_mask_for_dataset

logger = logging.getLogger(__name__)


def masked_unified_action_stats(
    acts: np.ndarray,
    *,
    supervised_mask: np.ndarray | None = None,
) -> dict[str, list[float] | list[int]]:
    """Per-dim min/max/mean/std; unsupervised dims stay mean=0, std=1."""
    acts = np.asarray(acts, dtype=np.float64)
    if acts.ndim != 2 or acts.shape[1] != D_UNIFIED:
        raise ValueError(f"expected acts (T, {D_UNIFIED}), got {acts.shape}")
    mask = (
        np.asarray(supervised_mask, dtype=bool).reshape(D_UNIFIED)
        if supervised_mask is not None
        else dim_mask_for_dataset("g1_sonic")
    )
    n = int(acts.shape[0])
    mean = np.zeros(D_UNIFIED, dtype=np.float64)
    std = np.ones(D_UNIFIED, dtype=np.float64)
    act_min = np.zeros(D_UNIFIED, dtype=np.float64)
    act_max = np.zeros(D_UNIFIED, dtype=np.float64)
    for j in np.flatnonzero(mask):
        col = acts[:, j]
        mean[j] = col.mean()
        std[j] = max(float(col.std(ddof=0)), 1e-12)
        act_min[j] = col.min()
        act_max[j] = col.max()
    std[mask & (std < 0.01)] = 1.0
    return {
        "min": act_min.tolist(),
        "max": act_max.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": [n],
    }


def merge_masked_unified_action_stats(
    episode_stats: Sequence[dict[str, Any]],
    *,
    supervised_mask: np.ndarray | None = None,
) -> dict[str, list[float]] | None:
    """Merge per-episode masked stats; unsupervised dims stay mean=0, std=1."""
    mask = (
        np.asarray(supervised_mask, dtype=bool).reshape(D_UNIFIED)
        if supervised_mask is not None
        else dim_mask_for_dataset("g1_sonic")
    )
    act_sum = np.zeros(D_UNIFIED, dtype=np.float64)
    act_sq = np.zeros(D_UNIFIED, dtype=np.float64)
    act_min = np.full(D_UNIFIED, np.inf)
    act_max = np.full(D_UNIFIED, -np.inf)
    total = 0
    for stats in episode_stats:
        ua = stats["unified_action"]
        count = int(ua["count"][0])
        mean = np.asarray(ua["mean"], dtype=np.float64)
        std = np.asarray(ua["std"], dtype=np.float64)
        for j in np.flatnonzero(mask):
            act_sum[j] += mean[j] * count
            act_sq[j] += (std[j] ** 2 + mean[j] ** 2) * count
            act_min[j] = min(act_min[j], float(ua["min"][j]))
            act_max[j] = max(act_max[j], float(ua["max"][j]))
        total += count
    if total <= 0:
        return None
    act_mean = np.zeros(D_UNIFIED, dtype=np.float64)
    act_std = np.ones(D_UNIFIED, dtype=np.float64)
    act_min_out = np.zeros(D_UNIFIED, dtype=np.float64)
    act_max_out = np.zeros(D_UNIFIED, dtype=np.float64)
    for j in np.flatnonzero(mask):
        act_mean[j] = act_sum[j] / total
        act_std[j] = float(np.sqrt(max(act_sq[j] / total - act_mean[j] ** 2, 1e-12)))
        act_min_out[j] = act_min[j]
        act_max_out[j] = act_max[j]
    act_std[mask & (act_std < 0.01)] = 1.0
    return {
        "mean": act_mean.tolist(),
        "std": act_std.tolist(),
        "q01": act_min_out.tolist(),
        "q99": act_max_out.tolist(),
        "min": act_min_out.tolist(),
        "max": act_max_out.tolist(),
    }


def _frame_action_vector(item: dict, *, field: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]:
    """Return D_raw vector + valid mask for one frame-level dataset item."""
    if field == "robot_proprio_7d" and "robot_proprio_7d" in item:
        raw = item["robot_proprio_7d"]
        if torch.is_tensor(raw):
            x7 = raw.reshape(-1).detach().cpu().numpy().astype(np.float64)
        else:
            x7 = np.asarray(raw, dtype=np.float64).reshape(-1)
        x = np.zeros(D_RAW, dtype=np.float64)
        x[: min(7, D_RAW)] = x7[: min(7, D_RAW)]
        valid = np.zeros(D_RAW, dtype=bool)
        valid[: min(7, D_RAW)] = True
        return x, valid

    if field == "robot_delta_7d" and "robot_delta_7d" in item:
        raw = item["robot_delta_7d"]
        if torch.is_tensor(raw):
            x7 = raw.reshape(-1).detach().cpu().numpy().astype(np.float64)
        else:
            x7 = np.asarray(raw, dtype=np.float64).reshape(-1)
        x = np.zeros(D_RAW, dtype=np.float64)
        x[: min(7, D_RAW)] = x7[: min(7, D_RAW)]
        valid = np.zeros(D_RAW, dtype=bool)
        valid[: min(DELTA_TRANSLATE_ROT_DIMS, D_RAW)] = True
        return x, valid

    if "robot_delta_7d" in item and field != "robot_proprio_7d":
        raw = item["robot_delta_7d"]
        if torch.is_tensor(raw):
            x7 = raw.reshape(-1).detach().cpu().numpy().astype(np.float64)
        else:
            x7 = np.asarray(raw, dtype=np.float64).reshape(-1)
        x = np.zeros(D_RAW, dtype=np.float64)
        x[: min(7, D_RAW)] = x7[: min(7, D_RAW)]
        valid = np.zeros(D_RAW, dtype=bool)
        # VLA-Adapter: delta dims 0-5 normalized; gripper (6) kept raw in [0,1].
        valid[: min(DELTA_TRANSLATE_ROT_DIMS, D_RAW)] = True
        return x, valid

    if "robot_proprio_7d" in item:
        raw = item["robot_proprio_7d"]
        if torch.is_tensor(raw):
            x7 = raw.reshape(-1).detach().cpu().numpy().astype(np.float64)
        else:
            x7 = np.asarray(raw, dtype=np.float64).reshape(-1)
        x = np.zeros(D_RAW, dtype=np.float64)
        x[: min(7, D_RAW)] = x7[: min(7, D_RAW)]
        valid = np.zeros(D_RAW, dtype=bool)
        valid[: min(7, D_RAW)] = True
        return x, valid

    if "robot_action_7d" in item:
        raw = item["robot_action_7d"]
        if torch.is_tensor(raw):
            x7 = raw.reshape(-1).detach().cpu().numpy().astype(np.float64)
        else:
            x7 = np.asarray(raw, dtype=np.float64).reshape(-1)
        x = np.zeros(D_RAW, dtype=np.float64)
        x[: min(7, D_RAW)] = x7[: min(7, D_RAW)]
        valid = np.zeros(D_RAW, dtype=bool)
        valid[: min(7, D_RAW)] = True
        return x, valid

    action = item["action"]
    if torch.is_tensor(action):
        x = action.reshape(-1).detach().cpu().numpy().astype(np.float64)
    else:
        x = np.asarray(action, dtype=np.float64).reshape(-1)
    pad = item["action_dim_is_pad"]
    if torch.is_tensor(pad):
        valid = (~pad).reshape(-1).cpu().numpy()
    else:
        valid = (~np.asarray(pad, dtype=bool)).reshape(-1)
    return x, valid


def _online_update(
    x: np.ndarray,
    valid: np.ndarray,
    count: np.ndarray,
    mean: np.ndarray,
    m2: np.ndarray,
) -> None:
    """Welford update for one sample vector (per-dimension, masked)."""
    for j in range(x.shape[0]):
        if not valid[j]:
            continue
        count[j] += 1
        delta = float(x[j]) - mean[j]
        mean[j] += delta / count[j]
        delta2 = float(x[j]) - mean[j]
        m2[j] += delta * delta2


def _iter_dataset_frames(
    datasets: Sequence[Dataset],
    *,
    field: Optional[str] = None,
    show_progress: bool = False,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    for ds in datasets:
        ds_name = getattr(ds, "DATASET_NAME", type(ds).__name__)
        n = len(ds)
        frame_iter: Iterable[int] = range(n)
        if show_progress:
            try:
                from tqdm import tqdm

                frame_iter = tqdm(
                    frame_iter,
                    desc=f"stats/{ds_name}",
                    unit="frame",
                    total=n,
                    leave=True,
                )
            except ImportError:
                pass
        for i in frame_iter:
            yield _frame_action_vector(ds[i], field=field)


def compute_action_stats_from_datasets(
    datasets: Sequence[Dataset],
    *,
    robot_action_semantics: str = STATS_SEMANTICS_ABSOLUTE,
    norm_mode: str = "z-score",
    normalize_gripper: bool = True,
    stats_field: Optional[str] = None,
    show_progress: bool = False,
    action_dim: int = D_RAW,
) -> Dict[str, Any]:
    """Scan frame-level datasets and return per-dim stats for supervised dims."""
    count = np.zeros(action_dim, dtype=np.int64)
    mean = np.zeros(action_dim, dtype=np.float64)
    m2 = np.zeros(action_dim, dtype=np.float64)
    percentile_buckets: List[List[float]] = [[] for _ in range(action_dim)]
    n_frames = 0
    norm_key = str(norm_mode).strip().lower()

    for x, valid in _iter_dataset_frames(datasets, field=stats_field, show_progress=show_progress):
        _online_update(x, valid, count, mean, m2)
        if norm_key == "bounds_q99":
            for j in range(action_dim):
                if valid[j]:
                    percentile_buckets[j].append(float(x[j]))
        n_frames += 1

    std = np.sqrt(m2 / np.maximum(count, 1))
    unsupervised = count == 0
    std[count < 2] = 1.0
    std[std < 1e-6] = 1.0
    mean[unsupervised] = 0.0
    std[unsupervised] = 1.0

    q01 = mean.copy()
    q99 = mean.copy()
    if norm_key == "bounds_q99":
        for j in range(action_dim):
            if percentile_buckets[j]:
                arr = np.asarray(percentile_buckets[j], dtype=np.float64)
                q01[j] = float(np.percentile(arr, 1))
                q99[j] = float(np.percentile(arr, 99))
                if abs(q99[j] - q01[j]) < 1e-6:
                    q99[j] = q01[j] + 1e-3

    supervised = (~unsupervised).tolist()
    if not normalize_gripper:
        for j in range(min(ROBOT_DIM, action_dim)):
            if j == GRIPPER_DIM:
                supervised[j] = False

    return stats_dict_from_tensors(
        mean=torch.tensor(mean, dtype=torch.float32),
        std=torch.tensor(std, dtype=torch.float32),
        q01=torch.tensor(q01, dtype=torch.float32),
        q99=torch.tensor(q99, dtype=torch.float32),
        robot_action_semantics=robot_action_semantics,
        norm_mode=norm_key,
        num_frames=n_frames,
        normalize_gripper=normalize_gripper,
        supervised_mask=supervised,
        action_dim=action_dim,
    )


def compute_action_stats_for_data_cfg(
    datasets: Sequence[Dataset],
    data_cfg: Any,
    *,
    proprio: bool = False,
    show_progress: bool = False,
) -> Dict[str, Any]:
    semantics = stats_semantics_for_cfg(data_cfg, proprio=proprio)
    if proprio:
        norm_mode = str(data_cfg.get("proprio_norm_mode", "z-score")).strip().lower()
        normalize_gripper = True
        stats_field = "robot_proprio_7d"
    elif bool(data_cfg.get("libero_delta_eef", False)):
        norm_mode = str(data_cfg.get("action_norm_mode", "bounds_q99")).strip().lower()
        normalize_gripper = False
        stats_field = "robot_delta_7d"
    else:
        norm_mode = str(data_cfg.get("action_norm_mode", "z-score")).strip().lower()
        normalize_gripper = True
        stats_field = None
    action_dim = int(data_cfg.get("action_dim", 0)) or (
        D_UNIFIED if str(data_cfg.get("xperience_action_rep", "keypoints")).lower() == "unified" else D_RAW
    )
    return compute_action_stats_from_datasets(
        datasets,
        robot_action_semantics=semantics,
        norm_mode=norm_mode,
        normalize_gripper=normalize_gripper,
        stats_field=stats_field,
        show_progress=show_progress,
        action_dim=action_dim,
    )


def save_action_stats(stats: Dict[str, Any], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return out


def load_action_stats(path: Path | str) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Action stats not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def stats_to_tensors(
    stats: Dict[str, Any],
    action_dim: int = D_RAW,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)
    q01 = torch.tensor(stats.get("q01", stats["mean"]), dtype=torch.float32)
    q99 = torch.tensor(stats.get("q99", stats["mean"]), dtype=torch.float32)
    if mean.numel() != action_dim or std.numel() != action_dim:
        raise ValueError(f"Stats dim {mean.numel()} != action_dim {action_dim}")
    return mean, std.clamp(min=1e-6), q01, q99


def resolve_action_stats_path(cfg_data: Any, output_dir: Optional[Path] = None) -> Optional[Path]:
    raw = cfg_data.get("action_stats_path") if cfg_data is not None else None
    if raw is None or str(raw).lower() in {"", "null", "none"}:
        if output_dir is not None:
            return Path(output_dir) / "action_stats.json"
        return None
    return Path(str(raw))


def resolve_proprio_stats_path(cfg_data: Any, output_dir: Optional[Path] = None) -> Optional[Path]:
    raw = cfg_data.get("proprio_stats_path") if cfg_data is not None else None
    if raw is None or str(raw).lower() in {"", "null", "none"}:
        return resolve_action_stats_path(cfg_data, output_dir=output_dir)
    return Path(str(raw))


def load_or_validate_stats(
    path: Path,
    data_cfg: Any,
    *,
    proprio: bool = False,
) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    stats = load_action_stats(path)
    validate_stats_for_cfg(data_cfg, stats, proprio=proprio)
    return stats
