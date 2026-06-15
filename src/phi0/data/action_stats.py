"""Action normalization statistics (FastWAM / DiT4DiT-style z-score over D_raw)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from phi0.schema.draw_schema import D_RAW


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


def compute_action_stats_from_datasets(
    datasets: Sequence[Dataset],
    *,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """Scan frame-level datasets and return per-dim mean/std (supervised dims only)."""
    count = np.zeros(D_RAW, dtype=np.int64)
    mean = np.zeros(D_RAW, dtype=np.float64)
    m2 = np.zeros(D_RAW, dtype=np.float64)
    n_frames = 0

    for ds in datasets:
        ds_name = getattr(ds, "DATASET_NAME", type(ds).__name__)
        n = len(ds)
        frame_iter = range(n)
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
            item = ds[i]
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
            _online_update(x, valid, count, mean, m2)
            n_frames += 1

    std = np.sqrt(m2 / np.maximum(count, 1))
    unsupervised = count == 0
    std[count < 2] = 1.0
    std[std < 1e-6] = 1.0
    mean[unsupervised] = 0.0
    std[unsupervised] = 1.0

    supervised = (~unsupervised).tolist()
    return {
        "version": 1,
        "action_dim": D_RAW,
        "norm_mode": "z-score",
        "num_frames": n_frames,
        "supervised_mask": supervised,
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "count_per_dim": count.astype(np.int64).tolist(),
    }


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


def stats_to_tensors(stats: Dict[str, Any], action_dim: int = D_RAW) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)
    if mean.numel() != action_dim or std.numel() != action_dim:
        raise ValueError(f"Stats dim {mean.numel()} != action_dim {action_dim}")
    return mean, std.clamp(min=1e-6)


def resolve_action_stats_path(cfg_data: Any, output_dir: Optional[Path] = None) -> Optional[Path]:
    raw = cfg_data.get("action_stats_path") if cfg_data is not None else None
    if raw is None or str(raw).lower() in {"", "null", "none"}:
        if output_dir is not None:
            return Path(output_dir) / "action_stats.json"
        return None
    return Path(str(raw))
