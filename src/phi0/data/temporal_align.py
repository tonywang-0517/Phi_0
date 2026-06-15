"""Unified control timeline: resample native datasets + DiT4DiT-style video subsampling."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_DATASET_NATIVE_FPS: Dict[str, float] = {
    "xperience": 20.0,
    "egodex": 30.0,
}


def native_span_frames(seq_len: int, control_fps: float, native_fps: float) -> int:
    """Native frames needed to cover ``seq_len`` control steps at ``control_fps``."""
    if seq_len <= 1:
        return 1
    if control_fps <= 0 or native_fps <= 0:
        raise ValueError(f"fps must be positive, got control={control_fps} native={native_fps}")
    return int(round((seq_len - 1) * native_fps / control_fps)) + 1


def max_native_span_frames(
    seq_len: int,
    control_fps: float,
    native_fps_map: Mapping[str, float] | None = None,
) -> int:
    fps_map = native_fps_map or DEFAULT_DATASET_NATIVE_FPS
    return max(native_span_frames(seq_len, control_fps, fps) for fps in fps_map.values())


def control_to_native_indices(src_len: int, dst_len: int) -> np.ndarray:
    """Map each control step to a native source index (inclusive endpoints)."""
    if dst_len <= 0:
        return np.zeros(0, dtype=np.int64)
    if src_len <= 1 or dst_len == 1:
        return np.zeros(dst_len, dtype=np.int64)
    return np.round(np.linspace(0, src_len - 1, dst_len)).astype(np.int64)


def video_sample_control_indices(seq_len: int, action_video_freq_ratio: int) -> List[int]:
    """DiT4DiT-style pixel subsample on the control timeline (0, ratio, 2*ratio, ...)."""
    ratio = max(1, int(action_video_freq_ratio))
    return list(range(0, int(seq_len), ratio))


def resample_action_sequence(action: torch.Tensor, src_len: int, dst_len: int) -> torch.Tensor:
    """Linear resample ``[src_len, D]`` -> ``[dst_len, D]``."""
    if src_len == dst_len:
        return action
    if src_len <= 0 or dst_len <= 0:
        raise ValueError(f"invalid resample lengths src={src_len} dst={dst_len}")
    x = action.reshape(src_len, -1).unsqueeze(0).permute(0, 2, 1).float()
    y = F.interpolate(x, size=dst_len, mode="linear", align_corners=True)
    return y.squeeze(0).permute(1, 0).to(dtype=action.dtype, device=action.device)


def resample_bool_sequence(flags: torch.Tensor, src_len: int, dst_len: int) -> torch.Tensor:
    """Nearest-neighbor resample for pad / validity flags ``[src_len, ...]``."""
    if src_len == dst_len:
        return flags
    idx = control_to_native_indices(src_len, dst_len)
    if flags.ndim == 1:
        return flags[idx]
    return flags[idx]


def resample_image_sequence(images: torch.Tensor, src_len: int, dst_len: int) -> torch.Tensor:
    """Temporal linear resample ``[src_len, C, H, W]`` -> ``[dst_len, C, H, W]``."""
    if src_len == dst_len:
        return images
    x = images.reshape(src_len, -1).unsqueeze(0).permute(0, 2, 1).float()
    y = F.interpolate(x, size=dst_len, mode="linear", align_corners=True)
    c, h, w = images.shape[1:]
    return y.squeeze(0).permute(1, 0).reshape(dst_len, c, h, w).to(
        dtype=images.dtype, device=images.device
    )


def resolve_native_fps(dataset_name: str, native_fps_map: Mapping[str, float]) -> float:
    if dataset_name not in native_fps_map:
        raise KeyError(f"Unknown dataset {dataset_name!r}; set data.dataset_native_fps")
    return float(native_fps_map[dataset_name])
