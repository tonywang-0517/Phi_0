"""Training-aligned deploy: 58-step control clip + subsampled video + GT history."""

from __future__ import annotations

from typing import Callable, List, Sequence

import torch

from phi0.data.temporal_align import (
    deploy_v2w_cond_video_control_indices,
    video_sample_control_indices,
)
from phi0.models.action_history import DEFAULT_ACTION_HISTORY_WINDOW


def control_step_to_native_frame(
    control_t: int,
    start_frame: int,
    deploy_fps: float,
    native_fps: float,
) -> int:
    return int(start_frame) + int(round(int(control_t) * float(native_fps) / float(deploy_fps)))


def deploy_clip_start(seg_start: int, history_window: int = DEFAULT_ACTION_HISTORY_WINDOW) -> int:
    """Control index where a training-style clip begins (history ends at ``seg_start``)."""
    return max(0, int(seg_start) - (int(history_window) - 1))


def deploy_control_clip_indices(
    seg_start: int,
    seq_len: int = 58,
    *,
    history_window: int = DEFAULT_ACTION_HISTORY_WINDOW,
) -> List[int]:
    """``seq_len`` control steps forward from clip start (matches ``SequenceDataset``)."""
    clip_start = deploy_clip_start(seg_start, history_window)
    return [clip_start + i for i in range(int(seq_len))]


def deploy_history_control_indices(
    seg_start: int,
    history_window: int = DEFAULT_ACTION_HISTORY_WINDOW,
) -> List[int]:
    """History = first ``history_window`` steps of the clip (training ``split_history_future``)."""
    clip_start = deploy_clip_start(seg_start, history_window)
    return [clip_start + i for i in range(int(history_window))]


def deploy_proprio_control_indices(
    seg_start: int,
    past_window: int = DEFAULT_ACTION_HISTORY_WINDOW,
) -> List[int]:
    """Deprecated alias for ``deploy_history_control_indices``."""
    return deploy_history_control_indices(seg_start, history_window=past_window)


def deploy_subsampled_video_control_indices(
    seg_start: int,
    *,
    seq_len: int = 58,
    action_video_freq_ratio: int = 2,
    history_window: int = DEFAULT_ACTION_HISTORY_WINDOW,
) -> List[int]:
    """Subsampled video control indices (training ``video_control_indices`` on deploy timeline)."""
    clip = deploy_control_clip_indices(seg_start, seq_len, history_window=history_window)
    rel = video_sample_control_indices(seq_len, action_video_freq_ratio)
    return [clip[i] for i in rel]


def deploy_past_subsampled_video_control_indices(
    current_step: int,
    *,
    control_fps: float = 20.0,
    video_history_seconds: float = 1.0,
    action_video_freq_ratio: int = 2,
    cond_pixel_frames: int | None = None,
) -> List[int]:
    """Past-only subsampled indices ending at ``current_step``.

    When ``cond_pixel_frames`` is set (default: official V2W = 5), uses the official
    multi-frame cond window instead of ``video_history_seconds``.
    """
    if cond_pixel_frames is not None:
        return deploy_v2w_cond_video_control_indices(
            int(current_step),
            cond_pixel_frames=int(cond_pixel_frames),
            action_video_freq_ratio=int(action_video_freq_ratio),
        )
    span = max(1, int(round(float(control_fps) * float(video_history_seconds))))
    clip_start = max(0, int(current_step) - span)
    rel_len = int(current_step) - clip_start + 1
    rel = video_sample_control_indices(rel_len, action_video_freq_ratio)
    return [clip_start + int(i) for i in rel]


def stack_rgb_to_video_tensor(
    rgb_frames: Sequence[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Stack CHW tensors in control order -> ``[1, 3, T, H, W]`` in [-1, 1]."""
    if not rgb_frames:
        raise ValueError("rgb_frames must be non-empty")
    frames: List[torch.Tensor] = []
    for fr in rgb_frames:
        t = fr
        if t.ndim == 4:
            t = t[0]
        if t.ndim != 3:
            raise ValueError(f"expected CHW tensor, got {tuple(t.shape)}")
        if t.dtype == torch.uint8:
            t = t.to(dtype=dtype, device=device) * (2.0 / 255.0) - 1.0
        else:
            t = t.to(dtype=dtype, device=device)
            if t.max() > 1.5:
                t = t * (2.0 / 255.0) - 1.0
            elif t.max() <= 1.0 and t.min() >= 0.0:
                t = t * 2.0 - 1.0
        frames.append(t)
    stack = torch.stack(frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
    return stack.contiguous()


def build_deploy_video_tensor(
    seg_start: int,
    read_chw: Callable[[int], torch.Tensor],
    *,
    seq_len: int = 58,
    action_video_freq_ratio: int = 2,
    history_window: int = DEFAULT_ACTION_HISTORY_WINDOW,
    past_window: int | None = None,
    device: torch.device,
    dtype: torch.dtype,
    max_control_t: int | None = None,
    past_only: bool = True,
    control_fps: float = 20.0,
    video_history_seconds: float = 1.0,
    cond_pixel_frames: int | None = 5,
) -> torch.Tensor:
    """Build subsampled video clip for Cosmos / VGGT towers.

    ``past_only=True`` (default): official V2W cond window (5 px @ n_lcf=2) ending at
    ``seg_start``, unless ``cond_pixel_frames`` is overridden.
    """
    if past_only:
        ctrl_indices = deploy_past_subsampled_video_control_indices(
            int(seg_start),
            control_fps=float(control_fps),
            video_history_seconds=float(video_history_seconds),
            action_video_freq_ratio=int(action_video_freq_ratio),
            cond_pixel_frames=cond_pixel_frames,
        )
    else:
        if past_window is not None:
            history_window = int(past_window)
        ctrl_indices = deploy_subsampled_video_control_indices(
            seg_start,
            seq_len=seq_len,
            action_video_freq_ratio=action_video_freq_ratio,
            history_window=history_window,
        )
        cap = int(max_control_t) if max_control_t is not None else None
        if cap is not None:
            ctrl_indices = [min(int(c), cap) for c in ctrl_indices]

    unique_ctrl: list[int] = []
    index_map: dict[int, int] = {}
    order: list[int] = []
    for c in ctrl_indices:
        key = int(c)
        if key not in index_map:
            index_map[key] = len(unique_ctrl)
            unique_ctrl.append(key)
        order.append(index_map[key])
    unique_frames = [read_chw(c) for c in unique_ctrl]
    frames = [unique_frames[i] for i in order]
    return stack_rgb_to_video_tensor(
        frames,
        device=device,
        dtype=dtype,
    )


def cosmos_video_from_native_bcthw(
    video_bcthw: torch.Tensor,
    *,
    size: tuple[int, int],
    crop_scale: float | None = None,
) -> torch.Tensor:
    """Native ``[B,3,T,H,W]`` in [-1,1] → Cosmos ``(H,W)`` (matches train ``prepare_model_batch_gpu``)."""
    from phi0.data.dit4dit_video import dit4dit_preprocess_video

    if video_bcthw.ndim != 5 or video_bcthw.shape[1] != 3:
        raise ValueError(f"expected [B,3,T,H,W], got {tuple(video_bcthw.shape)}")
    pixel = ((video_bcthw + 1.0) * 0.5).permute(0, 2, 1, 3, 4).contiguous()
    resized = dit4dit_preprocess_video(pixel, size=size, crop_scale=crop_scale)
    out = resized.permute(0, 2, 1, 3, 4).contiguous()
    return out * 2.0 - 1.0
