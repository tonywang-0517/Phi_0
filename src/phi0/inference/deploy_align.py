"""Training-aligned deploy: 33-step control clip + 17-frame video subsample + GT proprio."""

from __future__ import annotations

from typing import Callable, List, Sequence

import torch

from phi0.data.temporal_align import video_sample_control_indices


def control_step_to_native_frame(
    control_t: int,
    start_frame: int,
    deploy_fps: float,
    native_fps: float,
) -> int:
    return int(start_frame) + int(round(int(control_t) * float(native_fps) / float(deploy_fps)))


def deploy_clip_start(seg_start: int, past_window: int = 4) -> int:
    """Control index where a training-style clip begins (proprio prefix = first ``past_window`` steps)."""
    return max(0, int(seg_start) - int(past_window))


def deploy_control_clip_indices(
    seg_start: int,
    seq_len: int = 33,
    *,
    past_window: int = 4,
) -> List[int]:
    """``seq_len`` control steps forward from clip start (matches ``SequenceDataset``)."""
    clip_start = deploy_clip_start(seg_start, past_window)
    return [clip_start + i for i in range(int(seq_len))]


def deploy_proprio_control_indices(seg_start: int, past_window: int = 4) -> List[int]:
    """Proprio = first ``past_window`` steps of the clip (training ``split_proprio_future``)."""
    clip_start = deploy_clip_start(seg_start, past_window)
    return [clip_start + i for i in range(int(past_window))]


def deploy_subsampled_video_control_indices(
    seg_start: int,
    *,
    seq_len: int = 33,
    action_video_freq_ratio: int = 2,
    past_window: int = 4,
) -> List[int]:
    """17 absolute control indices (training ``video_control_indices`` on deploy timeline)."""
    clip = deploy_control_clip_indices(seg_start, seq_len, past_window=past_window)
    rel = video_sample_control_indices(seq_len, action_video_freq_ratio)
    return [clip[i] for i in rel]


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
    seq_len: int = 33,
    action_video_freq_ratio: int = 2,
    past_window: int = 4,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build training-aligned subsampled clip starting at ``seg_start - past_window``."""
    ctrl_indices = deploy_subsampled_video_control_indices(
        seg_start,
        seq_len=seq_len,
        action_video_freq_ratio=action_video_freq_ratio,
        past_window=past_window,
    )
    # Deduplicate reads when early clip repeats control index 0.
    unique_ctrl: list[int] = []
    index_map: dict[int, int] = {}
    order: list[int] = []
    for c in ctrl_indices:
        if c not in index_map:
            index_map[c] = len(unique_ctrl)
            unique_ctrl.append(c)
        order.append(index_map[c])
    unique_frames = [read_chw(c) for c in unique_ctrl]
    frames = [unique_frames[i] for i in order]
    return stack_rgb_to_video_tensor(
        frames,
        device=device,
        dtype=dtype,
    )
