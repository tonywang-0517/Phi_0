"""Real-Time Chunking (RTC) — Psi0 / GR00T-style action chunk blending."""

from __future__ import annotations

import math
from typing import Literal

import torch

RtcSchedule = Literal["exponential", "linear", "hard", "simple"]


def validate_rtc_params(
    horizon: int,
    inference_delay: int,
    execution_horizon: int,
) -> None:
    """Paper constraint: ``d <= s <= H - d`` (``predict_action_with_rtc_flow``)."""
    h = int(horizon)
    d = int(inference_delay)
    s = int(execution_horizon)
    if d <= 0 or s <= 0:
        raise ValueError(f"inference_delay and execution_horizon must be > 0, got d={d} s={s}")
    if not (d <= s <= h - d):
        raise ValueError(f"RTC constraint violated: need d <= s <= H-d, got d={d} s={s} H={h}")


def create_rtc_soft_mask(
    horizon: int,
    inference_delay: int,
    execution_horizon: int,
    *,
    schedule: RtcSchedule = "exponential",
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Soft mask W_i in [0, 1] — weight on **previous** chunk (1 = frozen)."""
    validate_rtc_params(horizon, inference_delay, execution_horizon)
    h = int(horizon)
    d = int(inference_delay)
    s = int(execution_horizon)
    mask = torch.zeros(h, device=device)
    if schedule == "hard":
        mask[:d] = 1.0
        return mask
    if schedule == "linear":
        mask[:d] = 1.0
        overlap_end = h - s
        if d < overlap_end:
            indices = torch.arange(d, overlap_end, device=device, dtype=torch.float32)
            mask[d:overlap_end] = 1.0 - (indices - float(d)) / float(overlap_end - d)
        return mask
    if schedule == "exponential":
        mask[:d] = 1.0
        overlap_end = h - s
        if d < overlap_end:
            indices = torch.arange(d, overlap_end, device=device, dtype=torch.float32)
            c_i = (overlap_end - indices) / float(overlap_end - d + 1)
            e = torch.tensor(math.e, device=device, dtype=torch.float32)
            mask[d:overlap_end] = c_i * (torch.exp(c_i) - 1.0) / (e - 1.0)
        return mask
    if schedule == "simple":
        mask[:d] = 1.0
        if d < h:
            indices = torch.arange(d, h, device=device, dtype=torch.float32)
            mask[d:] = torch.exp(-5.0 * (indices - float(d)) / float(h - d))
        return mask
    raise ValueError(f"Unknown RTC schedule: {schedule}")


def blend_action_chunks_rtc(
    new_chunk: torch.Tensor,
    prev_chunk: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """``mask * prev + (1 - mask) * new`` with broadcast over action dim."""
    if new_chunk.shape != prev_chunk.shape:
        raise ValueError(
            f"chunk shape mismatch: new={tuple(new_chunk.shape)} prev={tuple(prev_chunk.shape)}"
        )
    if mask.ndim != 1:
        raise ValueError(f"mask must be [H], got {tuple(mask.shape)}")
    h = int(mask.shape[0])
    if new_chunk.shape[-2] != h:
        raise ValueError(
            f"mask length {h} != chunk horizon {new_chunk.shape[-2]}"
        )
    m = mask.to(device=new_chunk.device, dtype=new_chunk.dtype).view(
        *([1] * (new_chunk.ndim - 2)), h, 1
    )
    return prev_chunk * m + new_chunk * (1.0 - m)
