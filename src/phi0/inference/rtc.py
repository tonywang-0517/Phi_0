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


def shift_action_chunk_rtc(chunk: torch.Tensor, steps: int) -> torch.Tensor:
    """Roll chunk forward by ``steps``; pad tail with last frame (next query time origin)."""
    s = int(steps)
    if s <= 0:
        return chunk
    if chunk.ndim == 3:
        c = chunk[0]
        if s >= c.shape[0]:
            return c[-1:].expand(c.shape[0], -1).unsqueeze(0)
        shifted = torch.cat([c[s:], c[-1:].expand(s, -1)], dim=0)
        return shifted.unsqueeze(0)
    if s >= chunk.shape[0]:
        return chunk[-1:].expand(chunk.shape[0], -1)
    return torch.cat([chunk[s:], chunk[-1:].expand(s, -1)], dim=0)


def resolve_rtc_deploy_cfg(
    cfg,
    *,
    rtc_flag: bool = False,
    inference_delay: int = 0,
    execution_horizon: int = 0,
    schedule: str = "",
) -> dict:
    """Merge model ``rtc.*`` with CLI overrides (non-zero / explicit flag wins)."""
    model_rtc = getattr(cfg, "rtc", None) or getattr(getattr(cfg, "model", None), "rtc", None) or {}
    enabled = bool(getattr(model_rtc, "enabled", False)) or bool(rtc_flag)
    d = int(inference_delay or getattr(model_rtc, "inference_delay", 2))
    s = int(execution_horizon or getattr(model_rtc, "execution_horizon", 4))
    sched = str(schedule or getattr(model_rtc, "schedule", "exponential"))
    return {
        "enabled": enabled,
        "inference_delay": d,
        "execution_horizon": s,
        "schedule": sched,
    }
