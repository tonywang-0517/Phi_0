"""Cosmos latent rectified-flow config and training-time flow matching."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VideoFMConfig:
    num_inference_timesteps: int = 1  # DiT4DiT training FM (single-step flow matching)
    preview_inference_timesteps: int = 36  # Cosmos official default for video decode preview
    inference_conditional_frame_timestep: float = 0.1  # Cosmos Predict pipeline default
    inference_guidance_scale: float = 7.0  # Cosmos Predict pipeline default (CFG)
    num_pixel_frames_out: int | None = None  # None -> derive at runtime (DiT4DiT train_num_frames_out)
    num_latent_conditional_frames: int = 2  # Cosmos Video2World training: 1 -> 1px cond, 2 -> 5px cond
    flow_time_distribution: str = "logit_normal"  # logit_normal | uniform (DiT4DiT flow_matching)
    flow_high_sigma_ratio: float | None = 0.05
    flow_high_sigma_min: float | None = 0.98


def sample_flow_matching_t(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    time_distribution: str = "logit_normal",
    high_sigma_ratio: float | None = 0.05,
    high_sigma_min: float | None = 0.98,
) -> torch.Tensor:
    """Sample rectified-flow time t in [0, 1] (DiT4DiT Cosmos25 flow_matching)."""
    if str(time_distribution).lower() == "logit_normal":
        t = torch.sigmoid(torch.randn((batch_size,), device=device, dtype=dtype))
    else:
        t = torch.rand((batch_size,), device=device, dtype=dtype)
    if high_sigma_ratio is not None and high_sigma_ratio > 0 and high_sigma_min is not None:
        high_mask = torch.rand((batch_size,), device=device) < float(high_sigma_ratio)
        high_t = (
            torch.rand((batch_size,), device=device, dtype=dtype) * (1.0 - float(high_sigma_min))
            + float(high_sigma_min)
        )
        t = torch.where(high_mask, high_t, t)
    return t
