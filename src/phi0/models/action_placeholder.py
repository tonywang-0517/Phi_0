"""Future action placeholder tokens (VLA-Adapter: zero base + fixed learnable perturbation)."""

from __future__ import annotations

import torch

# prismatic/models/action_heads.py learnable_random_perturbations: N(0, 0.02)
FUTURE_PLACEHOLDER_NOISE_STD = 0.02


def make_future_action_placeholder(
    batch_size: int,
    seq_len: int,
    raw_action_dim: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Inference / eval base: all-zero future tokens ``[B, T, D]``."""
    return torch.zeros(
        int(batch_size),
        int(seq_len),
        int(raw_action_dim),
        device=device,
        dtype=dtype,
    )


def apply_vla_future_placeholder_noise(
    future_tokens: torch.Tensor,
    *,
    noise_std: float = FUTURE_PLACEHOLDER_NOISE_STD,
) -> torch.Tensor:
    """Legacy i.i.d. noise helper (tests only). Training uses ``ActionACTDiT.future_placeholder_perturbation``."""
    std = float(noise_std)
    if std <= 0:
        return future_tokens
    if future_tokens.ndim != 3:
        raise ValueError(f"future_tokens must be [B,T,D], got {tuple(future_tokens.shape)}")
    batch_size, seq_len, dim = future_tokens.shape
    noise = torch.randn(
        seq_len,
        dim,
        device=future_tokens.device,
        dtype=future_tokens.dtype,
    ) * std
    return future_tokens + noise.unsqueeze(0).expand(batch_size, -1, -1)
