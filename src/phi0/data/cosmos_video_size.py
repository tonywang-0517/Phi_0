"""Cosmos video resolution helpers (train + deploy)."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

# Cosmos Predict2.5 robot 720-tier (704×1280 landscape; official post-train resolution).
DEFAULT_COSMOS_VIDEO_SIZE: Tuple[int, int] = (704, 1280)


def cosmos_video_size_from_cfg(data_cfg: Mapping[str, Any] | None) -> Tuple[int, int]:
    """Read ``data.cosmos_video_size`` as ``(H, W)``."""
    if not data_cfg:
        return DEFAULT_COSMOS_VIDEO_SIZE
    size = data_cfg.get("cosmos_video_size", DEFAULT_COSMOS_VIDEO_SIZE)
    if size is None:
        return DEFAULT_COSMOS_VIDEO_SIZE
    return (int(size[0]), int(size[1]))


def round_hw_to_multiple(height: int, width: int, base: int = 16) -> Tuple[int, int]:
    """Round H,W up to multiples of ``base`` (Cosmos VAE spatial factor)."""
    h = max(base, ((int(height) + base - 1) // base) * base)
    w = max(base, ((int(width) + base - 1) // base) * base)
    return h, w
