"""DiT4DiT-aligned Cosmos video preprocessing (crop + bilinear resize)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from phi0.data.cosmos_video_size import DEFAULT_COSMOS_VIDEO_SIZE


def _center_crop_btchw(pixel: torch.Tensor, crop_h: int, crop_w: int) -> torch.Tensor:
    """Center-crop ``[B,T,C,H,W]`` in [0,1] to ``(crop_h, crop_w)``."""
    _, _, _, h, w = pixel.shape
    if crop_h > h or crop_w > w:
        raise ValueError(f"crop ({crop_h}, {crop_w}) exceeds input ({h}, {w})")
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return pixel[:, :, :, top : top + crop_h, left : left + crop_w]


def dit4dit_preprocess_video(
    pixel_btchw: torch.Tensor,
    *,
    size: Tuple[int, int] = DEFAULT_COSMOS_VIDEO_SIZE,
    crop_scale: Optional[float] = None,
) -> torch.Tensor:
    """Resize video frames for Cosmos (DiT4DiT G1: bilinear 224²; Oxe: crop 0.95 then resize).

    Args:
        pixel_btchw: ``[B,T,C,H,W]`` float in [0, 1].
        size: ``(H, W)`` target resolution (DiT4DiT default 224×224).
        crop_scale: If set (e.g. 0.95), center-crop before resize (OxeDroid path).
    """
    if pixel_btchw.ndim != 5 or pixel_btchw.shape[2] != 3:
        raise ValueError(f"Expected [B,T,3,H,W], got {tuple(pixel_btchw.shape)}")
    out = pixel_btchw
    if crop_scale is not None:
        _, _, _, h, w = out.shape
        crop_h = max(1, int(round(h * float(crop_scale))))
        crop_w = max(1, int(round(w * float(crop_scale))))
        out = _center_crop_btchw(out, crop_h, crop_w)
    target_h, target_w = int(size[0]), int(size[1])
    if crop_scale is None and out.shape[3] == target_h and out.shape[4] == target_w:
        return out
    b, t, c, _, _ = out.shape
    flat = out.reshape(b * t, c, out.shape[3], out.shape[4])
    resized = F.interpolate(
        flat,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(b, t, c, target_h, target_w)


def dit4dit_preprocess_frame(
    pixel_chw: torch.Tensor,
    *,
    size: Tuple[int, int] = DEFAULT_COSMOS_VIDEO_SIZE,
    crop_scale: Optional[float] = None,
) -> torch.Tensor:
    """Resize a single ``[C,H,W]`` float frame in [0, 1] to Cosmos ``(H, W)``."""
    if pixel_chw.ndim != 3 or pixel_chw.shape[0] != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(pixel_chw.shape)}")
    clip = pixel_chw.unsqueeze(0).unsqueeze(0)
    out = dit4dit_preprocess_video(clip, size=size, crop_scale=crop_scale)
    return out[0, 0]
