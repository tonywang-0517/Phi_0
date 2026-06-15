"""VGGT-Omega official-style image preprocessing (balanced resize, aspect crop)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# Defaults from vggt_omega.utils.load_fn
DEFAULT_PATCH_SIZE = 16
DEFAULT_MIN_ASPECT = 0.5
DEFAULT_MAX_ASPECT = 2.0


def balanced_target_shape(
    height: int,
    width: int,
    *,
    image_resolution: int = 512,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> tuple[int, int]:
    """Return ``(target_h, target_w)`` matching official ``balanced`` mode."""
    aspect_ratio = float(height) / max(float(width), 1.0)
    token_number = (image_resolution // patch_size) ** 2
    w_patches = np.sqrt(token_number / aspect_ratio)
    h_patches = token_number / w_patches
    w_patches = max(1, int(np.round(w_patches)))
    h_patches = max(1, int(np.round(h_patches)))
    return h_patches * patch_size, w_patches * patch_size


def center_crop_to_supported_aspect(
    x: torch.Tensor,
    *,
    min_aspect_ratio: float = DEFAULT_MIN_ASPECT,
    max_aspect_ratio: float = DEFAULT_MAX_ASPECT,
) -> torch.Tensor:
    """Center-crop CHW or NCHW tensor to aspect ratio in ``[min, max]`` (height/width)."""
    if x.ndim == 3:
        _, height, width = x.shape
        aspect_ratio = float(height) / max(float(width), 1.0)
        if aspect_ratio < min_aspect_ratio:
            crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
            left = max((width - crop_width) // 2, 0)
            return x[:, :, left : left + crop_width]
        if aspect_ratio > max_aspect_ratio:
            crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
            top = max((height - crop_height) // 2, 0)
            return x[:, top : top + crop_height, :]
        return x

    if x.ndim != 4:
        raise ValueError(f"expected CHW or NCHW tensor, got {tuple(x.shape)}")
    _, _, height, width = x.shape
    aspect_ratio = float(height) / max(float(width), 1.0)
    if aspect_ratio < min_aspect_ratio:
        crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
        left = max((width - crop_width) // 2, 0)
        return x[:, :, :, left : left + crop_width]
    if aspect_ratio > max_aspect_ratio:
        crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
        top = max((height - crop_height) // 2, 0)
        return x[:, :, top : top + crop_height, :]
    return x


def batch_preprocess_balanced(
    frames: torch.Tensor,
    *,
    image_resolution: int = 512,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> torch.Tensor:
    """Batch ``[N,C,H,W]`` in [0,1] → official balanced resize (vectorized)."""
    if frames.ndim != 4:
        raise ValueError(f"expected [N,C,H,W], got {tuple(frames.shape)}")
    cropped = center_crop_to_supported_aspect(frames)
    _, _, h, w = cropped.shape
    target_h, target_w = balanced_target_shape(
        h, w, image_resolution=image_resolution, patch_size=patch_size
    )
    if h == target_h and w == target_w:
        return cropped
    return F.interpolate(
        cropped,
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0.0, 1.0)


def preprocess_frame_balanced(
    frame: torch.Tensor,
    *,
    image_resolution: int = 512,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> torch.Tensor:
    """Single frame ``[C,H,W]`` in [0,1] → official balanced resize."""
    if frame.ndim != 3:
        raise ValueError(f"expected [C,H,W], got {tuple(frame.shape)}")
    return batch_preprocess_balanced(
        frame.unsqueeze(0),
        image_resolution=image_resolution,
        patch_size=patch_size,
    ).squeeze(0)


def video_to_vggt_input(
    video: torch.Tensor,
    *,
    image_resolution: int = 512,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> torch.Tensor:
    """Convert Phi_0 ``[B,3,T,H,W]`` in [-1,1] to VGGT ``[B,T,3,h,w]`` in [0,1]."""
    if video.ndim != 5:
        raise ValueError(f"`video` must be [B,3,T,H,W], got {tuple(video.shape)}")
    x = video.permute(0, 2, 1, 3, 4).contiguous()
    x = (x + 1.0) * 0.5
    b, t, c, h, w = x.shape
    flat = batch_preprocess_balanced(
        x.view(b * t, c, h, w),
        image_resolution=image_resolution,
        patch_size=patch_size,
    )
    _, th, tw = flat.shape[1:]
    return flat.view(b, t, c, th, tw)
