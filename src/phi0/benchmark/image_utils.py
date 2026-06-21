from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def _as_hw(resize_size: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(resize_size, int):
        return int(resize_size), int(resize_size)
    if len(resize_size) != 2:
        raise ValueError(f"resize_size must be int or (h,w), got {resize_size}")
    return int(resize_size[0]), int(resize_size[1])


def check_image_format(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be np.ndarray")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got dtype {image.dtype}")


def resize_image_for_policy(image: np.ndarray, resize_size: int | Tuple[int, int]) -> np.ndarray:
    """Resize to policy size with LANCZOS for VLA-style eval preprocessing."""
    check_image_format(image)
    h, w = _as_hw(resize_size)
    pil = Image.fromarray(image, mode="RGB")
    pil = pil.resize((w, h), resample=Image.LANCZOS)
    return np.asarray(pil, dtype=np.uint8)


def center_crop_like_vla(image: np.ndarray, crop_scale: float = 0.9) -> np.ndarray:
    """Center crop and resize back, matching VLA-Adapter eval behavior."""
    check_image_format(image)
    if not (0.0 < crop_scale <= 1.0):
        raise ValueError(f"crop_scale must be in (0,1], got {crop_scale}")
    h, w = image.shape[:2]
    crop_h = max(1, int(round(h * crop_scale)))
    crop_w = max(1, int(round(w * crop_scale)))
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2
    cropped = image[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return resize_image_for_policy(cropped, (h, w))

