"""Tests for VGGT official-style preprocessing and register flattening."""

from __future__ import annotations

import numpy as np
import torch

from phi0.models.vggt.preprocess import (
    balanced_target_shape,
    batch_preprocess_balanced,
    center_crop_to_supported_aspect,
    preprocess_frame_balanced,
    video_to_vggt_input,
)
from phi0.models.vggt.tower import VGGT_NUM_REGISTERS, registers_from_aggregated


def _official_balanced_target(aspect_ratio: float, image_resolution: int = 512, patch_size: int = 16):
    token_number = (image_resolution // patch_size) ** 2
    w_patches = np.sqrt(token_number / aspect_ratio)
    h_patches = token_number / w_patches
    w_patches = max(1, int(np.round(w_patches)))
    h_patches = max(1, int(np.round(h_patches)))
    return h_patches * patch_size, w_patches * patch_size


def test_balanced_target_matches_official_xperience_resolution():
    # Xperience ego: H=480, W=640
    target_h, target_w = balanced_target_shape(480, 640, image_resolution=512)
    official_h, official_w = _official_balanced_target(480 / 640.0)
    assert (target_h, target_w) == (official_h, official_w)
    assert target_h % 16 == 0 and target_w % 16 == 0


def test_balanced_resize_not_square_stretch_for_4x3():
    frame = torch.rand(3, 480, 640)
    out = preprocess_frame_balanced(frame, image_resolution=512)
    assert out.shape == (3, 448, 592)
    # Bilinear square stretch would yield 512x512; official balanced keeps aspect.
    assert out.shape[1] != 512 or out.shape[2] != 512


def test_center_crop_extreme_aspect():
    # Very wide: aspect > 2.0 should crop width
    wide = torch.zeros(3, 100, 400)
    cropped = center_crop_to_supported_aspect(wide)
    assert cropped.shape[2] <= 400
    assert cropped.shape[1] == 100


def test_batch_preprocess_matches_single_frame():
    frame = torch.rand(3, 480, 640)
    single = preprocess_frame_balanced(frame, image_resolution=512)
    batch = batch_preprocess_balanced(frame.unsqueeze(0), image_resolution=512).squeeze(0)
    assert torch.allclose(single, batch, atol=1e-5)


def test_video_to_vggt_input_shape_and_range():
    video = torch.rand(2, 3, 5, 480, 640) * 2.0 - 1.0
    out = video_to_vggt_input(video, image_resolution=512)
    assert out.shape[0] == 2 and out.shape[1] == 5 and out.shape[2] == 3
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.shape[3] == 448 and out.shape[4] == 592


def test_registers_from_aggregated_strips_camera_and_flattens():
    b, s, d = 2, 3, 2048
    tokens = torch.randn(b, s, 1 + VGGT_NUM_REGISTERS, d)
    ctx, mask = registers_from_aggregated(tokens)
    assert ctx.shape == (b, s * VGGT_NUM_REGISTERS, d)
    assert mask.shape == (b, s * VGGT_NUM_REGISTERS)
    assert mask.all()
