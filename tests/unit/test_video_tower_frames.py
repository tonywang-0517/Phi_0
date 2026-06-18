"""Cosmos video tower frame matching and prepare_latents."""

from __future__ import annotations

import torch
from diffusers.video_processor import VideoProcessor

from phi0.models.cosmos.video_tower import CosmosVideoTower


def test_match_pixel_num_frames_no_repeat_interleave():
    tower = CosmosVideoTower.__new__(CosmosVideoTower)
    video = torch.randn(1, 17, 3, 8, 8)
    out = CosmosVideoTower._match_pixel_num_frames(tower, video, 17)
    assert out.shape == (1, 17, 3, 8, 8)
    assert torch.allclose(out, video)

    padded = CosmosVideoTower._match_pixel_num_frames(tower, video[:, :10], 17)
    assert padded.shape == (1, 17, 3, 8, 8)
    assert torch.allclose(padded[:, :10], video[:, :10])
    assert torch.allclose(padded[:, 10:], video[:, 9:10].expand(-1, 7, -1, -1, -1))

    truncated = CosmosVideoTower._match_pixel_num_frames(tower, video, 9)
    assert truncated.shape == (1, 9, 3, 8, 8)
    assert torch.allclose(truncated, video[:, :9])


def test_preprocess_pixels_for_vae_identity_on_native_resolution():
    tower = CosmosVideoTower.__new__(CosmosVideoTower)
    tower.vae_scale_factor_spatial = 8
    tower.video_processor = VideoProcessor(vae_scale_factor=8)
    video = torch.rand(2, 3, 5, 64, 80) * 2.0 - 1.0
    out = CosmosVideoTower._preprocess_pixels_for_vae(tower, video)
    assert out.shape == video.shape
    assert torch.allclose(out, video, atol=1e-6, rtol=0.0)
