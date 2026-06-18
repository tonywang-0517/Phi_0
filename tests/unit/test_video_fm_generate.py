"""Cosmos video FM config."""

from __future__ import annotations

from phi0.models.cosmos.video_fm import VideoFMConfig


def test_video_fm_config_default():
    cfg = VideoFMConfig()
    assert cfg.num_inference_timesteps == 1
    assert cfg.num_pixel_frames_out is None
    assert cfg.flow_time_distribution == "logit_normal"
    assert cfg.preview_inference_timesteps == 36
    assert cfg.inference_guidance_scale == 7.0
