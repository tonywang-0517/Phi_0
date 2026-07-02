"""Unit tests for VLM pretrain contract and RTC."""

from __future__ import annotations

import torch

from phi0.inference.rtc import (
    blend_action_chunks_rtc,
    create_rtc_soft_mask,
    validate_rtc_params,
)
from phi0.models.vlm.contract import (
    assert_training_deploy_vlm_frame_parity,
    resolve_vlm_observation_frame_index,
    slice_vlm_observation_pixel,
)
from phi0.models.vlm.preprocess import normalize_vlm_instruction, video_bcthw_to_pixel_batch


def test_normalize_vlm_instruction_lower():
    assert normalize_vlm_instruction("Pick Tissue") == "pick tissue"


def test_resolve_vlm_frame_single_pixel():
    assert resolve_vlm_observation_frame_index(pixel_time=1, past_action_window_size=1) == 0


def test_resolve_vlm_frame_deploy_last():
    assert (
        resolve_vlm_observation_frame_index(pixel_time=5, past_action_window_size=1) == 4
    )


def test_video_bcthw_to_pixel_uses_last_on_multi_frame():
    video = torch.zeros(1, 3, 3, 4, 4)
    video[:, :, -1] = 1.0
    pixel = video_bcthw_to_pixel_batch(video * 2.0 - 1.0, past_action_window_size=1)
    assert pixel.shape == (1, 1, 3, 4, 4)
    assert float(pixel.mean()) > 0.9


def test_training_deploy_vlm_frame_parity_train_obs_only():
    train_pixel = torch.rand(2, 1, 3, 8, 8)
    deploy_pixel = torch.rand(2, 1, 3, 8, 8)
    assert_training_deploy_vlm_frame_parity(
        train_pixel,
        deploy_pixel,
        past_action_window_size=1,
        subsampled_control_indices=[0],
    )


def test_rtc_mask_frozen_prefix():
    m = create_rtc_soft_mask(8, inference_delay=2, execution_horizon=3, schedule="hard")
    assert m[:2].tolist() == [1.0, 1.0]
    assert m[2:].tolist() == [0.0] * 6


def test_rtc_validate_params():
    validate_rtc_params(8, 2, 3)
    try:
        validate_rtc_params(8, 5, 3)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_rtc_blend():
    prev = torch.ones(1, 4, 2)
    new = torch.zeros(1, 4, 2)
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    out = blend_action_chunks_rtc(new, prev, mask)
    assert out[0, 0, 0] == 1.0
    assert out[0, 2, 0] == 0.0


def test_slice_vlm_observation_pixel():
    pixel = torch.arange(12).view(1, 3, 1, 2, 2).float()
    sl = slice_vlm_observation_pixel(pixel, 1)
    assert sl.shape == (1, 1, 1, 2, 2)
    assert float(sl[0, 0, 0, 0, 0]) == 4.0
