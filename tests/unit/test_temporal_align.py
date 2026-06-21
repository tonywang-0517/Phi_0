"""Unit tests for DiT4DiT-style temporal alignment helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from phi0.data.temporal_align import (
    DEFAULT_DATASET_NATIVE_FPS,
    build_video2world_prepare_clip,
    control_to_native_indices,
    dit4dit_train_num_frames_out,
    split_video_cond_future,
    max_native_span_frames,
    native_span_frames,
    resample_action_sequence,
    resample_bool_sequence,
    video_sample_control_indices,
    video2world_cond_pixel_frames,
    video2world_gt_reference_uint8,
    video2world_mae_metrics,
    video_cond_pixel_frames_for_training,
)


def test_native_span_frames_same_fps():
    assert native_span_frames(5, 20.0, 20.0) == 5
    assert native_span_frames(100, 20.0, 20.0) == 100


def test_native_span_frames_egodex_ratio():
    # 33 control steps @ 20Hz on 30Hz native: (32 * 30/20) + 1 = 49
    assert native_span_frames(33, 20.0, 30.0) == 49


def test_native_span_frames_single_step():
    assert native_span_frames(1, 20.0, 30.0) == 1


def test_native_span_frames_invalid_fps():
    with pytest.raises(ValueError, match="fps must be positive"):
        native_span_frames(5, 0.0, 20.0)


def test_max_native_span_frames_mixed_datasets():
    span = max_native_span_frames(33, 20.0, DEFAULT_DATASET_NATIVE_FPS)
    assert span == native_span_frames(33, 20.0, 30.0)


def test_control_to_native_indices_endpoints():
    idx = control_to_native_indices(10, 5)
    assert idx.tolist() == [0, 2, 4, 7, 9]
    assert control_to_native_indices(1, 4).tolist() == [0, 0, 0, 0]
    assert control_to_native_indices(5, 1).tolist() == [0]


def test_split_video_cond_future():
    v = torch.randn(2, 3, 5, 8, 8)
    cond, future = split_video_cond_future(v)
    assert cond.shape == (2, 3, 1, 8, 8)
    assert future is not None and future.shape == (2, 3, 4, 8, 8)
    cond1, fut1 = split_video_cond_future(v[:, :, :1])
    assert fut1 is None


def test_video2world_cond_pixel_frames():
    assert video2world_cond_pixel_frames(1) == 1
    assert video2world_cond_pixel_frames(2) == 5
    assert video_cond_pixel_frames_for_training(2) == 5


def test_video2world_gt_reference_and_mae():
    t, h, w = 17, 4, 4
    gt = np.stack([np.full((h, w, 3), i, dtype=np.uint8) for i in range(t)], axis=0)
    future = np.stack([np.full((h, w, 3), 100 + i, dtype=np.uint8) for i in range(12)], axis=0)
    cond_px = 5
    ref = video2world_gt_reference_uint8(gt, cond_px, gt_future_thw3=future)
    assert ref.shape == gt.shape
    assert np.array_equal(ref[:cond_px], gt[-cond_px:])
    assert np.array_equal(ref[cond_px:], future[:12])

    pred = ref.copy()
    m = video2world_mae_metrics(pred, gt, cond_px, gt_future_thw3=future)
    assert m["cond_mae"] == pytest.approx(0.0, abs=1e-6)
    assert m["aligned_full_mae"] == pytest.approx(0.0, abs=1e-6)
    assert m["chrono_full_mae"] > 0.1


def test_build_video2world_prepare_clip():
    v = torch.randn(1, 3, 17, 8, 8)
    clip, n_in = build_video2world_prepare_clip(v, num_frames_out=17, num_latent_conditional_frames=2)
    assert n_in == 5
    assert clip.shape == (1, 3, 17, 8, 8)
    # tail 5 + pad 12 copies of last
    assert torch.allclose(clip[:, :, 4], clip[:, :, 5])


def test_video_cond_pixel_frames_legacy():
    # DiT4DiT g1: 17 control deltas, ratio 2 -> 9 pixel frames
    assert dit4dit_train_num_frames_out(17, 2) == 9
    assert dit4dit_train_num_frames_out(58, 2) == 29
    assert video_sample_control_indices(5, 2) == [0, 2, 4]
    assert video_sample_control_indices(5, 1) == [0, 1, 2, 3, 4]
    assert video_sample_control_indices(4, 3) == [0, 3]
    assert video_sample_control_indices(3, 0) == [0, 1, 2]


def test_observation_subsampled_frame_index_past_w1():
    from phi0.data.temporal_align import (
        observation_subsampled_frame_index,
        proprio_current_control_step,
    )

    subsampled = video_sample_control_indices(9, 2)
    assert proprio_current_control_step(1) == 0
    assert observation_subsampled_frame_index(1, subsampled) == 0
    assert subsampled[0] == 0


def test_observation_subsampled_frame_index_past_w5():
    from phi0.data.temporal_align import observation_subsampled_frame_index

    subsampled = video_sample_control_indices(13, 2)
    assert observation_subsampled_frame_index(5, subsampled) == 2
    assert subsampled[2] == 4


def test_resample_action_sequence_identity():
    src = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    out = resample_action_sequence(src, 4, 4)
    assert torch.allclose(out, src)


def test_resample_action_sequence_linear_endpoints():
    src = torch.tensor([[0.0], [10.0]])
    out = resample_action_sequence(src, 2, 3)
    assert out.shape == (3, 1)
    assert out[0, 0].item() == pytest.approx(0.0)
    assert out[-1, 0].item() == pytest.approx(10.0)
    assert out[1, 0].item() == pytest.approx(5.0)


def test_resample_action_sequence_downsample():
    src = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    out = resample_action_sequence(src, 4, 2)
    assert out.shape == (2, 1)
    assert out[0, 0].item() == pytest.approx(0.0)
    assert out[1, 0].item() == pytest.approx(3.0)


def test_resample_bool_sequence_nearest():
    flags = torch.tensor([True, False, True, False])
    out = resample_bool_sequence(flags, 4, 2)
    assert out.tolist() == [True, False]


def test_resample_action_sequence_invalid_lengths():
    src = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="invalid resample lengths"):
        resample_action_sequence(src, 0, 2)
