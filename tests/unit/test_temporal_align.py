"""Unit tests for DiT4DiT-style temporal alignment helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from phi0.data.temporal_align import (
    DEFAULT_DATASET_NATIVE_FPS,
    control_to_native_indices,
    max_native_span_frames,
    native_span_frames,
    resample_action_sequence,
    resample_bool_sequence,
    video_sample_control_indices,
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


def test_video_sample_control_indices():
    assert video_sample_control_indices(5, 2) == [0, 2, 4]
    assert video_sample_control_indices(5, 1) == [0, 1, 2, 3, 4]
    assert video_sample_control_indices(4, 3) == [0, 3]
    assert video_sample_control_indices(3, 0) == [0, 1, 2]


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
