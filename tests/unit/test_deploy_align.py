"""Unit tests for training-aligned deploy clip indexing."""

from __future__ import annotations

from phi0.inference.deploy_align import (
    deploy_clip_start,
    deploy_control_clip_indices,
    deploy_proprio_control_indices,
    deploy_subsampled_video_control_indices,
)


def test_deploy_clip_start():
    assert deploy_clip_start(29, past_window=4) == 25
    assert deploy_clip_start(2, past_window=4) == 0


def test_deploy_control_clip_forward_from_clip_start():
    clip = deploy_control_clip_indices(29, seq_len=33, past_window=4)
    assert clip[0] == 25
    assert clip[-1] == 57
    assert len(clip) == 33


def test_deploy_subsampled_video_count():
    idx = deploy_subsampled_video_control_indices(
        29, seq_len=33, action_video_freq_ratio=2, past_window=4
    )
    assert len(idx) == 17
    assert idx[0] == 25
    assert idx[-1] == 57


def test_deploy_proprio_is_clip_prefix():
    assert deploy_proprio_control_indices(29, past_window=4) == [25, 26, 27, 28]
    assert deploy_proprio_control_indices(2, past_window=4) == [0, 1, 2, 3]
