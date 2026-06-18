"""Unit tests for training-aligned deploy clip indexing."""

from __future__ import annotations

from phi0.inference.deploy_align import (
    deploy_clip_start,
    deploy_control_clip_indices,
    deploy_history_control_indices,
    deploy_subsampled_video_control_indices,
)


def test_deploy_clip_start():
    assert deploy_clip_start(40, history_window=29) == 12
    assert deploy_clip_start(2, history_window=29) == 0


def test_deploy_control_clip_forward_from_clip_start():
    clip = deploy_control_clip_indices(40, seq_len=58, history_window=29)
    assert clip[0] == 12
    assert clip[-1] == 69
    assert len(clip) == 58


def test_deploy_subsampled_video_count():
    idx = deploy_subsampled_video_control_indices(
        40, seq_len=58, action_video_freq_ratio=2, history_window=29
    )
    assert len(idx) == 29
    assert idx[0] == 12
    assert idx[-1] == 68


def test_deploy_history_is_clip_prefix():
    assert deploy_history_control_indices(40, history_window=29) == list(range(12, 41))
    assert deploy_history_control_indices(2, history_window=29) == list(range(0, 29))
