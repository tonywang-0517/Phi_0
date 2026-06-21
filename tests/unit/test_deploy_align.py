"""Unit tests for training-aligned deploy clip indexing."""

from __future__ import annotations

from phi0.inference.deploy_align import (
    deploy_clip_start,
    deploy_control_clip_indices,
    deploy_history_control_indices,
    deploy_past_subsampled_video_control_indices,
    deploy_subsampled_video_control_indices,
    deploy_v2w_cond_video_control_indices,
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


def test_deploy_past_subsampled_one_second_at_step_25():
    idx = deploy_past_subsampled_video_control_indices(
        25,
        control_fps=20.0,
        video_history_seconds=1.0,
        action_video_freq_ratio=2,
    )
    assert idx[0] == 5
    assert idx[-1] == 25
    assert len(idx) == 11
    assert len(set(idx)) == len(idx)


def test_deploy_v2w_official_five_frames_at_step_25():
    idx = deploy_past_subsampled_video_control_indices(
        25,
        action_video_freq_ratio=2,
        cond_pixel_frames=5,
    )
    assert idx == [17, 19, 21, 23, 25]
    assert len(idx) == 5


def test_deploy_v2w_early_episode():
    idx = deploy_past_subsampled_video_control_indices(
        3,
        action_video_freq_ratio=2,
        cond_pixel_frames=5,
    )
    assert idx[-1] == 3
    assert len(idx) == 5
