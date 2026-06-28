"""Pick-tissue training video load spec (obs-only vs full subsample)."""

from __future__ import annotations

from phi0.data.pick_tissue_unified import training_video_load_spec
from phi0.data.temporal_align import proprio_current_control_step, video_sample_control_indices


def test_training_video_load_spec_obs_only():
    fps = 50.0
    seq_len, ratio = 33, 2
    ts, ctrl = training_video_load_spec(
        fps=fps,
        seq_len=seq_len,
        action_video_freq_ratio=ratio,
        train_obs_only_video=True,
        obs_control_index=0,
    )
    assert ts == [0.0]
    assert ctrl == [0]
    full = video_sample_control_indices(seq_len, ratio)
    assert len(full) == 17


def test_training_video_load_spec_full_subsample():
    fps = 50.0
    seq_len, ratio = 33, 2
    ts, ctrl = training_video_load_spec(
        fps=fps,
        seq_len=seq_len,
        action_video_freq_ratio=ratio,
        train_obs_only_video=False,
    )
    full = video_sample_control_indices(seq_len, ratio)
    assert ctrl == full
    assert ts == [t / fps for t in full]


def test_obs_only_aligns_with_proprio_current_past_w1():
    past_w = 1
    obs = proprio_current_control_step(past_w)
    _, ctrl = training_video_load_spec(
        fps=50.0,
        seq_len=33,
        action_video_freq_ratio=2,
        train_obs_only_video=True,
        obs_control_index=obs,
    )
    assert ctrl == [0]
