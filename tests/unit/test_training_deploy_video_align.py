"""Train vs deploy V2W video conditioning must share the same control timeline."""

from __future__ import annotations

import torch

from phi0.inference.deploy_align import build_deploy_video_tensor, stack_rgb_to_video_tensor
from phi0.data.temporal_align import (
    gather_subsampled_video_bcthw,
    proprio_current_control_step,
    select_proprio_aligned_tower_video,
    subsampled_positions_for_control_indices,
    training_v2w_cond_control_indices,
    video_sample_control_indices,
)


def test_training_cond_not_full_clip_tail():
    """Regression: old bug used tail-5 of the 12-frame subsampled clip."""
    seq_len, ratio, past_w = 24, 2, 5
    full = video_sample_control_indices(seq_len, ratio)
    old_tail = full[-5:]
    new_cond = training_v2w_cond_control_indices(
        past_action_window_size=past_w,
        action_video_freq_ratio=ratio,
        cond_pixel_frames=5,
    )
    assert old_tail == [14, 16, 18, 20, 22]
    assert new_cond == [0, 0, 0, 2, 4]
    assert set(old_tail).isdisjoint(set(range(past_w)))


def test_training_cond_ends_at_proprio_current():
    past_w = 5
    cond = training_v2w_cond_control_indices(past_action_window_size=past_w)
    assert cond[-1] == proprio_current_control_step(past_w)
    assert set(cond) & set(range(past_w))


def test_training_gather_matches_deploy_tensor_at_proprio_current():
    seq_len, ratio, past_w = 24, 2, 5
    current = proprio_current_control_step(past_w)

    frames = {c: torch.full((3, 16, 16), float(c)) for c in video_sample_control_indices(seq_len, ratio)}

    def read_chw(control_t: int) -> torch.Tensor:
        return frames[int(control_t)]

    deploy_clip = build_deploy_video_tensor(
        current,
        read_chw,
        past_only=True,
        cond_pixel_frames=5,
        action_video_freq_ratio=ratio,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    subsampled = video_sample_control_indices(seq_len, ratio)
    train_full = stack_rgb_to_video_tensor(
        [frames[c] for c in subsampled],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    train_aligned, cond_ctrl = select_proprio_aligned_tower_video(
        train_full,
        seq_len=seq_len,
        past_action_window_size=past_w,
        action_video_freq_ratio=ratio,
        cond_pixel_frames=5,
    )
    assert cond_ctrl == training_v2w_cond_control_indices(
        past_action_window_size=past_w,
        action_video_freq_ratio=ratio,
        cond_pixel_frames=5,
    )
    assert torch.equal(train_aligned, deploy_clip)


def test_training_gather_matches_deploy_at_eval_step_10():
    """Deploy at sim step 10 uses the same index rule as training when current=10."""
    ratio = 2
    current = 10
    frames = {c: torch.full((3, 8, 8), float(c)) for c in range(current + 1)}

    deploy_clip = build_deploy_video_tensor(
        current,
        lambda c: frames[int(c)],
        past_only=True,
        cond_pixel_frames=5,
        action_video_freq_ratio=ratio,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    cond_ctrl = training_v2w_cond_control_indices(
        past_action_window_size=current + 1,
        action_video_freq_ratio=ratio,
        cond_pixel_frames=5,
    )
    assert cond_ctrl[-1] == current
    gathered = gather_subsampled_video_bcthw(
        deploy_clip,
        subsampled_positions_for_control_indices(cond_ctrl, cond_ctrl),
    )
    assert torch.equal(gathered, deploy_clip)


def test_subsampled_position_lookup_with_padding_duplicates():
    subsampled = [0, 2, 4, 6, 8, 10]
    cond = [0, 0, 0, 2, 4]
    pos = subsampled_positions_for_control_indices(cond, subsampled)
    assert pos == [0, 0, 0, 1, 2]
