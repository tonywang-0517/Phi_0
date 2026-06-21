"""V2W triple-tower alignment (Cosmos / VGGT / action proprio prefix)."""

from __future__ import annotations

import pytest

from phi0.align.v2w_cond import (
    ACTION_PROPRIO_PREFIX_SIZE,
    validate_v2w_triple_align,
    v2w_cond_pixel_frames,
)
from phi0.data.temporal_align import video2world_cond_pixel_frames


def test_v2w_cond_pixel_frames_official():
    assert video2world_cond_pixel_frames(1) == 1
    assert video2world_cond_pixel_frames(2) == 5
    assert v2w_cond_pixel_frames(2) == 5
    assert ACTION_PROPRIO_PREFIX_SIZE == 5


def test_validate_v2w_triple_align_ok():
    validate_v2w_triple_align(
        past_action_window_size=5,
        num_latent_conditional_frames=2,
        seq_len=24,
    )


def test_validate_v2w_triple_align_mismatch():
    with pytest.raises(ValueError, match="past_action_window_size"):
        validate_v2w_triple_align(
            past_action_window_size=4,
            num_latent_conditional_frames=2,
            seq_len=24,
        )


def test_validate_v2w_seq_len_too_short():
    with pytest.raises(ValueError, match="seq_len"):
        validate_v2w_triple_align(
            past_action_window_size=5,
            num_latent_conditional_frames=2,
            seq_len=5,
        )
