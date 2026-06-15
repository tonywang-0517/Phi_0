"""Tests for padded video frame replacement (Cosmos + VGGT shared path)."""

from __future__ import annotations

import torch

from phi0.data.video_pad import apply_video_pad_replacement


def test_pad_replacement_uses_first_valid_frame():
    video = torch.zeros(1, 3, 4, 2, 2)
    for t in range(4):
        video[0, :, t, :, :] = float(t)
    pad = torch.tensor([[False, False, True, True]])
    out = apply_video_pad_replacement(video, pad)
    assert torch.allclose(out[0, :, 2], out[0, :, 0])
    assert torch.allclose(out[0, :, 3], out[0, :, 0])
    assert torch.allclose(out[0, :, 0], torch.zeros(3, 2, 2))
    assert torch.allclose(out[0, :, 1], torch.ones(3, 2, 2))


def test_pad_replacement_noop_without_mask():
    video = torch.randn(2, 3, 3, 8, 8)
    out = apply_video_pad_replacement(video, None)
    assert out is video or torch.equal(out, video)


def test_pad_replacement_preserves_valid_frames():
    video = torch.randn(1, 3, 2, 4, 4)
    pad = torch.tensor([[False, True]])
    out = apply_video_pad_replacement(video, pad)
    assert torch.allclose(out[0, :, 0], video[0, :, 0])
    assert torch.allclose(out[0, :, 1], video[0, :, 0])
