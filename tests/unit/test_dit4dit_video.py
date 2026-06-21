"""Unit tests for DiT4DiT-aligned Cosmos video preprocessing."""

from __future__ import annotations

import torch

from phi0.data.dit4dit_video import dit4dit_preprocess_video


def test_dit4dit_resize_480x640_to_224():
    pixel = torch.rand(1, 5, 3, 480, 640)
    out = dit4dit_preprocess_video(pixel, size=(224, 224))
    assert out.shape == (1, 5, 3, 224, 224)


def test_dit4dit_skip_noop_resize():
    pixel = torch.rand(1, 5, 3, 224, 224)
    out = dit4dit_preprocess_video(pixel, size=(224, 224))
    assert out is pixel


def test_dit4dit_crop_scale_then_resize():
    pixel = torch.rand(2, 3, 3, 480, 640)
    out = dit4dit_preprocess_video(pixel, size=(224, 224), crop_scale=0.95)
    assert out.shape == (2, 3, 3, 224, 224)
