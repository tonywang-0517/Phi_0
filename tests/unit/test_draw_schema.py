"""Unit tests for dim_mask_for_dataset and zero_unsupervised dims."""

from __future__ import annotations

import numpy as np
import torch

from phi0.schema.action_schema import D_RAW, KEYPOINTS_FLAT_DIM
from phi0.schema.draw_schema import DrawLayout, zero_unsupervised_action_dims, zero_unsupervised_action_dims_np


def test_dim_mask_xperience_supervises_keypoints_only():
    layout = DrawLayout()
    mask = layout.dim_mask_for_dataset("xperience")
    assert mask.shape == (D_RAW,)
    assert mask.sum() == KEYPOINTS_FLAT_DIM
    assert mask[:KEYPOINTS_FLAT_DIM].all()
    assert not mask[KEYPOINTS_FLAT_DIM:211].any()
    assert not mask[211:].any()


def test_dim_mask_egodex_supervises_keypoints_slice():
    layout = DrawLayout()
    mask = layout.dim_mask_for_dataset("egodex")
    assert mask.shape == (D_RAW,)
    assert mask.sum() == KEYPOINTS_FLAT_DIM
    assert mask[:KEYPOINTS_FLAT_DIM].all()


def test_zero_unsupervised_action_dims():
    action = torch.ones(2, 3, D_RAW)
    out = zero_unsupervised_action_dims(action)
    assert torch.all(out[..., :KEYPOINTS_FLAT_DIM] == 1.0)
    assert torch.all(out[..., KEYPOINTS_FLAT_DIM:] == 0.0)

    row = np.arange(D_RAW, dtype=np.float32)
    out_np = zero_unsupervised_action_dims_np(row)
    assert np.allclose(out_np[:KEYPOINTS_FLAT_DIM], row[:KEYPOINTS_FLAT_DIM])
    assert np.allclose(out_np[KEYPOINTS_FLAT_DIM:], 0.0)
