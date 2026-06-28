"""Unified 512-d deploy must not zero sonic / gripper (unlike legacy 256-d tail wipe)."""

from __future__ import annotations

import numpy as np
import torch

from phi0.schema.draw_schema import zero_unsupervised_action_dims, zero_unsupervised_action_dims_np
from phi0.schema.unified_action_schema import D_UNIFIED, SLICES, unsupervised_dim_mask_for_dataset


def test_g1_sonic_unsupervised_mask_keeps_sonic_and_gripper():
    unsup = unsupervised_dim_mask_for_dataset("g1_sonic")
    s, e = SLICES["sonic_motion_token_64"]
    gs, ge = SLICES["g1_gripper_joints_14"]
    assert not unsup[s:e].any()
    assert not unsup[gs:ge].any()
    assert unsup[SLICES["root_trans_local"][0] : SLICES["root_trans_local"][1]].all()


def test_zero_unsupervised_unified_preserves_sonic():
    x = torch.randn(2, 8, D_UNIFIED)
    y = zero_unsupervised_action_dims(x)
    s, e = SLICES["sonic_motion_token_64"]
    gs, ge = SLICES["g1_gripper_joints_14"]
    torch.testing.assert_close(y[..., s:e], x[..., s:e])
    torch.testing.assert_close(y[..., gs:ge], x[..., gs:ge])
    assert torch.all(y[..., unsupervised_dim_mask_for_dataset("g1_sonic")] == 0.0)

    row = np.arange(D_UNIFIED, dtype=np.float32)
    out = zero_unsupervised_action_dims_np(row)
    np.testing.assert_allclose(out[s:e], row[s:e])
