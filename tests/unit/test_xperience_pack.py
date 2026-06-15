"""Synthetic Xperience keypoints packing tests."""

from __future__ import annotations

import numpy as np

from phi0.schema.action_schema import KEYPOINTS_FLAT_DIM, pack_xperience_keypoints, unpack_keypoints_52


def test_pack_from_synthetic_keypoints_preserves_tail_zeros():
    keypoints = np.stack(
        [np.array([float(j), float(j) * 0.1, float(j) * 0.01], dtype=np.float32) for j in range(52)],
        axis=0,
    )
    packed = pack_xperience_keypoints(keypoints)
    assert packed[156:211].sum() == 0.0
    restored = unpack_keypoints_52(packed)
    assert np.allclose(restored, keypoints)
    assert packed[:KEYPOINTS_FLAT_DIM].shape[0] == KEYPOINTS_FLAT_DIM
