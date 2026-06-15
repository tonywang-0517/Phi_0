"""Unit tests for keypoints action schema pack/unpack."""

from __future__ import annotations

import numpy as np
import pytest

from phi0.schema.action_schema import (
    D_RAW,
    KEYPOINTS_FLAT_DIM,
    NUM_SKELETON_JOINTS,
    get_action_schema,
    pack_xperience_keypoints,
    unpack_keypoints_52,
)


def test_pose_dim_end_is_156():
    schema = get_action_schema()
    assert schema.rep == "keypoints"
    assert schema.pose_dim_end == 156
    assert KEYPOINTS_FLAT_DIM == 156


def test_pack_unpack_roundtrip():
    rng = np.random.RandomState(0)
    keypoints = rng.randn(NUM_SKELETON_JOINTS, 3).astype(np.float32)
    betas = rng.randn(16).astype(np.float32)
    tactile = rng.rand(10).astype(np.float32)

    packed = pack_xperience_keypoints(keypoints, betas, tactile)
    assert packed.shape == (D_RAW,)
    assert np.allclose(packed[:KEYPOINTS_FLAT_DIM], keypoints.reshape(-1))
    assert np.allclose(packed[211:227], betas)
    assert np.allclose(packed[227:237], tactile)
    assert np.allclose(packed[156:211], 0.0)

    restored = unpack_keypoints_52(packed)
    assert restored.shape == (NUM_SKELETON_JOINTS, 3)
    assert np.allclose(restored, keypoints)


def test_unpack_batch_shape():
    batch = np.stack(
        [pack_xperience_keypoints(np.full((52, 3), float(i), dtype=np.float32)) for i in range(4)],
        axis=0,
    )
    kp = unpack_keypoints_52(batch)
    assert kp.shape == (4, 52, 3)
    assert kp[2, 0, 0] == pytest.approx(2.0)
