"""Unit tests for Xperience → unified action ground truth conversion."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from phi0.data.xperience import XperienceDataset
from phi0.data.xperience_unified_gt import (
    pack_xperience_unified_frame_gt,
    repack_clip_root_trans_local,
    validate_contacts_match_hdf5,
    validate_packed_action_structure,
    validate_root_trans_local_consistency,
    validate_rot6d_matches_hdf5_quats,
    validate_unified_gt_fk_matches_hdf5_quat,
)
from phi0.schema.unified_action_schema import D_UNIFIED, SEMANTIC_DIM, dim_mask_for_dataset
from phi0.viz.smplh_fk import load_skeleton_constants


def _identity_quats(n: int) -> np.ndarray:
    q = np.zeros((n, 4), dtype=np.float32)
    q[:, 0] = 1.0
    return q


def write_synthetic_xperience_hdf5(path: Path, *, num_frames: int = 5) -> None:
    """Minimal Xperience-like HDF5 for GT conversion tests."""
    caption = json.dumps({"config": {"Main Task": "synthetic test task"}})
    with h5py.File(path, "w") as f:
        roots = np.zeros((num_frames, 7), dtype=np.float32)
        for t in range(num_frames):
            roots[t, :3] = np.array([0.1 * t, -0.05 * t, 0.9 + 0.01 * t], dtype=np.float32)
            roots[t, 3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        f.create_dataset("full_body_mocap/Ts_world_root", data=roots)
        f.create_dataset("full_body_mocap/body_quats", data=_identity_quats(21)[None].repeat(num_frames, 0))
        f.create_dataset("full_body_mocap/left_hand_quats", data=_identity_quats(15)[None].repeat(num_frames, 0))
        f.create_dataset("full_body_mocap/right_hand_quats", data=_identity_quats(15)[None].repeat(num_frames, 0))
        f.create_dataset("full_body_mocap/betas", data=np.zeros((num_frames, 16), dtype=np.float32))
        contacts = np.zeros((num_frames, 21), dtype=np.float32)
        contacts[:, 5] = 1.0
        f.create_dataset("full_body_mocap/contacts", data=contacts)
        mano = np.random.RandomState(0).randn(num_frames, 21, 3).astype(np.float32) * 0.02
        f.create_dataset("hand_mocap/left_joints_3d", data=mano)
        f.create_dataset("hand_mocap/right_joints_3d", data=mano + 0.1)
        f.create_dataset("caption", data=np.array(caption.encode("utf-8")))


@pytest.fixture
def synthetic_hdf5(tmp_path: Path) -> Path:
    path = tmp_path / "annotation.hdf5"
    write_synthetic_xperience_hdf5(path, num_frames=8)
    return path


@pytest.fixture
def skeleton_constants():
    try:
        return load_skeleton_constants()
    except FileNotFoundError:
        pytest.skip("SMPL-H skeleton constants not available")


def test_pack_structure_and_reserved_tail(synthetic_hdf5: Path):
    with h5py.File(synthetic_hdf5, "r") as f:
        gt = pack_xperience_unified_frame_gt(f, 3, state_t=1)
    assert gt.action.shape == (D_UNIFIED,)
    assert np.allclose(gt.action[SEMANTIC_DIM:], 0.0)
    validate_packed_action_structure(gt.action)


def test_root_trans_local_with_state_anchor(synthetic_hdf5: Path):
    with h5py.File(synthetic_hdf5, "r") as f:
        gt = pack_xperience_unified_frame_gt(f, 4, state_t=1)
    validate_root_trans_local_consistency(gt)
    expected = np.array([0.4, -0.2, 0.94], dtype=np.float32) - np.array(
        [0.1, -0.05, 0.91], dtype=np.float32
    )
    assert np.allclose(gt.action[:3], expected, atol=1e-6)


def test_contacts_and_rot6d_match_hdf5(synthetic_hdf5: Path):
    with h5py.File(synthetic_hdf5, "r") as f:
        gt = pack_xperience_unified_frame_gt(f, 2, state_t=2)
        validate_contacts_match_hdf5(f, 2, gt.action)
        validate_rot6d_matches_hdf5_quats(f, 2, gt.action)


def test_fk_matches_hdf5_quat_reference(synthetic_hdf5: Path, skeleton_constants):
    with h5py.File(synthetic_hdf5, "r") as f:
        for t in range(3):
            metrics = validate_unified_gt_fk_matches_hdf5_quat(
                f, t, state_t=t, constants=skeleton_constants, atol=1e-3
            )
            assert metrics["max_abs_m"] < 1e-3


def test_clip_repack_root_trans_local(synthetic_hdf5: Path):
    with h5py.File(synthetic_hdf5, "r") as f:
        gts = [pack_xperience_unified_frame_gt(f, t, state_t=t) for t in range(4)]
    actions = np.stack([g.action for g in gts])
    roots = np.stack([g.target_root_trans_world for g in gts])
    repacked = repack_clip_root_trans_local(actions, roots, anchor_index=0)
    assert np.allclose(repacked[0, :3], 0.0, atol=1e-6)
    assert np.allclose(repacked[2, :3], roots[2] - roots[0], atol=1e-6)


def test_xperience_dataset_unified_mode(synthetic_hdf5: Path):
    ds = XperienceDataset(
        hdf5_path=synthetic_hdf5,
        max_frames=4,
        cache_video=False,
        action_rep="unified",
    )
    assert ds.action_dim == D_UNIFIED
    item = ds[2]
    assert item["action"].shape == (1, D_UNIFIED)
    assert item["action_dim_is_pad"].shape == (D_UNIFIED,)
    assert item["action_dim_is_pad"][:346].eq(False).all()
    assert item["action_dim_is_pad"][346:].all()
    assert item["betas"].shape == (16,)
    validate_packed_action_structure(item["action"][0].numpy())


@pytest.mark.skipif(
    not Path(
        "/mnt/data2/wpy/workspace/Isaac-GR00T/demo_data/xperience-10m-sample/annotation.hdf5"
    ).exists()
    and not Path(
        "/mnt/data1/wpy/workspace/Isaac-GR00T/demo_data/xperience-10m-sample/annotation.hdf5"
    ).exists(),
    reason="Xperience demo HDF5 not available",
)
def test_real_xperience_hdf5_fk_roundtrip(skeleton_constants):
    from phi0.data.xperience import DEFAULT_HDF5

    with h5py.File(DEFAULT_HDF5, "r") as f:
        for t in (0, 10, 100):
            gt = pack_xperience_unified_frame_gt(f, t, state_t=t)
            validate_packed_action_structure(gt.action)
            validate_root_trans_local_consistency(gt)
            validate_contacts_match_hdf5(f, t, gt.action)
            validate_rot6d_matches_hdf5_quats(f, t, gt.action)
            validate_unified_gt_fk_matches_hdf5_quat(
                f, t, state_t=t, constants=skeleton_constants, atol=2e-3
            )


def test_dim_mask_xperience_supervises_346_dims():
    mask = dim_mask_for_dataset("xperience")
    assert mask.shape == (D_UNIFIED,)
    assert mask[:346].all()
    assert not mask[346:].any()
