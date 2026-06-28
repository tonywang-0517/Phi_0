"""Unit tests for Isaac-GR00T teleop → unified 512-d packing."""

from __future__ import annotations

import numpy as np

from phi0.data.groot_unified_io import pack_from_groot_teleop_row
from phi0.schema.unified_action_schema import D_UNIFIED, unpack_g1_gripper_joints_14


def _teleop_row(*, wbc: np.ndarray | None = None) -> dict:
    wbc = np.zeros(43, dtype=np.float32) if wbc is None else np.asarray(wbc, dtype=np.float32)
    joints = np.zeros((24, 3), dtype=np.float32)
    joints[0] = [0.1, 0.2, 0.9]
    return {
        "teleop.smpl_joints": joints,
        "teleop.body_quat_w": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "teleop.smpl_pose": np.zeros(63, dtype=np.float32),
        "action.wbc": wbc,
    }


def test_groot_pack_writes_gripper_from_wbc():
    wbc = np.linspace(0.0, 1.0, 43, dtype=np.float32)
    row = _teleop_row(wbc=wbc)
    gt = pack_from_groot_teleop_row(row)
    assert gt.action.shape == (D_UNIFIED,)
    expected = np.concatenate([wbc[22:29], wbc[36:43]]).astype(np.float32)
    assert np.allclose(unpack_g1_gripper_joints_14(gt.action), expected)
    assert np.allclose(gt.action[396:460], 0.0)
    assert np.allclose(gt.action[460:], 0.0)


def test_groot_pack_zeros_root_trans_without_base_trans():
    from phi0.schema.unified_action_schema import unpack_root_trans_local

    row = _teleop_row()
    joints = np.zeros((24, 3), dtype=np.float32)
    joints[0] = [1.0, 2.0, 3.0]
    row["teleop.smpl_joints"] = joints
    gt = pack_from_groot_teleop_row(row)
    assert np.allclose(unpack_root_trans_local(gt.action), 0.0)


def test_groot_pack_root_trans_delta_with_base_trans():
    from phi0.schema.unified_action_schema import unpack_root_trans_local

    row0 = _teleop_row()
    row1 = _teleop_row()
    row0["observation.base_trans"] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    row1["observation.base_trans"] = np.array([0.1, 0.0, 1.0], dtype=np.float64)
    gt = pack_from_groot_teleop_row(row1, state_t_row=row0)
    assert np.allclose(unpack_root_trans_local(gt.action), [0.1, 0.0, 0.0])


def test_invalid_smpl_row_detected_by_frame_index_zero():
    from phi0.data.groot_unified_io import is_invalid_smpl_teleop_row

    valid = _teleop_row()
    valid["teleop.smpl_frame_index"] = np.array([4517], dtype=np.int64)
    invalid = _teleop_row()
    invalid["teleop.smpl_joints"] = np.zeros((24, 3), dtype=np.float32)
    invalid["teleop.smpl_frame_index"] = np.array([0], dtype=np.int64)
    assert not is_invalid_smpl_teleop_row(valid)
    assert is_invalid_smpl_teleop_row(invalid)


def test_forward_fill_smpl_keeps_wbc_updates_root():
    from phi0.data.groot_unified_io import (
        forward_fill_smpl_teleop_row,
        pack_from_groot_teleop_row,
        prepare_groot_row_for_unified,
        read_groot_base_trans_world,
        read_groot_pelvis_world,
    )

    valid = _teleop_row(wbc=np.linspace(0, 1, 43, dtype=np.float32))
    valid["teleop.smpl_frame_index"] = np.array([100], dtype=np.int64)
    valid["observation.base_trans"] = np.array([0.2, -0.3, 0.85], dtype=np.float64)
    joints = np.zeros((24, 3), dtype=np.float32)
    joints[0] = [9.0, 9.0, 9.0]
    valid["teleop.smpl_joints"] = joints

    invalid_wbc = np.linspace(0.5, 1.5, 43, dtype=np.float32)
    invalid = _teleop_row(wbc=invalid_wbc)
    invalid["teleop.smpl_joints"] = np.zeros((24, 3), dtype=np.float32)
    invalid["teleop.smpl_frame_index"] = np.array([0], dtype=np.int64)
    invalid["observation.base_trans"] = np.array([0.5, 0.5, 0.9], dtype=np.float64)

    prepared, _, repaired = prepare_groot_row_for_unified(invalid, valid)
    assert repaired
    assert np.allclose(read_groot_base_trans_world(prepared), [0.5, 0.5, 0.9])
    assert np.allclose(read_groot_pelvis_world(prepared), [9.0, 9.0, 9.0])
    assert np.allclose(prepared["action.wbc"], invalid_wbc)

    gt = pack_from_groot_teleop_row(prepared)
    assert np.allclose(gt.target_root_trans_world, [0.5, 0.5, 0.9])

    filled = forward_fill_smpl_teleop_row(invalid, valid)
    assert int(np.asarray(filled["teleop.smpl_frame_index"]).reshape(-1)[0]) == 100


def test_read_groot_base_trans_over_smpl_pelvis():
    from phi0.data.groot_unified_io import read_groot_base_trans_world

    row = _teleop_row()
    row["observation.base_trans"] = np.array([1.0, 2.0, 0.79], dtype=np.float64)
    row["teleop.smpl_joints"][0] = [0.0, 0.0, 0.0]
    assert np.allclose(read_groot_base_trans_world(row), [1.0, 2.0, 0.79])
