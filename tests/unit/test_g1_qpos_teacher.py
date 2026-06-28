"""Unit tests for G1 qpos@36 from recorded WBC (no GMR)."""

from __future__ import annotations

import numpy as np

from phi0.data.g1_qpos_from_wbc import (
    body_dof29_from_wbc43,
    g1_body_qpos36_from_groot_row,
    g1_body_qpos36_from_unified_frame,
)
from phi0.data.g1_qpos_teacher import attach_g1_qpos_to_parquet_rows
from phi0.data.groot_unified_io import pack_from_groot_teleop_row
from phi0.schema.unified_action_schema import (
    NUM_G1_BODY_QPOS,
    unpack_g1_body_qpos_36,
    unpack_root_quat_wxyz,
)


def _sample_groot_row() -> dict:
    wbc = np.linspace(-0.5, 0.5, 43, dtype=np.float32)
    smpl_joints = np.zeros(72, dtype=np.float32)
    smpl_joints[0:3] = np.array([0.05, -0.02, 0.91], dtype=np.float32)
    return {
        "action.wbc": wbc,
        "observation.state": wbc + 0.01,
        "observation.root_orientation": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "teleop.smpl_joints": smpl_joints,
        "teleop.body_quat_w": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "teleop.smpl_pose": np.zeros(63, dtype=np.float32),
        "teleop.smpl_frame_index": np.array([1], dtype=np.int64),
    }


def test_body_dof29_matches_wbc_slices():
    wbc = np.arange(43, dtype=np.float32)
    dof = body_dof29_from_wbc43(wbc)
    assert dof.shape == (29,)
    assert np.allclose(dof[:15], wbc[:15])
    assert np.allclose(dof[15:22], wbc[15:22])
    assert np.allclose(dof[22:], wbc[29:36])


def test_g1_body_qpos36_from_groot_row():
    row = _sample_groot_row()
    q = g1_body_qpos36_from_groot_row(row)
    assert q.shape == (NUM_G1_BODY_QPOS,)
    assert np.allclose(q[:3], row["teleop.smpl_joints"][:3])
    assert np.allclose(q[3:7], [1, 0, 0, 0])
    assert np.allclose(q[7:], body_dof29_from_wbc43(row["action.wbc"]))


def test_g1_body_qpos36_root_from_unified_action():
    groot = _sample_groot_row()
    gt = pack_from_groot_teleop_row(groot)
    q = g1_body_qpos36_from_unified_frame(
        gt.action,
        groot["observation.state"],
        state_root_trans_world=gt.state_root_trans_world,
        root_quat_wxyz=groot["observation.root_orientation"],
    )
    assert np.allclose(q[:3], gt.target_root_trans_world)
    assert np.allclose(q[3:7], groot["observation.root_orientation"])
    assert np.allclose(q[7:], body_dof29_from_wbc43(groot["observation.state"]))


def test_attach_g1_qpos_to_parquet_rows():
    groot = _sample_groot_row()
    gt = pack_from_groot_teleop_row(groot)
    export = {
        "unified_action": gt.action.astype(np.float32).tolist(),
        "state_root_trans_world": gt.state_root_trans_world.astype(np.float32).tolist(),
        "target_root_trans_world": gt.target_root_trans_world.astype(np.float32).tolist(),
        "betas": gt.betas.astype(np.float32).tolist(),
    }
    attach_g1_qpos_to_parquet_rows([export], [groot])
    qpos = unpack_g1_body_qpos_36(np.asarray(export["unified_action"], dtype=np.float32))
    assert np.allclose(qpos[:3], gt.target_root_trans_world)
    assert np.allclose(qpos[3:7], groot["observation.root_orientation"])
    assert np.allclose(qpos[7:], body_dof29_from_wbc43(groot["observation.state"]))
