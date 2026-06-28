"""Unit tests for unified SMPL-H / SONIC 6D rotation action schema."""

from __future__ import annotations

import numpy as np
import pytest

from phi0.schema.unified_action_schema import (
    D_UNIFIED,
    NUM_BODY_JOINTS,
    NUM_G1_BODY_QPOS,
    NUM_G1_GRIPPER_JOINTS,
    NUM_JOINTS_SMPL24,
    NUM_JOINTS_SMPLH,
    NUM_LOCAL_JOINTS,
    RESERVED_PADDING_DIM,
    RESERVED_TAIL_DIM,
    SONIC_MOTION_TOKEN_DIM,
    ROT6D_DIM,
    ROT6D_IDENTITY,
    SLICES,
    SUPERVISED_POSE_DIM,
    body_rot6d_to_smpl_pose_aa,
    dim_mask_for_dataset,
    export_sonic_zmq_v3_fields,
    joints_pelvis_local_52_from_unified,
    joints_world_52_from_unified,
    matrix_to_rot6d,
    pack_unified,
    pack_unified_with_g1_gripper,
    pelvis_local_joints_to_world,
    quat_wxyz_to_rot6d,
    root_trans_world_from_unified,
    rot6d_to_matrix,
    rot6d_to_quat_wxyz,
    schema_field_table,
    smpl24_joints_local_from_unified,
    smpl_pose_aa_from_unified,
    unpack_body_rot6d_local,
    unpack_g1_body_qpos_36,
    unpack_g1_body_qpos_dof_29,
    unpack_g1_body_qpos_root_7,
    unpack_g1_gripper_joints_14,
    unpack_joint_rot6d_local_51,
    unpack_root_quat_wxyz,
    unpack_root_rot6d,
    unpack_root_trans_local,
    unpack_sonic_motion_token_64,
    world_joints_to_pelvis_local_52,
    write_g1_body_qpos_36,
    write_g1_gripper_joints_14,
    write_sonic_motion_token_64,
)
from phi0.viz.smplh_fk import joints_from_d_raw_batch, load_skeleton_constants, pack_xperience_frame_quat


def test_dim_and_slices_cover_buffer():
    assert D_UNIFIED == 512
    assert SUPERVISED_POSE_DIM == 346
    assert NUM_G1_GRIPPER_JOINTS == 14
    assert RESERVED_TAIL_DIM == 116
    assert SONIC_MOTION_TOKEN_DIM == 64
    assert RESERVED_PADDING_DIM == 52
    assert RESERVED_TAIL_DIM == SONIC_MOTION_TOKEN_DIM + RESERVED_PADDING_DIM
    assert D_UNIFIED - SUPERVISED_POSE_DIM == NUM_G1_GRIPPER_JOINTS + NUM_G1_BODY_QPOS + RESERVED_TAIL_DIM
    covered = np.zeros(D_UNIFIED, dtype=bool)
    for s, e in SLICES.values():
        covered[s:e] = True
    assert covered.all()
    assert SLICES["root_trans_local"] == (0, 3)
    assert SLICES["root_rot6d"] == (3, 9)
    assert SLICES["joint_rot6d_local_51"] == (9, 315)
    assert SLICES["contacts_body21"] == (315, 336)
    assert SLICES["g1_gripper_joints_14"] == (346, 360)
    assert SLICES["g1_body_qpos_36"] == (360, 396)
    assert SLICES["sonic_motion_token_64"] == (396, 460)
    assert SLICES["reserved_52"] == (460, 512)
    assert "joints_pelvis_local_52" not in SLICES
    assert "root_trans_world" not in SLICES


def test_rot6d_identity_roundtrip():
    m = rot6d_to_matrix(ROT6D_IDENTITY)
    assert np.allclose(m, np.eye(3), atol=1e-5)
    assert np.allclose(matrix_to_rot6d(np.eye(3)), ROT6D_IDENTITY, atol=1e-5)
    q = rot6d_to_quat_wxyz(ROT6D_IDENTITY)
    assert np.allclose(q, np.array([1, 0, 0, 0], dtype=np.float32), atol=1e-5)


def test_quat_rot6d_roundtrip():
    q = np.array([0.9238795, 0.0, 0.3826834, 0.0], dtype=np.float32)  # 45° about Y
    r6 = quat_wxyz_to_rot6d(q)
    q2 = rot6d_to_quat_wxyz(r6)
    # quat double cover
    assert np.allclose(np.abs(q), np.abs(q2), atol=1e-4) or np.allclose(q, -q2, atol=1e-4)


def test_root_trans_local_vs_state_t():
    state = np.array([1.0, 2.0, 0.8], dtype=np.float32)
    target = np.array([1.5, 2.3, 0.95], dtype=np.float32)
    ident6 = np.tile(ROT6D_IDENTITY, (NUM_BODY_JOINTS, 1))

    packed = pack_unified(
        root_trans_local=target - state,
        root_rot6d=ROT6D_IDENTITY,
        body_rot6d_local=ident6,
        left_hand_rot6d_local=np.tile(ROT6D_IDENTITY, (15, 1)),
        right_hand_rot6d_local=np.tile(ROT6D_IDENTITY, (15, 1)),
    )
    assert np.allclose(unpack_root_trans_local(packed), target - state)
    assert np.allclose(root_trans_world_from_unified(packed, state), target)

    packed2 = pack_unified(
        root_trans_world=target,
        state_root_trans_world=state,
        root_rot6d=ROT6D_IDENTITY,
        body_rot6d_local=ident6,
        left_hand_rot6d_local=np.tile(ROT6D_IDENTITY, (15, 1)),
        right_hand_rot6d_local=np.tile(ROT6D_IDENTITY, (15, 1)),
    )
    assert np.allclose(unpack_root_trans_local(packed2), target - state)


def test_pack_unpack_rot6d_roundtrip():
    rng = np.random.RandomState(2)
    state = rng.randn(3).astype(np.float32)
    root_t = state + np.array([0.1, -0.05, 0.02], dtype=np.float32)
    body6 = rng.randn(NUM_BODY_JOINTS, ROT6D_DIM).astype(np.float32)
    lh6 = rng.randn(15, ROT6D_DIM).astype(np.float32)
    rh6 = rng.randn(15, ROT6D_DIM).astype(np.float32)
    body6 = matrix_to_rot6d(rot6d_to_matrix(body6))
    lh6 = matrix_to_rot6d(rot6d_to_matrix(lh6))
    rh6 = matrix_to_rot6d(rot6d_to_matrix(rh6))
    root6 = ROT6D_IDENTITY.copy()
    contacts = rng.rand(21).astype(np.float32)

    packed = pack_unified(
        root_trans_local=root_t - state,
        root_rot6d=root6,
        body_rot6d_local=body6,
        left_hand_rot6d_local=lh6,
        right_hand_rot6d_local=rh6,
        contacts_body21=contacts,
    )
    assert packed.shape == (D_UNIFIED,)
    assert np.allclose(packed[346:], 0.0)
    assert np.allclose(unpack_root_rot6d(packed), root6)
    assert np.allclose(unpack_body_rot6d_local(packed), body6, atol=1e-5)
    local51 = unpack_joint_rot6d_local_51(packed)
    assert local51.shape == (NUM_LOCAL_JOINTS, ROT6D_DIM)


def test_fk_positions_not_in_buffer():
    state = np.array([1.0, 2.0, 0.8], dtype=np.float32)
    ident6 = np.tile(ROT6D_IDENTITY, (NUM_BODY_JOINTS, 1))
    lh6 = np.tile(ROT6D_IDENTITY, (15, 1))
    rh6 = np.tile(ROT6D_IDENTITY, (15, 1))
    packed = pack_unified(
        root_trans_local=np.zeros(3, dtype=np.float32),
        root_rot6d=ROT6D_IDENTITY,
        body_rot6d_local=ident6,
        left_hand_rot6d_local=lh6,
        right_hand_rot6d_local=rh6,
    )
    local = joints_pelvis_local_52_from_unified(
        packed, state_root_trans_world=state
    )
    assert local.shape == (NUM_JOINTS_SMPLH, 3)
    assert local[0] == pytest.approx(0.0, abs=1e-5)
    world = joints_world_52_from_unified(packed, state_root_trans_world=state)
    assert np.allclose(world_joints_to_pelvis_local_52(world, state), local, atol=1e-5)


def test_fk_matches_legacy_quat_pipeline():
    root = np.array([0.3, -0.1, 0.95, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    body_q = np.tile(np.array([1, 0, 0, 0], np.float32), (21, 1))
    lh_q = np.tile(np.array([1, 0, 0, 0], np.float32), (15, 1))
    rh_q = lh_q.copy()
    betas = np.zeros(16, dtype=np.float32)

    d_quat = pack_xperience_frame_quat(root, body_q, lh_q, rh_q, betas, np.zeros(10, np.float32))
    constants = load_skeleton_constants()
    gt_world = joints_from_d_raw_batch(d_quat[None], constants, use_d_raw_betas=True)[0]

    packed = pack_unified(
        root_trans_local=np.zeros(3, dtype=np.float32),
        root_quat_wxyz=root[3:7],
        body_quats_wxyz=body_q,
        left_hand_quats_wxyz=lh_q,
        right_hand_quats_wxyz=rh_q,
    )
    pred_world = joints_world_52_from_unified(
        packed, state_root_trans_world=root[:3], betas=betas, constants=constants
    )
    assert np.allclose(pred_world, gt_world, atol=1e-4)

    pred_local = joints_pelvis_local_52_from_unified(
        packed, state_root_trans_world=root[:3], betas=betas, constants=constants
    )
    gt_local = world_joints_to_pelvis_local_52(gt_world, root[:3])
    assert np.allclose(pred_local, gt_local, atol=1e-4)


def test_smpl24_and_pose_derived_at_export():
    state = np.array([0.1, -0.2, 0.9], dtype=np.float32)
    ident6 = np.tile(ROT6D_IDENTITY, (NUM_BODY_JOINTS, 1))
    packed = pack_unified(
        root_trans_local=np.array([0.02, 0.0, -0.01], dtype=np.float32),
        root_rot6d=ROT6D_IDENTITY,
        body_rot6d_local=ident6,
        left_hand_rot6d_local=np.tile(ROT6D_IDENTITY, (15, 1)),
        right_hand_rot6d_local=np.tile(ROT6D_IDENTITY, (15, 1)),
    )
    s24 = smpl24_joints_local_from_unified(packed, state_root_trans_world=state)
    assert s24.shape == (NUM_JOINTS_SMPL24, 3)
    assert s24[0] == pytest.approx(0.0, abs=1e-5)

    aa = smpl_pose_aa_from_unified(packed)
    assert aa.shape == (NUM_BODY_JOINTS, 3)
    assert np.allclose(aa, 0.0, atol=1e-5)
    sonic = export_sonic_zmq_v3_fields(packed, state_root_trans_world=state)
    assert np.allclose(sonic["smpl_joints"], s24)
    assert np.allclose(sonic["smpl_pose"], aa)
    assert np.allclose(sonic["body_quat_w"], unpack_root_quat_wxyz(packed), atol=1e-5)
    assert np.allclose(
        sonic["root_trans"],
        root_trans_world_from_unified(packed, state),
        atol=1e-5,
    )


def test_body_rot6d_identity_axis_angle():
    ident6 = np.tile(ROT6D_IDENTITY, (NUM_BODY_JOINTS, 1))
    aa = body_rot6d_to_smpl_pose_aa(ident6)
    assert aa.shape == (NUM_BODY_JOINTS, 3)
    assert np.allclose(aa, 0.0, atol=1e-5)


def test_dim_mask_excludes_unsupervised_tail():
    mask_g1 = dim_mask_for_dataset("g1_sonic")
    assert not mask_g1[0:3].any()  # no SMPL pelvis odometry GT on pick-tissue
    assert mask_g1[3:336].all()
    assert mask_g1[336:346].sum() == 0  # no tactile
    assert mask_g1[346:360].all()  # G1 gripper supervised
    assert mask_g1[360:396].all()  # G1 body qpos supervised
    assert mask_g1[396:460].all()  # SONIC motion_token supervised
    assert not mask_g1[460:].any()  # reserved padding masked

    mask_xp = dim_mask_for_dataset("xperience")
    assert mask_xp[0:346].all()
    assert not mask_xp[346:360].any()  # no gripper on xperience
    assert not mask_xp[360:396].any()  # no g1 qpos on xperience
    assert not mask_xp[396:460].any()  # no sonic token on xperience
    assert not mask_xp[460:].any()
    assert mask_xp.shape == (512,)


def test_g1_gripper_pack_unpack():
    ident6 = np.tile(ROT6D_IDENTITY, (NUM_BODY_JOINTS, 1))
    base = pack_unified(
        root_trans_local=np.zeros(3, dtype=np.float32),
        root_rot6d=ROT6D_IDENTITY,
        body_rot6d_local=ident6,
        left_hand_rot6d_local=np.tile(ROT6D_IDENTITY, (15, 1)),
        right_hand_rot6d_local=np.tile(ROT6D_IDENTITY, (15, 1)),
    )
    gripper = np.linspace(-0.5, 0.5, NUM_G1_GRIPPER_JOINTS, dtype=np.float32)
    write_g1_gripper_joints_14(base, gripper)
    assert np.allclose(unpack_g1_gripper_joints_14(base), gripper)
    assert np.allclose(base[396:460], 0.0)
    assert np.allclose(base[460:], 0.0)

    token = np.linspace(-0.3, 0.3, SONIC_MOTION_TOKEN_DIM, dtype=np.float32)
    write_sonic_motion_token_64(base, token)
    assert np.allclose(unpack_sonic_motion_token_64(base), token)

    qpos = np.linspace(-0.2, 0.2, NUM_G1_BODY_QPOS, dtype=np.float32)
    write_g1_body_qpos_36(base, qpos)
    assert np.allclose(unpack_g1_body_qpos_36(base), qpos)
    assert np.allclose(unpack_g1_body_qpos_root_7(base), qpos[:7])
    assert np.allclose(unpack_g1_body_qpos_dof_29(base), qpos[7:])

    attached = pack_unified_with_g1_gripper(base, g1_gripper_joints_14=gripper, g1_body_qpos_36=qpos)
    assert np.allclose(unpack_g1_gripper_joints_14(attached), gripper)
    assert np.allclose(unpack_g1_body_qpos_36(attached), qpos)


def test_schema_field_table_rot6d():
    table = schema_field_table()
    assert "6D" in table or "root_rot6d" in table
    assert "root_trans_local" in table
    assert "State_t" in table
    assert "body_rot6d_local" in table
    assert "FK-derived" in table
