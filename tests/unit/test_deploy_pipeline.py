"""Regression tests for Phi-0 → G1 deploy core pipeline (src/phi0/deploy/).

Coverage map (modules under test):
  gmr_retarget        FK → human_data, upsample, translate
  ref_traj_builder    postprocess, denorm_to_deploy_qpos (smpl / qpos)
  gt_io               GT backends + denorm_to_human_frames
  zmq_protocol        wire encode/decode
  dex3_gripper        20→50 upsample, 50-d merge
  g1_qpos_teacher     offline WBC → 360:396 (no GMR)
"""

from __future__ import annotations

import numpy as np
import pytest

from phi0.data.groot_unified_io import pack_from_groot_teleop_row
from phi0.deploy.dex3_gripper import (
    merge_body_and_gripper_qpos,
    upsample_rows_20_to_50,
    wbc_hand7_to_deploy,
)
from phi0.deploy.gmr_retarget import (
    GMR_SMPLX_BODY_NAMES,
    human_height_from_betas,
    translate_human_data_sequence,
    unified_chunk_to_gmr_human_data_list,
    upsample_qpos_20_to_50,
)
from phi0.deploy.gt_io import (
    XperienceHdf5GtBackend,
    build_eval_clip_context,
    build_gt_backend,
    denorm_to_human_frames,
    is_pick_tissue_unified_cfg,
)
from phi0.deploy.ref_traj_builder import (
    DEFAULT_QPOS_FULL,
    DEPLOY_MODE_QPOS,
    DEPLOY_MODE_SMPL,
    PostprocessConfig,
    denorm_to_deploy_qpos,
    denorm_to_gmr_qpos,
    human_frames_to_qpos20,
    postprocess_qpos_20,
)
from phi0.deploy.zmq_protocol import (
    STREAM_FORMAT_QPOS,
    decode_message,
    encode_done,
    encode_meta,
    encode_qpos_frame,
)
from phi0.schema.unified_action_schema import write_g1_body_qpos_36


def _mock_retarget(human_data: dict) -> np.ndarray:
    q = DEFAULT_QPOS_FULL.copy()
    q[:3] = np.asarray(human_data["pelvis"][0], dtype=np.float32)
    return q


def _synthetic_human_frames(n: int = 3) -> list[dict]:
    frames: list[dict] = []
    for t in range(n):
        pos = np.array([0.02 * t, 0.01 * t, 0.9 + 0.01 * t], dtype=np.float32)
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        frames.append({name: (pos.copy(), quat.copy()) for name in GMR_SMPLX_BODY_NAMES})
    return frames


def _groot_unified_row() -> np.ndarray:
    wbc = np.linspace(0.0, 1.0, 43, dtype=np.float32)
    joints = np.zeros((24, 3), dtype=np.float32)
    joints[0] = [0.1, 0.0, 0.88]
    row = {
        "teleop.smpl_joints": joints,
        "teleop.body_quat_w": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "teleop.smpl_pose": np.zeros(63, dtype=np.float32),
        "teleop.smpl_frame_index": np.array([42], dtype=np.int64),
        "action.wbc": wbc,
    }
    return pack_from_groot_teleop_row(row).action


# --- gmr_retarget ---


def test_translate_human_data_pelvis_at_origin_frame0():
    frames = _synthetic_human_frames(3)
    out = translate_human_data_sequence(frames)
    p0, _ = out[0]["pelvis"]
    assert np.allclose(p0[:2], 0.0, atol=1e-5)
    assert p0[2] >= 0.0


def test_unified_chunk_to_gmr_human_data_list(skeleton_constants):
    unified = np.stack([_groot_unified_row(), _groot_unified_row()], axis=0)
    anchor = np.array([0.2, -0.1, 0.0], dtype=np.float32)
    frames = unified_chunk_to_gmr_human_data_list(
        unified,
        state_root_trans_world=anchor,
        constants=skeleton_constants,
    )
    assert len(frames) == 2
    assert set(frames[0].keys()) == set(GMR_SMPLX_BODY_NAMES)
    pelvis_pos = frames[0]["pelvis"][0]
    assert np.all(np.isfinite(pelvis_pos))


def test_upsample_qpos_20_to_50_endpoints_and_shape():
    q20 = np.tile(DEFAULT_QPOS_FULL, (4, 1)).astype(np.float32)
    for i in range(4):
        q20[i, 0] = float(i)
    q50 = upsample_qpos_20_to_50(q20)
    assert q50.shape[0] == int(round((4 - 1) * 2.5) + 1)
    assert np.allclose(q50[0], q20[0], atol=1e-5)
    assert np.allclose(q50[-1], q20[-1], atol=1e-5)


def test_human_height_from_betas():
    assert human_height_from_betas(np.zeros(16, dtype=np.float32)) == pytest.approx(1.66)
    assert human_height_from_betas(np.array([0.5], dtype=np.float32)) == pytest.approx(1.71)


# --- ref_traj_builder deploy modes ---


def test_denorm_to_deploy_qpos_smpl_delegates_to_gmr_path(
    synthetic_hdf5, skeleton_constants
):
    backend = XperienceHdf5GtBackend(
        hdf5_path=str(synthetic_hdf5),
        native_start=0,
        native_fps=20.0,
        control_fps=20.0,
    )
    seq = backend.load_gt_unified_sequence(num_frames=4, proprio_w=4, chunk_h=4)
    cfg = PostprocessConfig(ema_alpha=0.55)
    via_smpl, _ = denorm_to_deploy_qpos(
        seq,
        backend,
        _mock_retarget,
        deploy_mode=DEPLOY_MODE_SMPL,
        proprio_w=4,
        chunk_h=4,
        constants=skeleton_constants,
        motion_deploy=True,
        postprocess=cfg,
    )
    via_gmr, _ = denorm_to_gmr_qpos(
        seq,
        backend,
        _mock_retarget,
        proprio_w=4,
        chunk_h=4,
        constants=skeleton_constants,
        motion_deploy=True,
        postprocess=cfg,
    )
    assert np.allclose(via_smpl, via_gmr, atol=1e-5)


def test_denorm_to_deploy_qpos_qpos_mode_applies_ema():
    qpos = np.tile(DEFAULT_QPOS_FULL.astype(np.float32), (4, 1))
    qpos[:, 7] = np.linspace(0.0, 0.4, 4, dtype=np.float32)
    unified = np.zeros((4, 512), dtype=np.float32)
    for i in range(4):
        write_g1_body_qpos_36(unified[i], qpos[i])

    raw, _ = denorm_to_deploy_qpos(
        unified,
        None,
        None,
        deploy_mode=DEPLOY_MODE_QPOS,
        proprio_w=0,
        chunk_h=4,
        constants={},
        motion_deploy=False,
        postprocess=PostprocessConfig(ema_alpha=None),
    )
    smooth, _ = denorm_to_deploy_qpos(
        unified,
        None,
        None,
        deploy_mode=DEPLOY_MODE_QPOS,
        proprio_w=0,
        chunk_h=4,
        constants={},
        motion_deploy=False,
        postprocess=PostprocessConfig(ema_alpha=0.55),
    )
    assert not np.allclose(raw, smooth)
    assert np.allclose(raw[0], qpos[0])


def test_denorm_to_deploy_qpos_invalid_mode_raises():
    with pytest.raises(ValueError, match="deploy_mode"):
        denorm_to_deploy_qpos(
            np.zeros((1, 512), dtype=np.float32),
            None,
            None,
            deploy_mode="invalid",
            proprio_w=0,
            chunk_h=1,
            constants={},
            motion_deploy=False,
        )


def test_denorm_to_deploy_qpos_smpl_requires_retarget_fn(synthetic_hdf5, skeleton_constants):
    backend = XperienceHdf5GtBackend(
        hdf5_path=str(synthetic_hdf5),
        native_start=0,
        native_fps=20.0,
        control_fps=20.0,
    )
    seq = backend.load_gt_unified_sequence(num_frames=2, proprio_w=2, chunk_h=2)
    with pytest.raises(ValueError, match="retarget_fn"):
        denorm_to_deploy_qpos(
            seq,
            backend,
            None,
            deploy_mode=DEPLOY_MODE_SMPL,
            proprio_w=2,
            chunk_h=2,
            constants=skeleton_constants,
            motion_deploy=True,
        )


# --- gt_io wiring ---


def test_is_pick_tissue_unified_cfg():
    assert is_pick_tissue_unified_cfg({"dataset": "pick_tissue_unified"})
    assert not is_pick_tissue_unified_cfg({"dataset": "xperience_unified"})


def test_build_gt_backend_xperience(synthetic_hdf5):
    ctx = build_eval_clip_context(
        {"dataset": "xperience_unified"},
        {"idx": 2},
        hdf5_path=str(synthetic_hdf5),
        native_fps=20.0,
        control_fps=20.0,
    )
    backend = build_gt_backend(ctx)
    seq = backend.load_gt_unified_sequence(num_frames=3, proprio_w=2, chunk_h=2)
    assert seq.shape == (3, 512)


def test_denorm_human_frames_produces_gmr_bodies(synthetic_hdf5, skeleton_constants):
    backend = XperienceHdf5GtBackend(
        hdf5_path=str(synthetic_hdf5),
        native_start=0,
        native_fps=20.0,
        control_fps=20.0,
    )
    seq = backend.load_gt_unified_sequence(num_frames=3, proprio_w=2, chunk_h=3)
    human_frames, root_quats = denorm_to_human_frames(
        seq,
        backend,
        proprio_w=2,
        chunk_h=3,
        constants=skeleton_constants,
        motion_deploy=True,
    )
    assert len(human_frames) == 3
    assert root_quats.shape == (3, 4)
    assert "pelvis" in human_frames[0]


# --- zmq_protocol ---


def test_encode_done_roundtrip():
    msg = decode_message(encode_done(num_frames=128))
    assert msg["type"] == "done"
    assert msg["num_frames"] == 128


def test_encode_meta_deploy_mode_field():
    msg = decode_message(
        encode_meta(
            clip_idx=0,
            num_frames=10,
            stream_format=STREAM_FORMAT_QPOS,
            deploy_mode=DEPLOY_MODE_QPOS,
        )
    )
    assert msg["deploy_mode"] == DEPLOY_MODE_QPOS


def test_zmq_burst_meta_and_frames_roundtrip():
    meta = decode_message(
        encode_meta(clip_idx=0, num_frames=2, control_fps=50.0, stream_format=STREAM_FORMAT_QPOS)
    )
    assert meta["num_frames"] == 2
    frames = []
    for i in range(2):
        q = DEFAULT_QPOS_FULL + i * 0.01
        frames.append(decode_message(encode_qpos_frame(i, q, is_last=(i == 1))))
    assert frames[0]["seq"] == 0
    assert frames[1]["is_last"] is True
    assert np.allclose(frames[1]["qpos"], DEFAULT_QPOS_FULL + 0.01)


# --- dex3_gripper ---


def test_upsample_rows_20_to_50_gripper():
    rows = np.linspace(0, 1, 20, dtype=np.float32).reshape(20, 1)
    out = upsample_rows_20_to_50(rows)
    assert out.shape[0] == int(round(19 * 2.5) + 1)
    assert out[0, 0] == pytest.approx(0.0)
    assert out[-1, 0] == pytest.approx(1.0)


def test_wbc_hand7_to_deploy_perm():
    wbc = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.float32)
    np.testing.assert_allclose(wbc_hand7_to_deploy(wbc), [4, 5, 6, 0, 1, 2, 3])


def test_merge_body_gripper_qpos50_layout():
    body = np.arange(36, dtype=np.float32)
    grip = np.linspace(0, 1, 14, dtype=np.float32)
    q50 = merge_body_and_gripper_qpos(body, grip)
    assert q50.shape == (50,)
    assert np.allclose(q50[:36], body)
    assert np.allclose(q50[36:43], wbc_hand7_to_deploy(grip[:7]))
    assert np.allclose(q50[43:50], wbc_hand7_to_deploy(grip[7:]))


# --- g1 qpos from WBC (recorded joints, no GMR) ---


def test_teacher_writes_wbc_qpos_slice():
    from phi0.data.g1_qpos_from_wbc import body_dof29_from_wbc43
    from phi0.data.g1_qpos_teacher import attach_g1_qpos_to_parquet_rows
    from phi0.data.groot_unified_io import pack_from_groot_teleop_row
    from phi0.schema.unified_action_schema import unpack_g1_body_qpos_36

    wbc = np.linspace(-0.3, 0.3, 43, dtype=np.float32)
    joints = np.zeros((24, 3), dtype=np.float32)
    joints[0] = [0.1, 0.0, 0.88]
    groot = {
        "action.wbc": wbc,
        "observation.state": wbc,
        "observation.root_orientation": np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32),
        "teleop.smpl_joints": joints.reshape(-1),
        "teleop.body_quat_w": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "teleop.smpl_pose": np.zeros(63, dtype=np.float32),
        "teleop.smpl_frame_index": np.array([1], dtype=np.int64),
    }
    gt = pack_from_groot_teleop_row(groot)
    export = {
        "unified_action": gt.action.astype(np.float32).tolist(),
        "state_root_trans_world": gt.state_root_trans_world.astype(np.float32).tolist(),
        "target_root_trans_world": gt.target_root_trans_world.astype(np.float32).tolist(),
        "betas": gt.betas.astype(np.float32).tolist(),
    }
    attach_g1_qpos_to_parquet_rows([export], [groot])
    qpos = unpack_g1_body_qpos_36(np.asarray(export["unified_action"], dtype=np.float32))
    assert np.allclose(qpos[7:], body_dof29_from_wbc43(wbc))


# --- end-to-end synthetic regression ---


def test_full_smpl_deploy_stages_commute(synthetic_hdf5, skeleton_constants):
    """GT unified → human_data → mock GMR → postprocess; single entry vs staged."""
    backend = XperienceHdf5GtBackend(
        hdf5_path=str(synthetic_hdf5),
        native_start=0,
        native_fps=20.0,
        control_fps=20.0,
    )
    seq = backend.load_gt_unified_sequence(num_frames=5, proprio_w=3, chunk_h=5)
    cfg = PostprocessConfig(ema_alpha=0.55)

    human_frames, root_quats = denorm_to_human_frames(
        seq,
        backend,
        proprio_w=3,
        chunk_h=5,
        constants=skeleton_constants,
        motion_deploy=True,
    )
    staged = postprocess_qpos_20(
        human_frames_to_qpos20(human_frames, _mock_retarget),
        root_quats,
        cfg,
    )
    pipeline, _ = denorm_to_deploy_qpos(
        seq,
        backend,
        _mock_retarget,
        deploy_mode=DEPLOY_MODE_SMPL,
        proprio_w=3,
        chunk_h=5,
        constants=skeleton_constants,
        motion_deploy=True,
        postprocess=cfg,
    )
    assert np.allclose(pipeline, staged, atol=1e-5)
