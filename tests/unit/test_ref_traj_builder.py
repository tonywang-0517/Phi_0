"""Unit tests for ref_traj_builder and qpos ZMQ protocol."""

from __future__ import annotations

import numpy as np
import pytest

from phi0.deploy.zmq_protocol import (
    STREAM_FORMAT_HUMAN_DATA,
    STREAM_FORMAT_QPOS,
    decode_message,
    encode_meta,
    encode_qpos_frame,
    stream_format_from_meta,
)
from phi0.deploy.ref_traj_builder import (
    DEFAULT_QPOS_FULL,
    DEPLOY_MODE_QPOS,
    PostprocessConfig,
    align_qpos_trajectory_to_default,
    apply_ema_qpos,
    denorm_to_gmr_qpos,
    has_g1_body_qpos_labels,
    human_frames_to_qpos20,
    inject_fk_root_quat,
    postprocess_qpos_20,
    unified_actions_to_qpos36,
)
from phi0.schema.unified_action_schema import write_g1_body_qpos_36
from phi0.deploy.gmr_retarget import GMR_SMPLX_BODY_NAMES


def _mock_retarget(human_data: dict) -> np.ndarray:
    q = DEFAULT_QPOS_FULL.copy()
    q[:3] = np.asarray(human_data["pelvis"][0], dtype=np.float32)
    return q


def _synthetic_human_frames(n: int = 4) -> list[dict]:
    frames: list[dict] = []
    for t in range(n):
        pos = np.array([0.01 * t, 0.0, 0.9], dtype=np.float32)
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        frames.append({name: (pos, quat) for name in GMR_SMPLX_BODY_NAMES})
    return frames


def test_protocol_qpos_roundtrip():
    q = np.linspace(-0.5, 0.5, 36, dtype=np.float32)
    g = np.linspace(-0.2, 0.2, 14, dtype=np.float32)
    payload = encode_qpos_frame(3, q, gripper_joints_14=g, is_last=True)
    msg = decode_message(payload)
    assert msg["seq"] == 3
    assert msg["is_last"] is True
    assert np.allclose(msg["qpos"], q)
    assert np.allclose(msg["gripper_joints_14"], g)


def test_encode_meta_qpos_fields():
    payload = encode_meta(
        clip_idx=1,
        num_frames=32,
        control_fps=50.0,
        stream_format=STREAM_FORMAT_QPOS,
        qpos_postprocessed=True,
        ema_alpha=0.55,
    )
    msg = decode_message(payload)
    assert msg["stream_format"] == STREAM_FORMAT_QPOS
    assert msg["qpos_postprocessed"] is True
    assert msg["qpos_dim"] == 36
    assert msg["ema_alpha"] == pytest.approx(0.55)


def test_stream_format_from_meta():
    assert stream_format_from_meta(None) == STREAM_FORMAT_HUMAN_DATA
    assert stream_format_from_meta({"stream_format": "qpos"}) == STREAM_FORMAT_QPOS
    assert stream_format_from_meta({"body_names": []}) == STREAM_FORMAT_HUMAN_DATA


def test_align_qpos_frame0_matches_default():
    q = np.tile(DEFAULT_QPOS_FULL, (3, 1)).copy()
    q[1, :3] += np.array([0.1, 0.05, -0.02], dtype=np.float32)
    q[2, :3] += np.array([0.2, 0.1, -0.04], dtype=np.float32)
    out = align_qpos_trajectory_to_default(q)
    assert np.allclose(out[0, :7], DEFAULT_QPOS_FULL[:7], atol=1e-5)
    assert np.allclose(out[1, :3] - out[0, :3], q[1, :3] - q[0, :3], atol=1e-4)


def test_inject_fk_root_quat_preserves_frame0_default():
    q = np.tile(DEFAULT_QPOS_FULL, (2, 1)).copy()
    rq = np.array([[1.0, 0.0, 0.0, 0.0], [0.9239, 0.0, 0.0, 0.3827]], dtype=np.float32)
    out = inject_fk_root_quat(q, rq)
    assert np.allclose(out[0, 3:7], DEFAULT_QPOS_FULL[3:7], atol=1e-5)


def test_apply_ema_qpos_single_frame_unchanged():
    q = DEFAULT_QPOS_FULL.reshape(1, -1)
    assert np.allclose(apply_ema_qpos(q, alpha=0.55), q)


def test_human_frames_to_qpos20_mock():
    frames = _synthetic_human_frames(3)
    qpos = human_frames_to_qpos20(frames, _mock_retarget)
    assert qpos.shape == (3, 36)
    assert np.allclose(qpos[0, :3], frames[0]["pelvis"][0])
    assert np.allclose(qpos[2, :3], frames[2]["pelvis"][0])


def test_postprocess_qpos_20_noop_when_disabled():
    q = np.tile(DEFAULT_QPOS_FULL, (2, 1)).copy()
    rq = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2, 1))
    cfg = PostprocessConfig(align_to_default=False, inject_fk_root=False, ema_alpha=None)
    out = postprocess_qpos_20(q, rq, cfg)
    assert np.allclose(out, q)


def test_denorm_to_gmr_qpos_matches_manual_pipeline(synthetic_hdf5, skeleton_constants):
    from phi0.deploy.gt_io import XperienceHdf5GtBackend, denorm_to_human_frames

    backend = XperienceHdf5GtBackend(
        hdf5_path=str(synthetic_hdf5),
        native_start=0,
        native_fps=20.0,
        control_fps=20.0,
    )
    seq = backend.load_gt_unified_sequence(num_frames=4, proprio_w=4, chunk_h=4)
    human_frames, root_quats = denorm_to_human_frames(
        seq,
        backend,
        proprio_w=4,
        chunk_h=4,
        constants=skeleton_constants,
        motion_deploy=True,
    )
    manual_raw = human_frames_to_qpos20(human_frames, _mock_retarget)
    manual = postprocess_qpos_20(manual_raw, root_quats, PostprocessConfig(ema_alpha=0.55))

    pipeline, rq2 = denorm_to_gmr_qpos(
        seq,
        backend,
        _mock_retarget,
        proprio_w=4,
        chunk_h=4,
        constants=skeleton_constants,
        motion_deploy=True,
        postprocess=PostprocessConfig(ema_alpha=0.55),
    )
    assert np.allclose(rq2, root_quats)
    assert np.allclose(pipeline, manual, atol=1e-5)


def test_unified_actions_to_qpos36_roundtrip():
    qpos = np.tile(DEFAULT_QPOS_FULL.astype(np.float32), (4, 1))
    qpos[:, 7] += np.linspace(0, 0.1, 4, dtype=np.float32)
    unified = np.zeros((4, 512), dtype=np.float32)
    for i in range(4):
        write_g1_body_qpos_36(unified[i], qpos[i])
    assert has_g1_body_qpos_labels(unified)
    out = unified_actions_to_qpos36(unified)
    np.testing.assert_allclose(out, qpos, rtol=1e-6, atol=1e-6)


def test_unified_actions_to_qpos36_missing_raises():
    unified = np.zeros((2, 512), dtype=np.float32)
    with pytest.raises(ValueError, match="g1_body_qpos_36"):
        unified_actions_to_qpos36(unified)


def test_denorm_to_deploy_qpos_qpos_mode():
    from phi0.deploy.ref_traj_builder import denorm_to_deploy_qpos

    qpos = np.tile(DEFAULT_QPOS_FULL.astype(np.float32), (3, 1))
    unified = np.zeros((3, 512), dtype=np.float32)
    for i in range(3):
        write_g1_body_qpos_36(unified[i], qpos[i])
    out, human = denorm_to_deploy_qpos(
        unified,
        backend=None,
        retarget_fn=None,
        deploy_mode=DEPLOY_MODE_QPOS,
        proprio_w=1,
        chunk_h=3,
        constants={},
        motion_deploy=False,
    )
    assert human is None
    np.testing.assert_allclose(out, qpos, rtol=1e-6, atol=1e-6)
