"""Unit tests for pick-tissue GT reader and ZMQ publisher gt_io."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from phi0.deploy.gt_io import (
    PickTissueGtBackend,
    XperienceHdf5GtBackend,
    build_eval_clip_context,
    build_gt_backend,
    control_index_to_native,
    denorm_to_human_frames,
    pick_tissue_clip_action_matches_deploy,
)
from phi0.deploy.pick_tissue_gt import (
    PickTissueGtReader,
    control_index_to_global_frame,
    reader_from_data_cfg,
)
from phi0.data.xperience_unified_gt import pack_xperience_unified_frame_gt
from phi0.schema.unified_action_schema import (
    root_trans_world_from_unified,
    unpack_root_trans_local,
)

PICK_TISSUE_ROOT = Path("/mnt/data2/wpy/workspace/Isaac-GR00T/data")
PICK_TISSUE_REPO = "pick_tissue_xperience_unified"
PICK_TISSUE_PATH = PICK_TISSUE_ROOT / PICK_TISSUE_REPO


def _pick_tissue_available() -> bool:
    return (PICK_TISSUE_PATH / "meta" / "info.json").is_file()


def test_control_index_to_native_roundtrip():
    assert control_index_to_native(100, 0, native_fps=50.0, control_fps=50.0) == 100
    assert control_index_to_native(100, 4, native_fps=50.0, control_fps=20.0) == 110


def test_xperience_backend_matches_hdf5_pack(synthetic_hdf5: Path):
    native_start = 5
    backend = XperienceHdf5GtBackend(
        hdf5_path=str(synthetic_hdf5),
        native_start=native_start,
        native_fps=20.0,
        control_fps=20.0,
    )
    d_raw, anchor = backend.pack_deploy_frame(control_idx=3, state_control_idx=1)
    with h5py.File(synthetic_hdf5, "r") as f:
        gt = pack_xperience_unified_frame_gt(f, native_start + 3, state_t=native_start + 1)
    assert np.allclose(d_raw, gt.action)
    assert np.allclose(anchor, gt.state_root_trans_world)
    assert np.allclose(
        root_trans_world_from_unified(d_raw, anchor),
        gt.target_root_trans_world,
        atol=1e-5,
    )


def test_xperience_gt_sequence_chunk_anchors(synthetic_hdf5: Path):
    backend = XperienceHdf5GtBackend(
        hdf5_path=str(synthetic_hdf5),
        native_start=0,
        native_fps=20.0,
        control_fps=20.0,
    )
    seq = backend.load_gt_unified_sequence(num_frames=10, proprio_w=4, chunk_h=4)
    assert seq.shape == (10, 512)
    _, anchor_seg0 = backend.pack_deploy_frame(control_idx=4, state_control_idx=4)
    _, anchor_seg1 = backend.pack_deploy_frame(control_idx=8, state_control_idx=8)
    assert not np.allclose(anchor_seg0, anchor_seg1)
    local0 = unpack_root_trans_local(seq[0])
    world0 = root_trans_world_from_unified(seq[0], anchor_seg0)
    assert np.allclose(local0, world0 - anchor_seg0, atol=1e-5)


@pytest.mark.skipif(not _pick_tissue_available(), reason="pick-tissue dataset missing")
def test_pick_tissue_clip_actions_match_deploy_repack():
    from phi0.data.pick_tissue_unified import PickTissueUnifiedClipDataset

    clip_ds = PickTissueUnifiedClipDataset(
        root_dir=str(PICK_TISSUE_ROOT),
        repo_id=PICK_TISSUE_REPO,
        use_predecoded_video=True,
        cache_video=False,
        val=False,
    )
    item = clip_ds[0]
    reader = reader_from_data_cfg(
        {"pick_tissue_root": str(PICK_TISSUE_ROOT), "pick_tissue_repo_id": PICK_TISSUE_REPO}
    )
    span = reader.episode_span(int(item["idx"]))
    pick_tissue_clip_action_matches_deploy(
        reader,
        span,
        item["action"].numpy(),
        control_fps=float(item["control_fps"]),
    )


@pytest.mark.skipif(not _pick_tissue_available(), reason="pick-tissue dataset missing")
def test_pick_tissue_deploy_frame_anchor_consistency():
    reader = reader_from_data_cfg(
        {"pick_tissue_root": str(PICK_TISSUE_ROOT), "pick_tissue_repo_id": PICK_TISSUE_REPO}
    )
    span = reader.episode_span(0)
    fps = float(reader.native_fps)
    d_raw, anchor = reader.pack_deploy_frame(
        span, control_idx=10, state_control_idx=4, native_fps=fps, control_fps=fps
    )
    world = root_trans_world_from_unified(d_raw, anchor)
    global_t = control_index_to_global_frame(span.frame_start, 10, native_fps=fps, control_fps=fps)
    expected = reader.read_target_root(global_t, span)
    assert np.allclose(world, expected, atol=1e-4)


@pytest.mark.skipif(not _pick_tissue_available(), reason="pick-tissue dataset missing")
def test_pick_tissue_backend_gt_sequence_matches_clip_prefix():
    from phi0.data.pick_tissue_unified import PickTissueUnifiedClipDataset

    clip_ds = PickTissueUnifiedClipDataset(
        root_dir=str(PICK_TISSUE_ROOT),
        repo_id=PICK_TISSUE_REPO,
        use_predecoded_video=True,
        cache_video=False,
        val=False,
    )
    item = clip_ds[0]
    reader = reader_from_data_cfg(
        {"pick_tissue_root": str(PICK_TISSUE_ROOT), "pick_tissue_repo_id": PICK_TISSUE_REPO}
    )
    span = reader.episode_span(int(item["idx"]))
    backend = PickTissueGtBackend(
        reader=reader,
        span=span,
        native_fps=float(reader.native_fps),
        control_fps=float(item["control_fps"]),
    )
    proprio_w = 4
    chunk_h = 8
    num_frames = 12
    seq = backend.load_gt_unified_sequence(
        num_frames=num_frames, proprio_w=proprio_w, chunk_h=chunk_h
    )
    for i in range(num_frames):
        seg_start = (i // chunk_h) * chunk_h
        expected, _ = backend.pack_deploy_frame(
            control_idx=proprio_w + i,
            state_control_idx=proprio_w + seg_start,
        )
        assert np.allclose(seq[i], expected, atol=1e-4), (
            f"deploy GT frame {i} mismatch (max={np.max(np.abs(seq[i]-expected)):.6g})"
        )


@pytest.mark.skipif(not _pick_tissue_available(), reason="pick-tissue dataset missing")
def test_denorm_human_frames_pick_tissue_gt(skeleton_constants):
    from phi0.data.pick_tissue_unified import PickTissueUnifiedClipDataset

    clip_ds = PickTissueUnifiedClipDataset(
        root_dir=str(PICK_TISSUE_ROOT),
        repo_id=PICK_TISSUE_REPO,
        use_predecoded_video=True,
        cache_video=False,
        val=False,
    )
    item = clip_ds[0]
    ctx = build_eval_clip_context(
        {"dataset": "pick_tissue_unified", "pick_tissue_root": str(PICK_TISSUE_ROOT), "pick_tissue_repo_id": PICK_TISSUE_REPO},
        item,
        hdf5_path="",
        native_fps=50.0,
        control_fps=float(item["control_fps"]),
    )
    backend = build_gt_backend(ctx)
    proprio_w = 4
    chunk_h = 8
    gt_future = item["action"].numpy()[proprio_w : proprio_w + chunk_h]
    human_frames, root_quats = denorm_to_human_frames(
        gt_future,
        backend,
        proprio_w=proprio_w,
        chunk_h=chunk_h,
        constants=skeleton_constants,
        motion_deploy=True,
    )
    assert len(human_frames) == chunk_h
    assert root_quats.shape == (chunk_h, 4)
    pelvis = human_frames[0]["pelvis"][0]
    assert np.all(np.isfinite(pelvis))


def test_build_eval_clip_context_xperience(synthetic_hdf5: Path):
    ctx = build_eval_clip_context(
        {"dataset": "xperience_unified"},
        {"idx": 3, "dataset": "xperience"},
        hdf5_path=str(synthetic_hdf5),
        native_start=7,
        native_fps=20.0,
        control_fps=20.0,
    )
    assert not ctx.is_pick_tissue
    backend = build_gt_backend(ctx)
    d_raw, _ = backend.pack_deploy_frame(control_idx=0, state_control_idx=0)
    assert d_raw.shape == (512,)


def test_denorm_xperience_synthetic(synthetic_hdf5: Path, skeleton_constants):
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
    assert len(human_frames) == 4
    assert root_quats.shape == (4, 4)


def test_protocol_gripper_roundtrip():
    from phi0.deploy.zmq_protocol import decode_message, encode_frame
    from phi0.deploy.gmr_retarget import GMR_SMPLX_BODY_NAMES

    human_data = {
        name: (np.zeros(3, np.float32), np.array([1, 0, 0, 0], np.float32))
        for name in GMR_SMPLX_BODY_NAMES
    }
    g = np.linspace(-0.2, 0.2, 14, dtype=np.float32)
    payload = encode_frame(0, human_data, gripper_joints_14=g)
    msg = decode_message(payload)
    assert np.allclose(msg["gripper_joints_14"], g)


def test_encode_meta_pick_tissue_fields():
    from phi0.deploy.zmq_protocol import decode_message, encode_meta

    payload = encode_meta(
        clip_idx=2,
        num_frames=80,
        control_fps=50.0,
        episode_idx=17,
        proprio_w=1,
        dataset="pick_tissue_unified",
        pick_tissue_root="/data/root",
        pick_tissue_repo_id="pick_tissue_xperience_unified",
    )
    msg = decode_message(payload)
    assert msg["episode_idx"] == 17
    assert msg["proprio_w"] == 1
    assert msg["dataset"] == "pick_tissue_unified"


def test_gt_views_track_mapping_and_composite():
    from experiments.phi0_hgpt_zmq.gt_views import (
        build_track_panel_indices,
        composite_tracker_gt_views,
        track_step_to_pub_idx,
        upsample_len_20_to_50,
        upsample_len_ctrl_to_tracker,
    )
    from phi0.deploy.pick_tissue_gt_images import letterbox_rgb

    assert letterbox_rgb(np.zeros((480, 640, 3), np.uint8), (180, 320)).shape == (180, 320, 3)

    num_pub = 400
    len50 = upsample_len_20_to_50(num_pub)
    assert len50 == 999
    assert upsample_len_ctrl_to_tracker(num_pub, control_fps=50.0, tracker_freq=50) == 400
    assert track_step_to_pub_idx(0, num_pub=num_pub, len_qpos_50=len50, stand_n=100, blend_n=25) == 0
    assert track_step_to_pub_idx(1073, num_pub=num_pub, len_qpos_50=len50, stand_n=100, blend_n=25) == 399
    idx = build_track_panel_indices(
        1074, num_pub=num_pub, stand_seconds=2.0, blend_seconds=0.5, freq=50
    )
    assert idx.shape == (1074,)
    assert idx[0] == 0
    assert idx[-1] == 399
    idx50 = build_track_panel_indices(
        475, num_pub=num_pub, stand_seconds=2.0, blend_seconds=0.5, freq=50, control_fps=50.0
    )
    assert idx50[350] == 275
    out = composite_tracker_gt_views(
        np.zeros((480, 640, 3), np.uint8),
        np.zeros((180, 320, 3), np.uint8),
        np.zeros((180, 320, 3), np.uint8),
    )
    assert out.shape == (660, 640, 3)


@pytest.mark.skipif(not _pick_tissue_available(), reason="pick-tissue dataset missing")
def test_pick_tissue_predecoded_reader():
    from phi0.deploy.pick_tissue_gt_images import PickTissuePredecodedReader

    reader = PickTissuePredecodedReader(
        root_dir=str(PICK_TISSUE_ROOT),
        repo_id=PICK_TISSUE_REPO,
    )
    span = reader.episode_span(0)
    ego, wrist = reader.read_ego_wrist_pair(span.frame_start, span)
    assert ego.shape[-1] == 3 and ego.dtype == np.uint8
    assert wrist.shape == ego.shape


@pytest.mark.skipif(not _pick_tissue_available(), reason="pick-tissue dataset missing")
def test_pick_tissue_read_ego_wrist():
    reader = reader_from_data_cfg(
        {"pick_tissue_root": str(PICK_TISSUE_ROOT), "pick_tissue_repo_id": PICK_TISSUE_REPO}
    )
    span = reader.episode_span(0)
    ego, wrist = reader.read_ego_wrist_pair(span.frame_start, span)
    assert ego.shape[-1] == 3 and ego.dtype == np.uint8
    assert wrist.shape == ego.shape


def test_merge_body_gripper_qpos50():
    from phi0.deploy.dex3_gripper import merge_body_and_gripper_qpos, wbc_hand7_to_deploy

    body = np.arange(36, dtype=np.float32)
    grip = np.linspace(0, 1, 14, dtype=np.float32)
    q50 = merge_body_and_gripper_qpos(body, grip)
    assert q50.shape == (50,)
    assert np.allclose(q50[:36], body)
    assert np.allclose(q50[36:43], wbc_hand7_to_deploy(grip[:7]))
    assert np.allclose(q50[43:50], wbc_hand7_to_deploy(grip[7:]))
