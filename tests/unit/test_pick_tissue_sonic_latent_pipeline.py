"""End-to-end unit tests for pick-tissue SONIC latent deploy pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from phi0.data.pick_tissue_unified import _dim_pad_mask
from phi0.deploy.dex3_gripper import split_gripper14_wbc_to_deploy, wbc_hand7_to_deploy
from phi0.deploy.gt_io import deploy_gt_norm_lut_indices
from phi0.deploy.sonic_latent_gt_replay import (
    build_replay_messages,
    load_sonic_latent_replay_arrays,
)
from phi0.deploy.sonic_zmq_io import unified_action_denorm_to_zmq_arrays
from phi0.schema.unified_action_schema import D_UNIFIED, SLICES, dim_mask_for_dataset
from gear_sonic.utils.zmq_pose_unpack import unpack_pose_message


def _slow_denorm_to_zmq(action_denorm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference loop (pre-vectorization semantics)."""
    from phi0.schema.unified_action_schema import (
        unpack_g1_gripper_joints_14,
        unpack_sonic_motion_token_64,
    )

    n = int(action_denorm.shape[0])
    tokens = np.stack([unpack_sonic_motion_token_64(action_denorm[i]) for i in range(n)])
    left = np.zeros((n, 7), dtype=np.float32)
    right = np.zeros((n, 7), dtype=np.float32)
    for i in range(n):
        left[i], right[i] = split_gripper14_wbc_to_deploy(
            unpack_g1_gripper_joints_14(action_denorm[i])
        )
    return tokens, left, right


def test_pick_tissue_g1_sonic_supervision_mask():
    mask = dim_mask_for_dataset("g1_sonic")
    assert not mask[0:3].any()
    assert mask[3:336].all()
    assert mask[346:360].all()
    assert mask[360:396].all()
    assert mask[396:460].all()
    assert not mask[460:].any()

    pad = _dim_pad_mask()
    assert not pad[346:460].any()


def test_unified_action_denorm_to_zmq_matches_loop():
    rng = np.random.default_rng(0)
    action = rng.standard_normal((17, D_UNIFIED), dtype=np.float32)
    fast = unified_action_denorm_to_zmq_arrays(action)
    slow = _slow_denorm_to_zmq(action)
    for a, b in zip(fast, slow):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)


def test_wbc_to_deploy_known_left_hand():
    # ep447-style WBC left: index/middle close, thumb open
    wbc_left = np.array(
        [-0.16726853, -0.2509028, -0.16726853, -0.2509028, 0.0, 0.11708797, 0.11708797],
        dtype=np.float32,
    )
    deploy = wbc_hand7_to_deploy(wbc_left)
    np.testing.assert_allclose(
        deploy,
        [0.0, 0.11708797, 0.11708797, -0.16726853, -0.2509028, -0.16726853, -0.2509028],
        rtol=0,
        atol=1e-6,
    )


def test_gt_replay_unified_hands_deploy_order(tmp_path: Path):
    n = 5
    ua = []
    for _ in range(n):
        row = np.zeros(D_UNIFIED, dtype=np.float32)
        s, e = SLICES["g1_gripper_joints_14"]
        row[s:e] = np.array(
            [0.1, 0.2, 0.1, 0.2, 0.0, -0.1, -0.1, 0.3, 0.4, 0.3, 0.4, 0.0, -0.2, -0.2],
            dtype=np.float32,
        )
        s, e = SLICES["sonic_motion_token_64"]
        row[s:e] = 0.05
        ua.append(row)
    pq.write_table(pa.table({"unified_action": ua}), tmp_path / "u.parquet")

    _, left, right, source = load_sonic_latent_replay_arrays(tmp_path / "u.parquet")
    assert source == "unified_slice+unified_gripper"
    np.testing.assert_allclose(left[0], [0.0, -0.1, -0.1, 0.1, 0.2, 0.1, 0.2], atol=1e-6)
    np.testing.assert_allclose(right[0], [0.0, -0.2, -0.2, 0.3, 0.4, 0.3, 0.4], atol=1e-6)


def test_prebuild_messages_hand_ramp_and_token():
    tokens = np.array([[0.1] * 64, [0.2] * 64], dtype=np.float32)
    left = np.ones((2, 7), dtype=np.float32) * 0.5
    right = np.ones((2, 7), dtype=np.float32) * 0.25
    msgs = build_replay_messages(tokens, left, right, hand_ramp_frames=2)
    m0 = unpack_pose_message(msgs[0])
    m1 = unpack_pose_message(msgs[1])
    np.testing.assert_allclose(m0["token_state"].reshape(-1), tokens[0], atol=1e-6)
    np.testing.assert_allclose(m0["left_hand_joints"].reshape(-1), left[0] * 0.0, atol=1e-6)
    np.testing.assert_allclose(m1["left_hand_joints"].reshape(-1), left[1] * 0.5, atol=1e-5)


def test_lazy_lut_pin_device():
    import torch
    from phi0.deploy.gt_io import LazyDeployGtNormLut

    class _Backend:
        def pack_deploy_frame(self, *, control_idx, state_control_idx):
            d = np.zeros(512, dtype=np.float32)
            d[396] = float(control_idx)
            return d, np.zeros(3, dtype=np.float32)

    class _Proc:
        def _normalize_action(self, t):
            return t

    lut = LazyDeployGtNormLut(_Backend(), _Proc(), [0, 1])
    _ = lut[0]
    lut.pin_device("cpu")
    assert lut[0].device.type == "cpu"
    assert lut[0][396].item() == 0.0


def test_deploy_lut_indices_smaller_than_full_timeline():
    idx = deploy_gt_norm_lut_indices(num_frames=831, proprio_w=4, chunk_h=29, history_w=4)
    assert len(idx) < 4 + 831
    assert 4 in idx


@pytest.mark.skipif(
    not Path(
        "/mnt/data2/wpy/workspace/Isaac-GR00T/data/pick_tissue_xperience_unified/data/chunk-000/episode_000447.parquet"
    ).is_file(),
    reason="pick-tissue ep447 parquet not on disk",
)
def test_ep447_unified_gripper_matches_wbc():
    import pyarrow.parquet as pq

    root = Path("/mnt/data2/wpy/workspace/Isaac-GR00T/data")
    ua = np.stack(
        pq.read_table(
            root / "pick_tissue_xperience_unified/data/chunk-000/episode_000447.parquet",
            columns=["unified_action"],
        )
        .column("unified_action")
        .to_numpy()
    )
    wbc = np.stack(
        pq.read_table(
            root / "pick_tissue_valid/data/chunk-000/episode_000524.parquet",
            columns=["action.wbc"],
        )
        .column("action.wbc")
        .to_numpy()
    )
    grip = ua[:, SLICES["g1_gripper_joints_14"][0] : SLICES["g1_gripper_joints_14"][1]]
    wbc_grip = np.concatenate([wbc[:, 22:29], wbc[:, 36:43]], axis=1)
    np.testing.assert_allclose(grip, wbc_grip, atol=1e-5)
    _, left, _ = unified_action_denorm_to_zmq_arrays(ua)
    np.testing.assert_allclose(left[600], wbc_hand7_to_deploy(wbc[600, 22:29]), atol=1e-5)
