"""Unit tests for pick-tissue SONIC latent GT replay loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from phi0.deploy.sonic_latent_gt_replay import (
    build_replay_messages,
    load_sonic_latent_replay_arrays,
    resolve_token_source,
)
from phi0.schema.unified_action_schema import D_UNIFIED, SLICES
from gear_sonic.utils.zmq_pose_unpack import unpack_pose_message


def _write_unified_parquet(path: Path, n: int = 6) -> None:
    ua = []
    for i in range(n):
        row = np.zeros(D_UNIFIED, dtype=np.float32)
        s, e = SLICES["sonic_motion_token_64"]
        row[s:e] = np.full(64, float(i) * 0.01, dtype=np.float32)
        ua.append(row)
    pq.write_table(
        pa.table({"unified_action": ua}),
        path,
    )


def _write_valid_parquet(path: Path, n: int = 6) -> None:
    pq.write_table(
        pa.table(
            {
                "action.motion_token": [np.full(64, float(i), dtype=np.float32) for i in range(n)],
                "teleop.left_hand_joints": [np.full(7, 0.1 * i, dtype=np.float32) for i in range(n)],
                "teleop.right_hand_joints": [np.full(7, 0.2 * i, dtype=np.float32) for i in range(n)],
            }
        ),
        path,
    )


def test_resolve_token_source_auto():
    assert resolve_token_source({"unified_action"}, "auto") == "unified_slice"
    assert resolve_token_source({"action.motion_token"}, "auto") == "valid_column"
    with pytest.raises(ValueError):
        resolve_token_source({"timestamp"}, "auto")


def test_load_unified_with_external_hands(tmp_path: Path):
    unified = tmp_path / "unified.parquet"
    valid = tmp_path / "valid.parquet"
    _write_unified_parquet(unified, n=8)
    _write_valid_parquet(valid, n=8)

    tokens, left, right, source = load_sonic_latent_replay_arrays(
        unified,
        valid_parquet_for_hands=valid,
        max_frames=5,
    )
    assert source == "unified_slice+unified_gripper"
    assert tokens.shape == (5, 64)
    assert left.shape == (5, 7)
    assert right.shape == (5, 7)
    np.testing.assert_allclose(tokens[3], 0.03, atol=1e-6)
    # pick-tissue: teleop hands in valid are ignored; unified gripper slice is all-zero in fixture
    np.testing.assert_allclose(left[2], 0.0, atol=1e-6)


def test_unified_gripper_hands_used_when_teleop_degenerate(tmp_path: Path):
    unified = tmp_path / "unified.parquet"
    n = 4
    ua = []
    for i in range(n):
        row = np.zeros(D_UNIFIED, dtype=np.float32)
        s, e = SLICES["g1_gripper_joints_14"]
        row[s:e] = np.array([0.1, 0.2, 0.1, 0.2, 0.0, -0.1, -0.1, 0.3, 0.4, 0.3, 0.4, 0.0, -0.2, -0.2], np.float32)
        ua.append(row)
    pq.write_table(pa.table({"unified_action": ua}), unified)
    valid = tmp_path / "valid.parquet"
    _write_valid_parquet(valid, n=n)  # teleop hands are non-zero but ignored for unified_slice

    _, left, right, source = load_sonic_latent_replay_arrays(unified)
    assert source == "unified_slice+unified_gripper"
    # WBC left [index×2, middle×2, thumb×3] -> deploy [thumb×3, index×2, middle×2]
    np.testing.assert_allclose(left[0], [0.0, -0.1, -0.1, 0.1, 0.2, 0.1, 0.2], atol=1e-6)
    np.testing.assert_allclose(right[0], [0.0, -0.2, -0.2, 0.3, 0.4, 0.3, 0.4], atol=1e-6)


def test_load_valid_column_single_parquet(tmp_path: Path):
    valid = tmp_path / "valid.parquet"
    _write_valid_parquet(valid, n=4)
    tokens, left, right, source = load_sonic_latent_replay_arrays(valid)
    assert source == "valid_column"
    assert tokens.shape == (4, 64)
    np.testing.assert_allclose(tokens[1], 1.0)


def test_build_replay_messages_roundtrip(tmp_path: Path):
    valid = tmp_path / "valid.parquet"
    _write_valid_parquet(valid, n=3)
    tokens, left, right, _ = load_sonic_latent_replay_arrays(valid)
    msgs = build_replay_messages(tokens, left, right, hand_ramp_frames=2)
    assert len(msgs) == 3
    msg = unpack_pose_message(msgs[1])
    np.testing.assert_allclose(msg["token_state"].reshape(-1), tokens[1], atol=1e-6)
    np.testing.assert_allclose(msg["right_hand_joints"].reshape(-1), 0.1, atol=1e-5)


def test_hand_length_mismatch_raises(tmp_path: Path):
    valid = tmp_path / "valid.parquet"
    short = tmp_path / "short.parquet"
    _write_valid_parquet(valid, n=5)
    _write_valid_parquet(short, n=3)
    with pytest.raises(ValueError, match="hands parquet T="):
        load_sonic_latent_replay_arrays(valid, valid_parquet_for_hands=short)
