"""Unit tests for SONIC unified I/O (43-d state, 100-d action, ego + left wrist)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from phi0.data.processor import Phi0Processor
from phi0.data.sonic_action_norm import (
    SONIC_ACTION_DIM,
    SONIC_STATE_DIM,
    STATS_SEMANTICS_SONIC_UNIFIED,
    load_sonic_stats_json,
)
from phi0.data.sonic_lerobot import LEFT_WRIST_IMAGE_KEY, SonicUnifiedClipDataset
from phi0.data.sonic_unified_io import (
    DEFAULT_TORSO_HEIGHT,
    MIN_TORSO_HEIGHT,
    PLANNER_DISABLED,
    SONIC_ACTION_BASE_DIM,
    SONIC_MOTION_TOKEN_DIM,
    build_action_from_sonic_row,
    build_pi05_base_action_from_row,
    build_state_from_sonic_row,
    build_state_from_state43,
    hands_from_wbc,
    torso_height_from_knees,
    torso_rpy_from_wbc,
)

REAL_PARQUET = Path(
    "/mnt/data2/wpy/workspace/Isaac-GR00T/data/pick_tissue_valid/data/chunk-000/episode_000000.parquet"
)


def _make_wbc43() -> np.ndarray:
    wbc = np.zeros(43, dtype=np.float64)
    wbc[22:29] = np.arange(22, 29, dtype=np.float64) * 0.1
    wbc[36:43] = np.arange(36, 43, dtype=np.float64) * 0.1
    wbc[15:22] = np.arange(15, 22, dtype=np.float64) * 0.2
    wbc[29:36] = np.arange(29, 36, dtype=np.float64) * 0.2
    wbc[12:15] = [0.11, 0.22, 0.33]
    wbc[3] = 0.25
    wbc[9] = 0.25
    return wbc


def _dummy_row(*, planner_height: float = 0.8, state_hand_zero: bool = False) -> pd.Series:
    state = _make_wbc43().copy()
    wbc = _make_wbc43()
    if state_hand_zero:
        state[22:29] = 0.0
        state[36:43] = 0.0
    token = np.linspace(-1.0, 1.0, SONIC_MOTION_TOKEN_DIM, dtype=np.float64)
    return pd.Series(
        {
            "observation.state": state.tolist(),
            "action.wbc": wbc.tolist(),
            "action.motion_token": token.tolist(),
            "teleop.planner_height": planner_height,
            "teleop.planner_movement": [0.5, 0.0, 0.1],
            "teleop.planner_speed": 0.4,
            "teleop.delta_heading": 0.2,
            "teleop.left_hand_joints": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "teleop.right_hand_joints": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
        }
    )


def test_state_and_action_dims():
    row = _dummy_row()
    states = build_state_from_sonic_row(row)
    action = build_action_from_sonic_row(row)
    assert states.shape == (SONIC_STATE_DIM,)
    assert action.shape == (SONIC_ACTION_DIM,)


def test_state43_layout():
    state43 = _make_wbc43()
    states = build_state_from_state43(state43)
    assert states.shape == (SONIC_STATE_DIM,)
    assert np.allclose(states[:29], np.concatenate([state43[:15], state43[15:22], state43[29:36]]))
    assert np.allclose(states[29:], np.concatenate([state43[22:29], state43[36:43]]))


def test_action_splits_base_and_motion_token():
    row = _dummy_row()
    action = build_action_from_sonic_row(row)
    base = build_pi05_base_action_from_row(row)
    token = np.asarray(row["action.motion_token"], dtype=np.float32)
    assert action.shape == (SONIC_ACTION_DIM,)
    assert np.allclose(action[:SONIC_ACTION_BASE_DIM], base)
    assert np.allclose(action[SONIC_ACTION_BASE_DIM:], token)


def test_action_locomotion_slice():
    row = _dummy_row()
    action = build_action_from_sonic_row(row)
    assert np.isclose(action[31], 0.8)
    assert np.isclose(action[32], 0.2)
    assert np.isclose(action[33], 0.0)
    assert np.isclose(action[34], 0.04)
    assert np.isclose(action[35], 0.2)


def test_hand_arm_slices():
    row = _dummy_row()
    action = build_action_from_sonic_row(row)
    wbc = np.asarray(row["action.wbc"], dtype=np.float64)
    assert np.allclose(action[:14], hands_from_wbc(wbc))
    assert np.allclose(action[14:28], np.concatenate([wbc[15:22], wbc[29:36]]))


def test_torso_rpy_reorder():
    wbc = _make_wbc43()
    rpy = torso_rpy_from_wbc(wbc)
    assert np.allclose(rpy, [0.22, 0.33, 0.11])


def test_height_from_knees():
    stand = np.zeros(43)
    stand[3] = 0.25
    stand[9] = 0.25
    squat = np.zeros(43)
    squat[3] = 0.90
    squat[9] = 0.90
    assert np.isclose(torso_height_from_knees(stand), DEFAULT_TORSO_HEIGHT)
    assert torso_height_from_knees(squat) < DEFAULT_TORSO_HEIGHT
    assert torso_height_from_knees(squat) >= MIN_TORSO_HEIGHT


def test_height_fallback_when_planner_disabled():
    row = _dummy_row(planner_height=PLANNER_DISABLED)
    action = build_action_from_sonic_row(row)
    assert action[31] >= MIN_TORSO_HEIGHT


@pytest.mark.skipif(not REAL_PARQUET.is_file(), reason="real parquet not available")
def test_real_parquet_row():
    df = pd.read_parquet(REAL_PARQUET, columns=None)
    row = df.iloc[0]
    states = build_state_from_sonic_row(row)
    action = build_action_from_sonic_row(row)
    assert states.shape == (SONIC_STATE_DIM,)
    assert action.shape == (SONIC_ACTION_DIM,)
    assert np.isfinite(states).all()
    assert np.isfinite(action).all()


def test_load_sonic_stats_json(tmp_path):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        '{"action": {"mean": [0.0], "std": [1.0], "q01": [0.0], "q99": [1.0]}, '
        '"states": {"mean": [0.0], "std": [1.0]}}',
        encoding="utf-8",
    )
    loaded = load_sonic_stats_json(stats_path)
    assert loaded["robot_action_semantics"] == STATS_SEMANTICS_SONIC_UNIFIED
    assert loaded["action_dim"] == SONIC_ACTION_DIM
    assert loaded["state_dim"] == SONIC_STATE_DIM


def test_processor_stacks_ego_and_left_wrist():
    processor = Phi0Processor(action_dim=SONIC_ACTION_DIM, normalize=False, use_wrist_view=True)
    batch = {
        "task": ["pick tissue"],
        "idx": torch.tensor([0]),
        "image_is_pad": torch.zeros(1, 1, dtype=torch.bool),
        "action_is_pad": torch.zeros(1, 2, dtype=torch.bool),
        "action_dim_is_pad": torch.zeros(1, 128, dtype=torch.bool),
        "action": torch.zeros(1, 2, 128),
        "images": {
            "ego_view": torch.ones(1, 1, 3, 8, 8),
            "wrist_view": torch.ones(1, 1, 3, 8, 8) * 2.0,
        },
    }
    out = processor.preprocess(batch)
    assert out["pixel_values"].shape == (1, 2, 1, 3, 8, 8)
    assert out["pixel_values"][0, 0, 0, 0, 0, 0] == 1.0
    assert out["pixel_values"][0, 1, 0, 0, 0, 0] == 2.0


@pytest.mark.skipif(
    not Path("/mnt/data2/wpy/workspace/Isaac-GR00T/data/pick_tissue_sonic_unified").is_dir(),
    reason="converted dataset not available",
)
def test_sonic_unified_dataset_collate():
    ds = SonicUnifiedClipDataset(
        root_dir="/mnt/data2/wpy/workspace/Isaac-GR00T/data",
        repo_id="pick_tissue_sonic_unified",
        future_action_steps=4,
        use_left_wrist=True,
        val=False,
        val_ratio=0.01,
    )
    item = ds[0]
    assert item["robot_proprio_43d"].shape == (1, SONIC_STATE_DIM)
    assert item["robot_future_100d"].shape[0] == 4
    assert item["robot_future_100d"].shape[1] == SONIC_ACTION_DIM
    if ds.use_left_wrist:
        assert "wrist_view" in item["images"]
        assert item["images"]["ego_view"].shape == item["images"]["wrist_view"].shape
    batch = SonicUnifiedClipDataset.collate_fn([item, item])
    assert batch["robot_proprio_43d"].shape == (2, 1, SONIC_STATE_DIM)
    assert batch["robot_future_100d"].shape == (2, 4, SONIC_ACTION_DIM)
    if ds.use_left_wrist:
        assert batch["images"]["wrist_view"].shape[0] == 2
    assert batch["action_dim_is_pad"].shape == (2, item["action_dim_is_pad"].numel())
    assert batch["action_dim_is_pad"][:, :SONIC_ACTION_DIM].all()


@pytest.mark.skipif(
    not Path("/mnt/data2/wpy/workspace/Isaac-GR00T/data/pick_tissue_sonic_unified").is_dir(),
    reason="converted dataset not available",
)
def test_sonic_unified_dataset_mono_skips_wrist():
    ds = SonicUnifiedClipDataset(
        root_dir="/mnt/data2/wpy/workspace/Isaac-GR00T/data",
        repo_id="pick_tissue_sonic_unified",
        future_action_steps=2,
        use_left_wrist=False,
        val=False,
        val_ratio=0.01,
    )
    item = ds[0]
    assert "wrist_view" not in item["images"]
