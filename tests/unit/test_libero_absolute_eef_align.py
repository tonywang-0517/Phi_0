"""Absolute EEF train/deploy alignment for LIBERO."""

from __future__ import annotations

import numpy as np
import torch

from phi0.benchmark.adapters import libero_obs_to_eef_7d
from phi0.benchmark.rlds_adapters import libero_rlds_state_to_eef_7d
from phi0.benchmark.rlds_io import RldsStep
from phi0.models.action_proprio import split_proprio_future


def _fake_libero_obs(state: np.ndarray) -> dict:
    import math

    def axisangle2quat(aa: np.ndarray) -> np.ndarray:
        angle = float(np.linalg.norm(aa))
        if angle < 1e-8:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        axis = aa / angle
        half = angle / 2.0
        return np.concatenate([axis * math.sin(half), [math.cos(half)]]).astype(np.float32)

    s = np.asarray(state, dtype=np.float32).reshape(-1)
    return {
        "robot0_eef_pos": s[:3],
        "robot0_eef_quat": axisangle2quat(s[3:6]),
        "robot0_gripper_qpos": s[6:8] if s.shape[0] >= 8 else s[6:7],
        "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
    }


def test_rlds_state_and_obs_eef_match():
    state = np.array(
        [-0.20, -0.01, 1.17, 3.14, 0.0, -0.08, 0.039, -0.039],
        dtype=np.float32,
    )
    step = RldsStep(
        rgb_static=np.zeros((8, 8, 3), dtype=np.uint8),
        rgb_gripper=np.zeros((8, 8, 3), dtype=np.uint8),
        state=state,
        action=np.zeros(7, dtype=np.float32),
        language="task",
    )
    from_rlds = libero_rlds_state_to_eef_7d(step.state)
    from_obs = libero_obs_to_eef_7d(_fake_libero_obs(step.state))
    assert np.allclose(from_rlds[:6], from_obs[:6], atol=1e-5)
    assert from_rlds[6] == from_obs[6] == 1.0


def test_proprio_future_split_is_current_and_future_eef():
    eef = torch.tensor(
        [
            [0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.3, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.4, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.6, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    encoded = eef.unsqueeze(0)
    proprio, future = split_proprio_future(encoded, 5)
    assert proprio.shape == (1, 5, 7)
    assert future.shape == (1, 1, 7)
    assert torch.allclose(proprio[0, -1, :3], eef[4, :3])
    assert torch.allclose(future[0, 0, :3], eef[5, :3])


def test_action_stats_use_robot_action_7d():
    from phi0.data.action_stats import _frame_action_vector

    item = {
        "robot_action_7d": torch.tensor([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]),
        "action": torch.zeros(256),
        "action_dim_is_pad": torch.zeros(256, dtype=torch.bool),
    }
    x, valid = _frame_action_vector(item)
    assert valid[:7].all()
    assert not valid[7:].any()
    assert np.isclose(x[0], 0.1)
    assert np.isclose(x[6], 1.0)
