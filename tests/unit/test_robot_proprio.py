"""Deploy g1_debug -> unified proprio packing."""

from __future__ import annotations

import numpy as np

from phi0.deploy.dex3_gripper import wbc_hand7_to_deploy
from phi0.deploy.robot_proprio import (
    deploy_hand7_to_wbc,
    g1_body_qpos36_from_g1_debug,
    gripper14_wbc_from_g1_debug,
    unified_from_g1_debug,
)
from phi0.schema.unified_action_schema import SLICES, SONIC_MOTION_TOKEN_DIM


def _sample_debug() -> dict:
    tok = np.linspace(-0.5, 0.5, SONIC_MOTION_TOKEN_DIM, dtype=np.float32)
    return {
        "base_trans_measured": [0.1, 0.2, 0.79],
        "base_quat_measured": [1.0, 0.0, 0.0, 0.0],
        "body_q_measured": np.arange(29, dtype=np.float32) * 0.01,
        "left_hand_q_measured": np.arange(7, dtype=np.float32) * 0.1,
        "right_hand_q_measured": np.arange(7, dtype=np.float32) * -0.1,
        "token_state": tok,
    }


def test_deploy_wbc_hand_roundtrip():
    wbc = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    back = deploy_hand7_to_wbc(wbc_hand7_to_deploy(wbc))
    assert np.allclose(back, wbc)


def test_unified_from_g1_debug_fills_robot_slices():
    d_raw, root = unified_from_g1_debug(_sample_debug())
    assert root.shape == (3,)
    s, e = SLICES["g1_gripper_joints_14"]
    assert np.any(d_raw[s:e] != 0)
    s, e = SLICES["g1_body_qpos_36"]
    qpos = g1_body_qpos36_from_g1_debug(_sample_debug())
    assert np.allclose(d_raw[s:e], qpos)
    s, e = SLICES["sonic_motion_token_64"]
    assert np.allclose(d_raw[s:e], _sample_debug()["token_state"])


def test_gripper14_matches_deploy_to_wbc():
    msg = _sample_debug()
    g14 = gripper14_wbc_from_g1_debug(msg)
    assert g14.shape == (14,)
