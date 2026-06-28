"""Unified 512-d deploy outputs -> SONIC ZMQ v4 arrays (tokens + deploy-order hands)."""

from __future__ import annotations

import numpy as np

from phi0.deploy.dex3_gripper import WBC_TO_DEPLOY_HAND7_IDX, NUM_GRIPPER_EACH
from phi0.schema.unified_action_schema import SLICES

_TOKEN_SLICE = slice(*SLICES["sonic_motion_token_64"])
_GRIP_SLICE = slice(*SLICES["g1_gripper_joints_14"])
_WBC_TO_DEPLOY = np.asarray(WBC_TO_DEPLOY_HAND7_IDX, dtype=np.int64)


def unified_action_denorm_to_zmq_arrays(
    action_denorm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized split: sonic tokens + left/right Dex3 in deploy joint order."""
    a = np.asarray(action_denorm, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] < _GRIP_SLICE.stop:
        raise ValueError(f"expected (T, >={_GRIP_SLICE.stop}), got {a.shape}")
    grip = a[:, _GRIP_SLICE]
    tokens = np.ascontiguousarray(a[:, _TOKEN_SLICE])
    left = np.ascontiguousarray(grip[:, :NUM_GRIPPER_EACH][:, _WBC_TO_DEPLOY])
    right = np.ascontiguousarray(grip[:, NUM_GRIPPER_EACH:][:, _WBC_TO_DEPLOY])
    return tokens, left, right
