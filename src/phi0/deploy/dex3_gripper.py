"""G1 Dex3 three-finger gripper (14-d) helpers for ZMQ eval."""

from __future__ import annotations

import numpy as np

NUM_GRIPPER_JOINTS = 14
NUM_GRIPPER_EACH = 7
# WBC / unified 346:360 (Isaac-GR00T g1_43dof): index×2, middle×2, thumb×3
# Deploy ZMQ + MuJoCo actuated order: thumb×3, index×2, middle×2
WBC_TO_DEPLOY_HAND7_IDX = (4, 5, 6, 0, 1, 2, 3)
# qpos layout for GMR ``g1_mocap_29dof_with_hands.xml``: 7 root + 29 body + 7 L + 7 R
QPOS_WITH_HANDS_DIM = 50
BODY_QPOS_DIM = 36
GRIPPER_LEFT_QPOS_SLICE = slice(36, 43)
GRIPPER_RIGHT_QPOS_SLICE = slice(43, 50)


def wbc_hand7_to_deploy(hand7: np.ndarray) -> np.ndarray:
    """Reorder one Dex3 hand vector from WBC/unified layout to deploy ZMQ layout."""
    h = np.asarray(hand7, dtype=np.float32).reshape(NUM_GRIPPER_EACH)
    return h[np.asarray(WBC_TO_DEPLOY_HAND7_IDX, dtype=np.int64)]


def split_gripper14_wbc_to_deploy(gripper14: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split unified/WBC 14-d gripper into deploy-order left/right 7-vectors."""
    g = np.asarray(gripper14, dtype=np.float32).reshape(NUM_GRIPPER_JOINTS)
    return wbc_hand7_to_deploy(g[:NUM_GRIPPER_EACH]), wbc_hand7_to_deploy(g[NUM_GRIPPER_EACH:])


def upsample_rows_20_to_50(rows: np.ndarray) -> np.ndarray:
    """Same 20→50 Hz resampling as body qpos (linear; root quat unused here)."""
    from scipy.interpolate import interp1d

    q = np.asarray(rows, dtype=np.float32)
    if q.ndim != 2:
        raise ValueError(f"expected (T, D), got {q.shape}")
    if q.shape[0] < 2:
        return q.copy()
    t20 = np.arange(q.shape[0], dtype=np.float64)
    t50 = np.linspace(0, q.shape[0] - 1, int(round((q.shape[0] - 1) * 2.5) + 1))
    return interp1d(t20, q, axis=0, kind="linear")(t50).astype(np.float32)


def merge_body_and_gripper_qpos(body_qpos: np.ndarray, gripper14: np.ndarray) -> np.ndarray:
    """Build 50-d with_hands qpos from 36-d body qpos + 14-d WBC/unified Dex3 angles."""
    body = np.asarray(body_qpos, dtype=np.float32).reshape(BODY_QPOS_DIM)
    left, right = split_gripper14_wbc_to_deploy(gripper14)
    out = np.zeros(QPOS_WITH_HANDS_DIM, dtype=np.float32)
    out[:BODY_QPOS_DIM] = body
    out[GRIPPER_LEFT_QPOS_SLICE] = left
    out[GRIPPER_RIGHT_QPOS_SLICE] = right
    return out
