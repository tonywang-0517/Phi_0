"""EgoDex sparse quat cache -> keypoints (52×3) for Phi_0 training."""

from __future__ import annotations

import numpy as np

from phi0.schema.action_schema import D_RAW, KEYPOINTS_FLAT_DIM, NUM_SKELETON_JOINTS
from phi0.schema.action_schema import LEGACY_QUAT_SLICES as QUAT_SLICES
from phi0.viz.smplh_fk import joints_from_d_raw


def quat_dim_available_to_keypoint_dim_available(dim_available: np.ndarray) -> np.ndarray:
    """Map legacy quat-layout ``dim_available`` (256,) -> keypoints flat mask (156,).

    A joint's xyz is supervised when its corresponding quat block (or root trans+quat)
    was marked available in the sparse EgoDex SMPL+H cache.
    """
    dim_available = np.asarray(dim_available, dtype=bool).reshape(-1)
    if dim_available.shape[0] < D_RAW:
        padded = np.zeros(D_RAW, dtype=bool)
        padded[: dim_available.shape[0]] = dim_available
        dim_available = padded

    out = np.zeros(KEYPOINTS_FLAT_DIM, dtype=bool)

    if dim_available[QUAT_SLICES["root_trans"][0] : QUAT_SLICES["root_quat"][1]].any():
        out[0:3] = True

    body_s, body_e = QUAT_SLICES["body_quats"]
    for body_i in range(21):
        sl = slice(body_s + body_i * 4, body_s + (body_i + 1) * 4)
        if dim_available[sl].any():
            j = body_i + 1
            out[j * 3 : (j + 1) * 3] = True

    lh_s, _ = QUAT_SLICES["left_hand_quats"]
    for i in range(15):
        sl = slice(lh_s + i * 4, lh_s + (i + 1) * 4)
        if dim_available[sl].any():
            j = 22 + i
            out[j * 3 : (j + 1) * 3] = True

    rh_s, _ = QUAT_SLICES["right_hand_quats"]
    for i in range(15):
        sl = slice(rh_s + i * 4, rh_s + (i + 1) * 4)
        if dim_available[sl].any():
            j = 37 + i
            out[j * 3 : (j + 1) * 3] = True

    return out


def pack_egodex_keypoints_d_raw(
    d_raw_quat: np.ndarray,
    dim_available_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert sparse EgoDex quat ``d_raw`` row to keypoints pose + per-dim GT mask.

    Returns:
        d_raw: (256,) with ``d_raw[:156]`` = FK joint positions (camera frame).
        dim_available: (256,) bool, True on supervised keypoint xyz (+ legacy extras if set).
    """
    d_raw_quat = np.asarray(d_raw_quat, dtype=np.float32).reshape(-1)
    if d_raw_quat.shape[0] < D_RAW:
        padded = np.zeros(D_RAW, dtype=np.float32)
        padded[: d_raw_quat.shape[0]] = d_raw_quat
        d_raw_quat = padded

    dim_quat = np.asarray(dim_available_quat, dtype=bool).reshape(-1)
    if dim_quat.shape[0] < D_RAW:
        padded = np.zeros(D_RAW, dtype=bool)
        padded[: dim_quat.shape[0]] = dim_quat
        dim_quat = padded

    keypoints = joints_from_d_raw(d_raw_quat).reshape(NUM_SKELETON_JOINTS, 3)
    kp_dim = quat_dim_available_to_keypoint_dim_available(dim_quat)

    out = np.zeros(D_RAW, dtype=np.float32)
    out[:KEYPOINTS_FLAT_DIM] = keypoints.reshape(-1)

    dim_available = np.zeros(D_RAW, dtype=bool)
    dim_available[:KEYPOINTS_FLAT_DIM] = kp_dim
    # Preserve tactile / reserved flags from quat cache when present.
    for field in ("tactile_storage", "betas_storage"):
        pass  # egodex cache does not supervise these in quat layout
    if dim_quat[227:237].any():
        out[227:237] = d_raw_quat[227:237]
        dim_available[227:237] = dim_quat[227:237]

    return out, dim_available
