"""Action representation schema for Phi_0 D_raw buffer (256-d).

Training / deploy supervise **keypoints only** (52 joints × xyz = 156-d, indices 0:156).
Indices 156:211 are a fixed legacy buffer gap (always masked). 211:256 are storage slots
(betas / tactile / reserved), never predicted. Legacy quat layout 0:211 lives in
``LEGACY_QUAT_SLICES`` for FK audit scripts only.

For the full SMPL-H + SONIC + contact + tactile layout see
``phi0.schema.unified_action_schema`` (``D_UNIFIED=512``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

D_RAW = 256
NUM_SKELETON_JOINTS = 52
KEYPOINTS_FLAT_DIM = NUM_SKELETON_JOINTS * 3  # 156
SUPERVISED_POSE_DIM = KEYPOINTS_FLAT_DIM
LEGACY_BUFFER_GAP = (KEYPOINTS_FLAT_DIM, 211)  # 156:211, always masked in train/deploy

SLICES: Dict[str, Tuple[int, int]] = {
    "keypoints_52": (0, KEYPOINTS_FLAT_DIM),
    "legacy_buffer_gap": LEGACY_BUFFER_GAP,
    "betas_storage": (211, 227),
    "tactile_storage": (227, 237),
    "reserved": (237, 256),
}

# Fixed buffer offsets for legacy quat preprocessing / FK audit (not used in train/deploy).
LEGACY_QUAT_SLICES: Dict[str, Tuple[int, int]] = {
    "root_trans": (0, 3),
    "root_quat": (3, 7),
    "body_quats": (7, 91),
    "left_hand_quats": (91, 151),
    "right_hand_quats": (151, 211),
    "betas": (211, 227),
    "tactile": (227, 237),
    "reserved": (237, 256),
}

_DATASET_KEYPOINTS = {
    "keypoints_52": True,
    "legacy_buffer_gap": False,
    "betas_storage": False,
    "tactile_storage": False,
    "reserved": False,
}


@dataclass(frozen=True)
class ActionSchema:
    rep: str
    pose_dim_end: int
    slices: Dict[str, Tuple[int, int]]
    dataset_dim_available: Dict[str, Dict[str, bool]]


SCHEMA_KEYPOINTS = ActionSchema(
    rep="keypoints",
    pose_dim_end=KEYPOINTS_FLAT_DIM,
    slices=dict(SLICES),
    dataset_dim_available={
        "xperience": dict(_DATASET_KEYPOINTS),
        # EgoDex: sparse upper-body + hands via ARKit transforms (per-frame mask in loader).
        "egodex": dict(_DATASET_KEYPOINTS),
    },
)


def get_action_schema() -> ActionSchema:
    return SCHEMA_KEYPOINTS


def pack_xperience_keypoints(
    keypoints: np.ndarray,
    betas: np.ndarray | None = None,
    tactile: np.ndarray | None = None,
) -> np.ndarray:
    """Pack Xperience ``full_body_mocap/keypoints`` (52, 3) into D_raw."""
    out = np.zeros(D_RAW, dtype=np.float32)
    kp = np.asarray(keypoints, dtype=np.float32).reshape(NUM_SKELETON_JOINTS, 3)
    out[:KEYPOINTS_FLAT_DIM] = kp.reshape(-1)
    if betas is not None:
        out[211:227] = np.asarray(betas, dtype=np.float32).reshape(-1)[:16]
    if tactile is not None:
        out[227:237] = np.asarray(tactile, dtype=np.float32).reshape(-1)[:10]
    return out


def unpack_keypoints_52(d_raw: np.ndarray) -> np.ndarray:
    """(…, D_RAW) -> (…, 52, 3) keypoints from pose slice."""
    d_raw = np.asarray(d_raw, dtype=np.float32)
    flat = d_raw[..., :KEYPOINTS_FLAT_DIM]
    return flat.reshape(*flat.shape[:-1], NUM_SKELETON_JOINTS, 3)
