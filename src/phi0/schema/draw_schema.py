"""Unified D_raw buffer layout for Phi_0 (52×3 keypoints pose slice)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from phi0.schema.action_schema import (
    D_RAW,
    KEYPOINTS_FLAT_DIM,
    NUM_SKELETON_JOINTS,
    SLICES,
    get_action_schema,
    pack_xperience_keypoints,
    unpack_keypoints_52,
)

POSE_DIM_END = get_action_schema().pose_dim_end
DATASET_DIM_AVAILABLE: Dict[str, Dict[str, bool]] = {
    k: dict(v) for k, v in get_action_schema().dataset_dim_available.items()
}

FINGER_TIP_KEYS = {
    "left": [
        "leftThumbTip",
        "leftIndexFingerTip",
        "leftMiddleFingerTip",
        "leftRingFingerTip",
        "leftLittleFingerTip",
    ],
    "right": [
        "rightThumbTip",
        "rightIndexFingerTip",
        "rightMiddleFingerTip",
        "rightRingFingerTip",
        "rightLittleFingerTip",
    ],
}

MANO_TIP_INDICES = [4, 8, 12, 16, 20]


@dataclass
class DrawLayout:
    dim: int = D_RAW
    slices: Dict[str, Tuple[int, int]] | None = None

    def __post_init__(self):
        schema = get_action_schema()
        if self.slices is None:
            self.slices = dict(schema.slices)

    def dim_mask_for_dataset(self, dataset_name: str) -> np.ndarray:
        schema = get_action_schema()
        avail = schema.dataset_dim_available[dataset_name]
        mask = np.zeros(self.dim, dtype=bool)
        for field, (s, e) in schema.slices.items():
            if avail.get(field, False):
                mask[s:e] = True
        return mask

    def field_names(self) -> List[str]:
        return list(get_action_schema().slices.keys())


def compute_tactile_proxy_from_mano_joints(joints_3d: np.ndarray) -> np.ndarray:
    palm = joints_3d[0]
    pressures = []
    for idx in MANO_TIP_INDICES:
        dist = float(np.linalg.norm(joints_3d[idx] - palm))
        pressures.append(float(np.clip(1.0 - dist / 0.12, 0.0, 1.0)))
    return np.asarray(pressures, dtype=np.float32)


def compute_tactile_proxy_from_tip_positions(
    tip_positions: Dict[str, np.ndarray],
    hand: str,
) -> np.ndarray:
    keys = FINGER_TIP_KEYS[hand]
    wrist_key = "leftHand" if hand == "left" else "rightHand"
    palm = tip_positions.get(wrist_key, tip_positions[keys[0]])
    out = []
    for key in keys:
        dist = float(np.linalg.norm(tip_positions[key] - palm))
        out.append(float(np.clip(1.0 - dist / 0.12, 0.0, 1.0)))
    return np.asarray(out, dtype=np.float32)


def zero_unsupervised_action_dims(action, *, unified_dataset: str = "g1_sonic"):
    import torch

    from phi0.schema.unified_action_schema import D_UNIFIED, zero_unsupervised_unified_action_dims

    if not isinstance(action, torch.Tensor):
        raise TypeError("zero_unsupervised_action_dims expects a torch.Tensor")
    if int(action.shape[-1]) == D_UNIFIED:
        return zero_unsupervised_unified_action_dims(action, dataset_name=unified_dataset)
    out = action.clone()
    out[..., get_action_schema().pose_dim_end :] = 0.0
    return out


def zero_unsupervised_action_dims_np(d_raw: np.ndarray, *, unified_dataset: str = "g1_sonic") -> np.ndarray:
    from phi0.schema.unified_action_schema import D_UNIFIED, zero_unsupervised_unified_action_dims_np

    out = np.asarray(d_raw, dtype=np.float32).copy()
    if int(out.shape[-1]) == D_UNIFIED:
        return zero_unsupervised_unified_action_dims_np(out, dataset_name=unified_dataset)
    out[..., get_action_schema().pose_dim_end :] = 0.0
    return out


def neutral_betas(num: int = 1) -> np.ndarray:
    return np.zeros((num, 16), dtype=np.float32)


def unpack_action_for_viz(d_raw: np.ndarray) -> Dict[str, np.ndarray]:
    """Dataset-agnostic unpack for deploy / viz."""
    return {"keypoints_52": unpack_keypoints_52(d_raw)}
