"""SONIC unified I/O for pi0.5 and Phi0 training.

State (43-d, Psi0 finetune-real-sonic-psi0 layout):
    qpos(29) = legs(12) + waist(3) + arms(14), then hands(14)

Action (100-d):
    pi05 whole-body base (36) + motion_token (64)

Source columns (GR00T / Isaac-GR00T LeRobot):
    observation.state[43], action.wbc[43], action.motion_token[64], teleop.*
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

SONIC_STATE_DIM = 43
SONIC_ACTION_BASE_DIM = 36
SONIC_MOTION_TOKEN_DIM = 64
SONIC_ACTION_DIM = SONIC_ACTION_BASE_DIM + SONIC_MOTION_TOKEN_DIM
STATS_SEMANTICS_SONIC_UNIFIED = "sonic_unified_43s_100a"

# Joint slices inside observation.state / action.wbc (SONIC modality.json).
LEFT_LEG = slice(0, 6)
RIGHT_LEG = slice(6, 12)
WAIST = slice(12, 15)
LEFT_KNEE = 3
RIGHT_KNEE = 9
LEFT_HAND = slice(22, 29)
RIGHT_HAND = slice(36, 43)
LEFT_ARM = slice(15, 22)
RIGHT_ARM = slice(29, 36)

QPOS_SLICES = [(0, 15), (15, 22), (29, 36)]  # legs+waist, larm, rarm -> 29
HAND_SLICES = [(22, 29), (36, 43)]  # -> 14

DEFAULT_TORSO_HEIGHT = 0.75
MIN_TORSO_HEIGHT = 0.24
PLANNER_DISABLED = -1.0
KNEE_STAND_REF = 0.25
KNEE_SQUAT_REF = 0.90
HAND_DEGENERATE_EPS = 1e-6


def take_slices(vec: np.ndarray, slices: list[tuple[int, int]]) -> np.ndarray:
    return np.concatenate([vec[a:b] for a, b in slices])


def _scalar(val: Any, default: float = 0.0) -> float:
    arr = np.asarray(val, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return default
    return float(arr[0])


def _vec7(val: Any) -> np.ndarray | None:
    if val is None:
        return None
    arr = np.asarray(val, dtype=np.float64).reshape(-1)
    if arr.size != 7:
        return None
    return arr.astype(np.float32)


def _is_degenerate_hand(hands: np.ndarray) -> bool:
    return bool(np.max(np.abs(hands)) < HAND_DEGENERATE_EPS)


def hands_from_wbc(vec: np.ndarray) -> np.ndarray:
    return np.concatenate([vec[LEFT_HAND], vec[RIGHT_HAND]]).astype(np.float32)


def hands_from_teleop(row: Mapping[str, Any]) -> np.ndarray | None:
    left = _vec7(row.get("teleop.left_hand_joints"))
    right = _vec7(row.get("teleop.right_hand_joints"))
    if left is None or right is None:
        return None
    return np.concatenate([left, right]).astype(np.float32)


def resolve_hands(row: Mapping[str, Any], primary: np.ndarray, *, fallback: np.ndarray | None = None) -> np.ndarray:
    hands = hands_from_wbc(primary)
    if _is_degenerate_hand(hands) and fallback is not None:
        hands = hands_from_wbc(fallback)
    if _is_degenerate_hand(hands):
        teleop_hands = hands_from_teleop(row)
        if teleop_hands is not None and not _is_degenerate_hand(teleop_hands):
            hands = teleop_hands
    return hands.astype(np.float32)


def arms_from_wbc(vec: np.ndarray) -> np.ndarray:
    return np.concatenate([vec[LEFT_ARM], vec[RIGHT_ARM]]).astype(np.float32)


def torso_rpy_from_wbc(vec: np.ndarray) -> np.ndarray:
    waist = vec[WAIST]
    return np.asarray([waist[1], waist[2], waist[0]], dtype=np.float32)


def torso_height_from_knees(vec: np.ndarray) -> float:
    knee = 0.5 * (float(vec[LEFT_KNEE]) + float(vec[RIGHT_KNEE]))
    t = float(np.clip((knee - KNEE_STAND_REF) / (KNEE_SQUAT_REF - KNEE_STAND_REF), 0.0, 1.0))
    return float(DEFAULT_TORSO_HEIGHT + t * (MIN_TORSO_HEIGHT - DEFAULT_TORSO_HEIGHT))


def torso_height_from_row(row: Mapping[str, Any], *, wbc43: np.ndarray | None = None) -> float:
    height = _scalar(row.get("teleop.planner_height"), PLANNER_DISABLED)
    if height >= 0.0:
        return height
    if wbc43 is not None:
        return torso_height_from_knees(wbc43)
    state43 = row.get("observation.state")
    if state43 is not None:
        return torso_height_from_knees(np.asarray(state43, dtype=np.float64))
    return DEFAULT_TORSO_HEIGHT


def locomotion_from_row(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    movement = np.asarray(row.get("teleop.planner_movement", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
    speed = _scalar(row.get("teleop.planner_speed"), PLANNER_DISABLED)
    if speed < 0:
        speed = 0.0
    vx = float(movement[0] * speed) if movement.size > 0 else 0.0
    vy = float(movement[1] * speed) if movement.size > 1 else 0.0
    vyaw = float(movement[2] * speed) if movement.size > 2 else 0.0
    target_yaw = _scalar(row.get("teleop.delta_heading"), 0.0)
    return vx, vy, vyaw, target_yaw


def build_state_from_state43(state43: np.ndarray) -> np.ndarray:
    vec = np.asarray(state43, dtype=np.float64).reshape(-1)
    if vec.shape[0] != 43:
        raise ValueError(f"expected state43 dim 43, got {vec.shape}")
    states = np.concatenate([take_slices(vec, QPOS_SLICES), take_slices(vec, HAND_SLICES)])
    if states.shape != (SONIC_STATE_DIM,):
        raise ValueError(f"expected states dim {SONIC_STATE_DIM}, got {states.shape}")
    return states.astype(np.float32)


def build_pi05_base_action_from_row(row: Mapping[str, Any]) -> np.ndarray:
    """36-d pi05 whole-body action (hand+arm+torso+locomotion)."""
    wbc43 = np.asarray(row["action.wbc"], dtype=np.float64)
    height = torso_height_from_row(row, wbc43=wbc43)
    vx, vy, vyaw, target_yaw = locomotion_from_row(row)
    action = np.concatenate(
        [
            resolve_hands(row, wbc43),
            arms_from_wbc(wbc43),
            torso_rpy_from_wbc(wbc43),
            np.asarray([height, vx, vy, vyaw, target_yaw], dtype=np.float32),
        ]
    )
    if action.shape != (SONIC_ACTION_BASE_DIM,):
        raise ValueError(f"expected base action dim {SONIC_ACTION_BASE_DIM}, got {action.shape}")
    return action.astype(np.float32)


def build_action_from_sonic_row(row: Mapping[str, Any]) -> np.ndarray:
    """100-d = pi05 base (36) + motion_token (64)."""
    base = build_pi05_base_action_from_row(row)
    token = np.asarray(row["action.motion_token"], dtype=np.float32).reshape(-1)
    if token.shape != (SONIC_MOTION_TOKEN_DIM,):
        raise ValueError(f"expected motion_token dim {SONIC_MOTION_TOKEN_DIM}, got {token.shape}")
    action = np.concatenate([base, token])
    if action.shape != (SONIC_ACTION_DIM,):
        raise ValueError(f"expected action dim {SONIC_ACTION_DIM}, got {action.shape}")
    return action.astype(np.float32)


def build_state_from_sonic_row(row: Mapping[str, Any]) -> np.ndarray:
    state43 = np.asarray(row["observation.state"], dtype=np.float64)
    return build_state_from_state43(state43)


def build_state_from_sonic_frame(row: pd.Series) -> list[float]:
    return build_state_from_sonic_row(row).tolist()


def build_action_from_sonic_frame(row: pd.Series) -> list[float]:
    return build_action_from_sonic_row(row).tolist()
