"""RLDS step -> VLA observation / 7D action (VLA-Adapter conventions)."""

from __future__ import annotations

import numpy as np

from phi0.benchmark.image_utils import center_crop_like_vla, resize_image_for_policy
from phi0.benchmark.rlds_io import RldsStep
from phi0.benchmark.vla_types import VLAObservation


def _prep_image(img: np.ndarray, *, image_size: int, center_crop: bool, flip: bool) -> np.ndarray:
    out = np.asarray(img, dtype=np.uint8)
    if flip:
        out = out[::-1, ::-1]
    out = resize_image_for_policy(out, image_size)
    if center_crop:
        out = center_crop_like_vla(out)
    return out


def libero_rlds_to_vla_obs(
    step: RldsStep,
    *,
    image_size: int = 224,
    center_crop: bool = True,
) -> VLAObservation:
    full = _prep_image(step.rgb_static, image_size=image_size, center_crop=center_crop, flip=False)
    wrist = _prep_image(step.rgb_gripper, image_size=image_size, center_crop=center_crop, flip=False)
    state = np.asarray(step.state, dtype=np.float32).reshape(-1)
    if state.shape[0] >= 8:
        state = np.concatenate([state[:3], state[3:6], state[6:7]], axis=0)
    return VLAObservation(full_image=full, wrist_image=wrist, state=state, raw={"rlds": step})


def calvin_rlds_to_env_obs(step: RldsStep) -> dict:
    return {
        "rgb_obs": {
            "rgb_static": np.asarray(step.rgb_static, dtype=np.uint8),
            "rgb_gripper": np.asarray(step.rgb_gripper, dtype=np.uint8),
        },
        "robot_obs": np.asarray(step.state, dtype=np.float32).reshape(-1),
    }


def libero_gripper_qpos_to_train(gripper_qpos: np.ndarray, *, open_threshold: float = 0.025) -> float:
    """Map LIBERO finger qpos to OpenVLA gripper label (1=open, 0=close)."""
    q = np.asarray(gripper_qpos, dtype=np.float32).reshape(-1)
    opening = float(np.mean(np.abs(q)))
    return 1.0 if opening > float(open_threshold) else 0.0


def libero_rlds_state_to_eef_7d(state: np.ndarray) -> np.ndarray:
    """Absolute EEF at the current RLDS step: pos(3) + axis-angle(3) + gripper(1).

    Matches ``libero_obs_to_eef_7d`` / OpenVLA ``observation/state`` layout.
    """
    s = np.asarray(state, dtype=np.float32).reshape(-1)
    if s.shape[0] < 7:
        raise ValueError(f"LIBERO RLDS state must have >=7 dims, got {s.shape[0]}")
    gripper = libero_gripper_qpos_to_train(s[6:8] if s.shape[0] >= 8 else s[6:7])
    return np.concatenate([s[:3], s[3:6], np.asarray([gripper], dtype=np.float32)], axis=0)


def libero_rlds_action_to_train(action: np.ndarray) -> np.ndarray:
    """Match VLA-Adapter libero_dataset_transform gripper (+1 open, 0 close)."""
    out = np.asarray(action, dtype=np.float32).reshape(7).copy()
    g = float(np.clip(out[6], 0.0, 1.0))
    out[6] = 1.0 - g
    return out


def calvin_rlds_action_to_train(action: np.ndarray) -> np.ndarray:
    """Match VLA-Adapter calvin_dataset_transform gripper (clip only)."""
    out = np.asarray(action, dtype=np.float32).reshape(7).copy()
    out[6] = float(np.clip(out[6], 0.0, 1.0))
    return out


def calvin_eval_gripper_flip(action: np.ndarray) -> np.ndarray:
    """Pre-process bridge output before process_vla_action (VLA-Adapter CALVIN eval)."""
    out = np.asarray(action, dtype=np.float32).copy()
    out[..., 6] = 1.0 - np.clip(out[..., 6], 0.0, 1.0)
    return out
