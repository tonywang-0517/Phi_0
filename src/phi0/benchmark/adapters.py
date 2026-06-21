from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from phi0.benchmark.image_utils import center_crop_like_vla, resize_image_for_policy
from phi0.benchmark.rlds_adapters import calvin_eval_gripper_flip  # noqa: F401
from phi0.benchmark.vla_types import VLAObservation
from phi0.data.dit4dit_video import dit4dit_preprocess_frame


def make_vla_prompt(instruction: str) -> str:
    """Prompt template aligned with VLA-Adapter/OpenVLA eval path."""
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def normalize_gripper_action(action: np.ndarray, *, binarize: bool = True) -> np.ndarray:
    """[0,1] -> [-1,1], optional binarization."""
    out = np.asarray(action, dtype=np.float32).copy()
    if out.ndim == 0:
        out = out.reshape(1)
    grip = out[..., -1]
    grip = 2.0 * grip - 1.0
    if binarize:
        sign = np.sign(grip)
        sign = np.where(sign == 0.0, 1.0, sign)
        grip = sign
    out[..., -1] = grip
    return out


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    out = action.copy()
    out[..., -1] *= -1.0
    return out


def process_vla_action(action: np.ndarray, *, invert_openvla_gripper: bool = False) -> np.ndarray:
    out = normalize_gripper_action(action, binarize=True)
    if invert_openvla_gripper:
        out = invert_gripper_action(out)
    return out


def process_libero_absolute_eef_action(
    action: np.ndarray,
    *,
    invert_openvla_gripper: bool = True,
) -> np.ndarray:
    """OSC absolute pose: pass pos + axis-angle through; only gripper uses VLA env convention."""
    out = np.asarray(action, dtype=np.float32).copy()
    if out.ndim == 1:
        grip = out[6:7]
        grip = normalize_gripper_action(grip, binarize=True)
        if invert_openvla_gripper:
            grip = invert_gripper_action(grip)
        out[6:7] = grip
        return out
    grip = out[..., 6:7]
    grip = normalize_gripper_action(grip, binarize=True)
    if invert_openvla_gripper:
        grip = invert_gripper_action(grip)
    out[..., 6:7] = grip
    return out


def libero_obs_to_eef_7d(obs: dict[str, Any]) -> np.ndarray:
    """Absolute EEF from LIBERO sim obs (aligned with ``libero_rlds_state_to_eef_7d``)."""
    import math

    def quat2axisangle(quat: np.ndarray) -> np.ndarray:
        q = quat.astype(np.float32).copy()
        q[3] = np.clip(q[3], -1.0, 1.0)
        den = np.sqrt(max(1e-8, 1.0 - q[3] * q[3]))
        if math.isclose(float(den), 0.0):
            return np.zeros(3, dtype=np.float32)
        return (q[:3] * 2.0 * math.acos(float(q[3])) / den).astype(np.float32)

    from phi0.benchmark.rlds_adapters import libero_gripper_qpos_to_train

    gripper = libero_gripper_qpos_to_train(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32))
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)),
            np.asarray([gripper], dtype=np.float32),
        ],
        axis=0,
    )


def libero_obs_to_native_frame(
    obs: dict[str, Any],
    *,
    camera: str = "agentview_image",
    flip: bool = True,
) -> torch.Tensor:
    """CHW float [0,1] at sim native resolution (RLDS-aligned flip, no Cosmos resize)."""
    img = np.asarray(obs[camera], dtype=np.uint8)
    if flip:
        img = img[::-1, ::-1]
    return torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0


def libero_obs_to_train_frame(
    obs: dict[str, Any],
    *,
    target_size: Tuple[int, int],
    crop_scale: Optional[float] = None,
    camera: str = "agentview_image",
    flip: bool = True,
) -> torch.Tensor:
    """CHW float [0,1] aligned with OpenVLA RLDS + ``dit4dit_preprocess``.

    OpenVLA ``modified_libero_rlds`` stores agentview already rotated 180°; LIBERO sim
   needs ``[::-1, ::-1]`` to match (see VLA-Adapter ``libero_utils.get_libero_image``).
    """
    frame = libero_obs_to_native_frame(obs, camera=camera, flip=flip)
    th, tw = int(target_size[0]), int(target_size[1])
    if frame.shape[1] != th or frame.shape[2] != tw:
        frame = F.interpolate(
            frame.unsqueeze(0),
            size=(th, tw),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return dit4dit_preprocess_frame(frame, size=(th, tw), crop_scale=crop_scale)


def libero_obs_to_vla(
    obs: dict[str, Any],
    *,
    image_size: int = 224,
    center_crop: bool = True,
) -> VLAObservation:
    """
    Convert LIBERO env observation to VLA-style observation.

    state = [eef_pos(3), eef_axis_angle(3), gripper_qpos(1)].
    """
    import math

    def quat2axisangle(quat: np.ndarray) -> np.ndarray:
        q = quat.astype(np.float32).copy()
        q[3] = np.clip(q[3], -1.0, 1.0)
        den = np.sqrt(max(1e-8, 1.0 - q[3] * q[3]))
        if math.isclose(float(den), 0.0):
            return np.zeros(3, dtype=np.float32)
        return (q[:3] * 2.0 * math.acos(float(q[3])) / den).astype(np.float32)

    full = np.asarray(obs["agentview_image"])[::-1, ::-1]
    wrist = np.asarray(obs["robot0_eye_in_hand_image"])[::-1, ::-1]
    full = resize_image_for_policy(full, image_size)
    wrist = resize_image_for_policy(wrist, image_size)
    if center_crop:
        full = center_crop_like_vla(full)
        wrist = center_crop_like_vla(wrist)
    state = np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ],
        axis=0,
    )
    return VLAObservation(full_image=full, wrist_image=wrist, state=state, raw=obs)


def calvin_obs_to_vla(
    obs: dict[str, Any],
    *,
    image_size: int = 224,
    center_crop: bool = True,
) -> VLAObservation:
    """
    Convert CALVIN obs to VLA-style observation.

    state = [robot_obs[:7], robot_obs[-1]] to match VLA-Adapter CALVIN setup.
    """
    full = np.asarray(obs["rgb_obs"]["rgb_static"], dtype=np.uint8)
    wrist = np.asarray(obs["rgb_obs"]["rgb_gripper"], dtype=np.uint8)
    full = resize_image_for_policy(full, image_size)
    wrist = resize_image_for_policy(wrist, image_size)
    if center_crop:
        full = center_crop_like_vla(full)
        wrist = center_crop_like_vla(wrist)
    robot_obs = np.asarray(obs["robot_obs"], dtype=np.float32)
    state = np.concatenate([robot_obs[:7], robot_obs[-1:]], axis=0).astype(np.float32)
    return VLAObservation(full_image=full, wrist_image=wrist, state=state, raw=obs)

