from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from phi0.schema.action_schema import unpack_keypoints_52


@dataclass
class ProjectionConfig:
    """
    Heuristic projection from Phi_0 keypoints (52x3) to 7D arm action.

    Output order: [dx, dy, dz, droll, dpitch, dyaw, gripper].
    """

    wrist_joint: int = 21
    hand_tip_joint: int = 51
    max_delta_pos: float = 0.03
    gripper_open_threshold: float = 0.06
    gripper_close_threshold: float = 0.04


class KeypointToArmActionProjector:
    """
    Converts predicted d_raw chunk to VLA-style arm actions.

    Notes:
    - This is a bootstrap heuristic so eval can run end-to-end.
    - Replace with IK/controller-level projection for best performance.
    """

    def __init__(self, cfg: Optional[ProjectionConfig] = None) -> None:
        self.cfg = cfg or ProjectionConfig()
        self._prev_wrist: Optional[np.ndarray] = None
        self._prev_gripper: float = 1.0

    def reset(self) -> None:
        self._prev_wrist = None
        self._prev_gripper = 1.0

    def _gripper_from_pose(self, joints52: np.ndarray) -> float:
        wrist = joints52[self.cfg.wrist_joint]
        tip = joints52[self.cfg.hand_tip_joint]
        dist = float(np.linalg.norm(tip - wrist))
        if dist >= self.cfg.gripper_open_threshold:
            self._prev_gripper = 1.0
        elif dist <= self.cfg.gripper_close_threshold:
            self._prev_gripper = 0.0
        return float(self._prev_gripper)

    def project_chunk(self, pred_norm_chunk: torch.Tensor, processor) -> np.ndarray:
        """
        Args:
            pred_norm_chunk: [T, D] normalized output from ActionInferenceSession.predict
            processor: Phi0Processor for denormalization
        Returns:
            np.ndarray: [T, 7] relative actions
        """
        if pred_norm_chunk.ndim != 2:
            raise ValueError(f"Expected [T,D], got {tuple(pred_norm_chunk.shape)}")
        with torch.no_grad():
            d_raw = (
                processor.postprocess(pred_norm_chunk.unsqueeze(0))
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        keypoints = unpack_keypoints_52(d_raw)  # [T,52,3]

        out = np.zeros((keypoints.shape[0], 7), dtype=np.float32)
        for t in range(keypoints.shape[0]):
            wrist = keypoints[t, self.cfg.wrist_joint].astype(np.float32)
            prev = wrist if self._prev_wrist is None else self._prev_wrist
            delta = wrist - prev
            delta = np.clip(delta, -self.cfg.max_delta_pos, self.cfg.max_delta_pos)

            out[t, :3] = delta
            out[t, 3:6] = 0.0  # Keep orientation static in bootstrap projector
            out[t, 6] = self._gripper_from_pose(keypoints[t])
            self._prev_wrist = wrist
        return out

