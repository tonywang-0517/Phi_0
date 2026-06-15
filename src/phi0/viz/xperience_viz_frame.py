"""Map Phi_0 FK joints into Xperience HDF5 keypoints visualization frame.

Xperience ``full_body_mocap/keypoints`` (T, 52, 3) is a self-consistent skeleton
graph in a dataset-specific frame. Joint 0 stores ``Ts_world_root`` quaternion xyz
(not pelvis translation); joints 1–51 are positions in the same frame. Do **not**
replace joint 0 with ``root_trans`` — that breaks parent–child bones (~1.2 m edges).

Minimal FK outputs SMPL-style coordinates; a fixed similarity transform (calibrated
on demo HDF5 frame 0, all 52 joints) maps FK → keypoints frame for pred overlay.
"""

from __future__ import annotations

import numpy as np

# Calibrated: xperience-10m-sample annotation.hdf5 frame 0 (all joints).
_FK_TO_KEYPOINTS_R = np.array(
    [
        [-0.6817178, -0.6058074, 0.41019294],
        [-0.71946174, 0.4533433, -0.5261698],
        [0.13279934, -0.6538174, -0.7449075],
    ],
    dtype=np.float32,
)
_FK_TO_KEYPOINTS_SCALE = np.float32(0.9773015)
_FK_TO_KEYPOINTS_T = np.array([0.6086176, 0.5804571, -0.7189194], dtype=np.float32)


def hdf5_keypoints_for_viz(keypoints_hdf5: np.ndarray) -> np.ndarray:
    """Return HDF5 keypoints as-is for skeleton drawing (joint 0 = root quat xyz anchor)."""
    return np.asarray(keypoints_hdf5, dtype=np.float32).copy()


def fk_joints_to_keypoints_frame(fk_joints: np.ndarray) -> np.ndarray:
    """Apply fixed FK→keypoints similarity transform to all joints uniformly."""
    fk = np.asarray(fk_joints, dtype=np.float32)
    return _FK_TO_KEYPOINTS_SCALE * (fk @ _FK_TO_KEYPOINTS_R) + _FK_TO_KEYPOINTS_T
