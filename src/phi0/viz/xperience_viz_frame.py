"""Map Phi_0 FK joints into Xperience HDF5 ``keypoints`` visualization frame.

Xperience ``full_body_mocap/keypoints`` (T, 52, 3) is a self-consistent skeleton
graph in a dataset-specific frame. Joint 0 stores ``Ts_world_root`` quaternion **xyz**
(not pelvis translation); joints 1–51 are positions in the same frame.

FK from ``Ts_world_root`` + body/hand quats lives in a different absolute frame
(mocap / SMPL ``transl`` convention).  **A single global Sim(3) does not align a
moving clip**; use per-frame Procrustes on joints 1–51 (~3 cm vs HDF5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Sim3:
    """Similarity transform: ``dst ≈ scale * (src @ rotation) + translation``."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return apply_sim3(points, self)


def fit_sim3_procrustes(src: np.ndarray, dst: np.ndarray) -> Sim3:
    """Fit Sim(3) mapping ``src`` (N,3) → ``dst`` (N,3) via SVD (Umeyama)."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError(f"fit_sim3_procrustes needs matching (N,3), N>=3; got {src.shape}, {dst.shape}")

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    x = src - mu_s
    y = dst - mu_d
    cov = (x.T @ y) / src.shape[0]
    u, singular, vt = np.linalg.svd(cov)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        vt = vt.copy()
        vt[-1, :] *= -1.0
        rotation = u @ vt
    var_s = float((x * x).sum() / src.shape[0])
    scale = float(singular.sum() / var_s) if var_s > 1e-12 else 1.0
    translation = mu_d - scale * (mu_s @ rotation)
    return Sim3(
        scale=scale,
        rotation=rotation.astype(np.float32),
        translation=translation.astype(np.float32),
    )


def apply_sim3(points: np.ndarray, sim3: Sim3) -> np.ndarray:
    """Apply Sim(3) to points with shape (..., 3)."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    out = sim3.scale * (pts @ sim3.rotation) + sim3.translation.reshape(1, 3)
    return out.reshape(points.shape).astype(np.float32)


def keypoints_joint0_from_root(ts_world_root: np.ndarray) -> np.ndarray:
    """Joint-0 slot in HDF5 keypoints: root quaternion xyz (wxyz stored at 3:7)."""
    root = np.asarray(ts_world_root, dtype=np.float32).reshape(7)
    return root[4:7].copy()


def hdf5_keypoints_for_viz(keypoints_hdf5: np.ndarray) -> np.ndarray:
    """Return HDF5 keypoints as-is for skeleton drawing (joint 0 = root quat xyz anchor)."""
    return np.asarray(keypoints_hdf5, dtype=np.float32).copy()


def align_fk_joints_to_keypoints_frame(
    fk_joints: np.ndarray,
    target_keypoints: np.ndarray,
) -> tuple[np.ndarray, Sim3]:
    """Per-frame Sim(3): align FK joints 1–51 to HDF5 ``keypoints`` 1–51.

    Returns:
        aligned: (52, 3) in keypoints frame; joint 0 copied from ``target_keypoints``.
        sim3: transform fitted on joints 1–51.
    """
    fk = np.asarray(fk_joints, dtype=np.float32).reshape(52, 3)
    target = np.asarray(target_keypoints, dtype=np.float32).reshape(52, 3)
    sim3 = fit_sim3_procrustes(fk[1:], target[1:])
    aligned = np.empty_like(target)
    aligned[0] = target[0]
    aligned[1:] = sim3.apply(fk[1:])
    return aligned, sim3


def procrustes_joint_errors(
    fk_joints: np.ndarray,
    target_keypoints: np.ndarray,
) -> np.ndarray:
    """Per-joint L2 (m) on indices 1–51 after Sim(3) alignment."""
    aligned, _ = align_fk_joints_to_keypoints_frame(fk_joints, target_keypoints)
    target = np.asarray(target_keypoints, dtype=np.float32).reshape(52, 3)
    return np.linalg.norm(aligned[1:] - target[1:], axis=-1).astype(np.float32)


def fk_joints_to_keypoints_frame(
    fk_joints: np.ndarray,
    target_keypoints: np.ndarray,
) -> np.ndarray:
    """Align FK joints into the HDF5 keypoints frame (per-frame Sim(3))."""
    aligned, _ = align_fk_joints_to_keypoints_frame(fk_joints, target_keypoints)
    return aligned


def world_fk_to_dataset_keypoints_viz(
    fk_joints: np.ndarray,
    target_keypoints: np.ndarray,
) -> np.ndarray:
    """Map mocap-world FK (52, 3) into the HDF5 keypoints visualization frame."""
    return fk_joints_to_keypoints_frame(fk_joints, target_keypoints)


def compute_keypoints_viz_bounds(
    *joint_sets: np.ndarray,
    margin: float = 1.15,
) -> tuple[np.ndarray, float]:
    """Fixed cubic scene bounds for keypoints-frame skeleton plots."""
    if not joint_sets:
        raise ValueError("compute_keypoints_viz_bounds needs at least one joint array")
    pts = np.concatenate(
        [np.asarray(j, dtype=np.float32).reshape(-1, 3) for j in joint_sets],
        axis=0,
    )
    center = pts.mean(axis=0)
    radius = float(np.max(np.linalg.norm(pts - center, axis=1))) * margin + 1e-3
    return center.astype(np.float32), radius
