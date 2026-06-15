"""SMPL-H 52-joint skeleton helpers for Phi_0 (no body-model mesh rendering).

Reused / adapted from:
- GR00T-WholeBodyControl/gear_sonic/trl/utils/smplx/smplx_utils.py  (SMPLH_PARENTS)
- Isaac-GR00T/demo_data/xperience-10m-sample/README.md  (keypoints T,52,3 — **not** world joint xyz; use FK)
- Phi_0/src/phi0/data/xperience.py  (HDF5 paths)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from phi0.schema.draw_schema import D_RAW
from phi0.schema.action_schema import pack_xperience_keypoints

# fmt: off
# Source: GR00T-WholeBodyControl/gear_sonic/trl/utils/smplx/smplx_utils.py
SMPLH_PARENTS = np.array(
    [
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
        16, 17, 18, 19, 20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34,
        35, 21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50,
    ],
    dtype=np.int32,
)
# fmt: on

NUM_JOINTS = len(SMPLH_PARENTS)

# SMPL-X has 55 joints; indices 22–24 are face (parent=head). Xperience / GR00T 52-joint
# keypoints omit those and pack left/right hand chains at 22–36 and 37–51 instead.
# Source: compare SMPL-X kintree vs SMPLH_PARENTS (GR00T smplx_utils.py).
SMPLX_TO_XPERIENCE_JOINT_REMAP = np.array(
    list(range(22)) + list(range(25, 40)) + list(range(40, 55)),
    dtype=np.int32,
)


def skeleton_edges(parents: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Return (parent, child) pairs for drawing bones."""
    parents = SMPLH_PARENTS if parents is None else parents
    return [(int(parents[j]), j) for j in range(len(parents)) if int(parents[j]) >= 0]


def draw_skeleton(
    ax,
    keypoints: np.ndarray,
    *,
    color: str = "darkgreen",
    alpha: float = 0.9,
    linewidth: float = 1.2,
    parents: np.ndarray | None = None,
) -> None:
    """Draw one 52-joint skeleton on a matplotlib 3D axis."""
    parents = SMPLH_PARENTS if parents is None else parents
    for parent, child in skeleton_edges(parents):
        xs = [keypoints[parent, 0], keypoints[child, 0]]
        ys = [keypoints[parent, 1], keypoints[child, 1]]
        zs = [keypoints[parent, 2], keypoints[child, 2]]
        ax.plot(xs, ys, zs, color=color, alpha=alpha, linewidth=linewidth)


def set_equal_3d_limits(ax, points: np.ndarray, margin: float = 1.2) -> None:
    """Set cubic axis limits centered on joint cloud."""
    center = points.reshape(-1, 3).mean(axis=0)
    radius = float(np.max(np.linalg.norm(points.reshape(-1, 3) - center, axis=1))) * margin + 1e-3
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def load_jsonl_predictions(path: Path) -> tuple[np.ndarray, list[dict]]:
    """Load deploy JSONL; return (d_raw[T,D_RAW], raw frame dicts)."""
    frames: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    if not frames:
        raise ValueError(f"No frames in {path}")

    if "d_raw" in frames[0]:
        d_raw = np.stack([np.asarray(fr["d_raw"], dtype=np.float32) for fr in frames], axis=0)
        return d_raw, frames

    if "keypoints_52" in frames[0]:
        d_raw = np.zeros((len(frames), D_RAW), dtype=np.float32)
        for i, fr in enumerate(frames):
            kp = np.asarray(fr["keypoints_52"], dtype=np.float32).reshape(-1)
            d_raw[i, : kp.shape[0]] = kp
        return d_raw, frames

    raise ValueError(
        f"{path} must contain 'd_raw' or 'keypoints_52' per frame; got keys {list(frames[0].keys())}"
    )


def load_gt_d_raw_from_hdf5(hdf5_path: Path, start: int, count: int) -> np.ndarray:
    """Pack GT D_raw rows from Xperience annotation.hdf5 (same as XperienceDataset)."""
    import h5py

    rows: list[np.ndarray] = []
    with h5py.File(hdf5_path, "r") as f:
        end = start + count
        for t in range(start, end):
            keypoints = f["full_body_mocap/keypoints"][t].astype(np.float32)
            betas = f["full_body_mocap/betas"][t].astype(np.float32)
            rows.append(pack_xperience_keypoints(keypoints, betas))
    return np.stack(rows, axis=0)


def load_gt_from_hdf5(hdf5_path: Path, start: int, count: int) -> dict[str, np.ndarray]:
    """Load GT root / packed pose from Xperience annotation.hdf5.

    ``full_body_mocap/keypoints`` is a self-consistent skeleton graph; joint 0 holds
    ``Ts_world_root`` quaternion xyz (not translation). Use as-is for viz — see
    ``phi0.viz.xperience_viz_frame.hdf5_keypoints_for_viz``.
    """
    import h5py

    out: dict[str, np.ndarray] = {}
    with h5py.File(hdf5_path, "r") as f:
        end = start + count
        out["keypoints_hdf5"] = f["full_body_mocap/keypoints"][start:end].astype(np.float32)
        out["root_trans"] = f["full_body_mocap/Ts_world_root"][start:end, :3].astype(np.float32)
        out["left_hand_joints"] = f["hand_mocap/left_joints_3d"][start:end].astype(np.float32)
        out["right_hand_joints"] = f["hand_mocap/right_joints_3d"][start:end].astype(np.float32)
    out["d_raw"] = load_gt_d_raw_from_hdf5(hdf5_path, start, count)
    # Back-compat alias — do not use for skeleton drawing.
    out["keypoints"] = out["keypoints_hdf5"]
    return out


def subsample_frame_indices(n_frames: int, max_frames: int | None) -> np.ndarray:
    if max_frames is None or max_frames >= n_frames:
        return np.arange(n_frames, dtype=np.int32)
    return np.linspace(0, n_frames - 1, max_frames, dtype=int)


def compute_scene_bounds(
    gt_keypoints: np.ndarray | None,
    pred_root: np.ndarray,
    gt_root: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Return (center, radius) for consistent camera framing."""
    chunks: list[np.ndarray] = [pred_root.reshape(-1, 3)]
    if gt_root is not None:
        chunks.append(gt_root.reshape(-1, 3))
    if gt_keypoints is not None:
        chunks.append(gt_keypoints.reshape(-1, 3))
    pts = np.concatenate(chunks, axis=0)
    center = pts.mean(axis=0)
    radius = float(np.max(np.linalg.norm(pts - center, axis=1))) * 1.15 + 1e-3
    return center, radius


def apply_scene_limits(ax, center: np.ndarray, radius: float) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def iter_bone_segments(keypoints: np.ndarray, parents: np.ndarray | None = None) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield (parent_xyz, child_xyz) for each bone at one frame."""
    parents = SMPLH_PARENTS if parents is None else parents
    for parent, child in skeleton_edges(parents):
        yield keypoints[parent], keypoints[child]
