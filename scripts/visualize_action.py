#!/usr/bin/env python3
"""Lightweight Phi_0 action visualization — no SMPL-H body model download required."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phi0.schema.action_schema import D_RAW, LEGACY_QUAT_SLICES as SLICES
from phi0.viz.skeleton import draw_skeleton, load_gt_from_hdf5, load_jsonl_predictions

FINGER_TACTILE_NAMES = [
    "L_thumb", "L_index", "L_middle", "L_ring", "L_little",
    "R_thumb", "R_index", "R_middle", "R_ring", "R_little",
]

EULER_GROUPS = {
    "root": (SLICES["root_quat"][0], 1),
    "body": (SLICES["body_quats"][0], 21),
    "left_hand": (SLICES["left_hand_quats"][0], 15),
    "right_hand": (SLICES["right_hand_quats"][0], 15),
}


def parse_args():
    p = argparse.ArgumentParser(description="Visualize Phi_0 actions without SMPL-H mesh models")
    p.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="JSONL from deploy_g1.py (or synthetic with d_raw field)",
    )
    p.add_argument("--output-dir", type=str, default="./viz_action_out")
    p.add_argument(
        "--hdf5",
        type=str,
        default=None,
        help="Optional Xperience annotation.hdf5 for GT keypoints / root overlay",
    )
    p.add_argument("--start-frame", type=int, default=0, help="HDF5 frame offset when aligning GT")
    p.add_argument("--dpi", type=int, default=120)
    return p.parse_args()


def _quat_wxyz_to_euler_deg(quats: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 3) roll,pitch,yaw degrees."""
    from scipy.spatial.transform import Rotation

    flat = quats.reshape(-1, 4)
    # scipy expects xyzw
    xyzw = flat[:, [1, 2, 3, 0]]
    euler = Rotation.from_quat(xyzw).as_euler("xyz", degrees=True)
    return euler.reshape(*quats.shape[:-1], 3)


def plot_tactile_timeseries(d_raw: np.ndarray, out_path: Path, dpi: int):
    import matplotlib.pyplot as plt

    tactile = d_raw[:, SLICES["tactile"][0]:SLICES["tactile"][1]]
    if np.allclose(tactile, 0):
        return False
    t = np.arange(len(d_raw))
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, name in enumerate(FINGER_TACTILE_NAMES):
        ax.plot(t, tactile[:, i], label=name, alpha=0.85)
    ax.set_xlabel("frame")
    ax.set_ylabel("tactile proxy")
    ax.set_title("10-finger tactile (predicted / packed)")
    ax.legend(ncol=2, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def plot_component_groups(d_raw: np.ndarray, out_path: Path, dpi: int):
    import matplotlib.pyplot as plt

    groups = {
        "root_trans": d_raw[:, SLICES["root_trans"][0]:SLICES["root_trans"][1]],
        "betas (first 8)": d_raw[:, SLICES["betas"][0]:SLICES["betas"][0] + 8],
    }
    t = np.arange(len(d_raw))
    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 2.5 * len(groups)), sharex=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, (title, data) in zip(axes, groups.items()):
        for j in range(data.shape[1]):
            ax.plot(t, data[:, j], alpha=0.8, label=f"d{j}")
        ax.set_ylabel(title)
        ax.legend(ncol=min(8, data.shape[1]), fontsize=6, loc="upper right")
    axes[-1].set_xlabel("frame")
    fig.suptitle("Action component time series", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_euler_groups(d_raw: np.ndarray, out_path: Path, dpi: int):
    import matplotlib.pyplot as plt

    has_quats = not np.allclose(d_raw[:, SLICES["body_quats"][0]:SLICES["body_quats"][1]], 0)
    if not has_quats:
        return False

    t = np.arange(len(d_raw))
    fig, axes = plt.subplots(len(EULER_GROUPS), 1, figsize=(10, 2.2 * len(EULER_GROUPS)), sharex=True)
    for ax, (name, (start, n_joints)) in zip(axes, EULER_GROUPS.items()):
        quats = d_raw[:, start:start + n_joints * 4].reshape(len(d_raw), n_joints, 4)
        euler = _quat_wxyz_to_euler_deg(quats)
        # mean abs euler per joint group for readability
        for j in range(n_joints):
            ax.plot(t, euler[:, j, 0], alpha=0.35, color="C0")
            ax.plot(t, euler[:, j, 1], alpha=0.35, color="C1")
            ax.plot(t, euler[:, j, 2], alpha=0.35, color="C2")
        ax.plot(t, euler.mean(axis=1)[:, 0], color="C0", label="roll (mean)")
        ax.plot(t, euler.mean(axis=1)[:, 1], color="C1", label="pitch (mean)")
        ax.plot(t, euler.mean(axis=1)[:, 2], color="C2", label="yaw (mean)")
        ax.set_ylabel(f"{name} euler (deg)")
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("frame")
    fig.suptitle("Quaternion → Euler (per joint group)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_root_trajectory(
    pred_root: np.ndarray,
    gt_root: np.ndarray | None,
    out_path: Path,
    dpi: int,
):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pred_root[:, 0], pred_root[:, 1], pred_root[:, 2], "b-o", markersize=4, label="pred root")
    ax.scatter(
        pred_root[0, 0], pred_root[0, 1], pred_root[0, 2], c="cyan", s=40, label="pred start"
    )
    ax.scatter(
        pred_root[-1, 0], pred_root[-1, 1], pred_root[-1, 2], c="blue", s=40, label="pred end"
    )
    if gt_root is not None:
        ax.plot(gt_root[:, 0], gt_root[:, 1], gt_root[:, 2], "r--", alpha=0.8, label="GT root")
        ax.scatter(gt_root[0, 0], gt_root[0, 1], gt_root[0, 2], c="orange", s=40, label="GT start")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Root translation trajectory (world)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_gt_skeleton_frames(
    keypoints: np.ndarray,
    pred_root: np.ndarray | None,
    out_path: Path,
    dpi: int,
):
    import matplotlib.pyplot as plt

    n = len(keypoints)
    pick = [0, n // 2, n - 1] if n >= 3 else list(range(n))
    fig = plt.figure(figsize=(4 * len(pick), 4))
    for i, fi in enumerate(pick):
        ax = fig.add_subplot(1, len(pick), i + 1, projection="3d")
        kp = keypoints[fi]
        draw_skeleton(ax, kp, color="darkgreen")
        if pred_root is not None:
            ax.scatter(
                pred_root[fi, 0], pred_root[fi, 1], pred_root[fi, 2],
                c="blue", s=30, label="pred root",
            )
        ax.set_title(f"GT skeleton frame {fi}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # equal aspect for readability
        center = kp.mean(axis=0)
        radius = float(np.max(np.linalg.norm(kp - center, axis=1))) * 1.2 + 1e-3
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
    fig.suptitle("GT SMPL-H keypoints (52 joints) — no body model needed", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    out_dir: Path,
    d_raw: np.ndarray,
    frames: list[dict],
    gt: dict[str, np.ndarray] | None,
    written: list[str],
):
    pred_root = d_raw[:, 0:3]
    summary = {
        "n_frames": len(frames),
        "d_raw_shape": list(d_raw.shape),
        "has_tactile": bool(not np.allclose(d_raw[:, SLICES["tactile"][0]:SLICES["tactile"][1]], 0)),
        "has_joint_quats": bool(
            not np.allclose(d_raw[:, 7:211], 0)
        ),
        "pred_root_trans_range": {
            "min": pred_root.min(axis=0).tolist(),
            "max": pred_root.max(axis=0).tolist(),
        },
        "gt_aligned": gt is not None,
        "plots": written,
        "note": "Mesh rendering requires SMPL-H .pkl; this script uses trajectories and GT keypoints only.",
    }
    if gt is not None:
        gt_root = gt["root_trans"]
        summary["root_l2_per_frame"] = np.linalg.norm(pred_root - gt_root, axis=1).tolist()
        summary["root_l2_mean"] = float(np.mean(summary["root_l2_per_frame"]))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        raise SystemExit("matplotlib required: pip install matplotlib")

    pred_path = Path(args.predictions)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d_raw, frames = load_jsonl_predictions(pred_path)
    start = int(frames[0].get("source_frame_index", args.start_frame))
    gt = None
    if args.hdf5:
        gt = load_gt_from_hdf5(Path(args.hdf5), start, len(frames))

    written: list[str] = []
    if plot_tactile_timeseries(d_raw, out_dir / "tactile_timeseries.png", args.dpi):
        written.append("tactile_timeseries.png")
    plot_component_groups(d_raw, out_dir / "action_components.png", args.dpi)
    written.append("action_components.png")
    if plot_euler_groups(d_raw, out_dir / "euler_angles.png", args.dpi):
        written.append("euler_angles.png")

    gt_root = gt["root_trans"] if gt is not None else None
    plot_root_trajectory(d_raw[:, 0:3], gt_root, out_dir / "root_trajectory.png", args.dpi)
    written.append("root_trajectory.png")

    if gt is not None:
        plot_gt_skeleton_frames(
            gt["keypoints"], d_raw[:, 0:3], out_dir / "gt_skeleton.png", args.dpi
        )
        written.append("gt_skeleton.png")
        np.save(out_dir / "gt_keypoints.npy", gt["keypoints"])

    np.save(out_dir / "pred_d_raw.npy", d_raw)
    write_summary(out_dir, d_raw, frames, gt, written)
    print(f"Wrote {len(written)} plots + summary to {out_dir}")
    for name in written:
        print(f"  - {out_dir / name}")


if __name__ == "__main__":
    main()
