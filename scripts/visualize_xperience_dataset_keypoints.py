#!/usr/bin/env python3
"""Visualize Xperience dataset GT skeleton from ``full_body_mocap/keypoints`` as-is.

Original viz path (see ``scripts/visualize_skeleton.py``):
- HDF5 keypoints are drawn directly — joint 0 is ``Ts_world_root`` quat xyz anchor,
  not pelvis translation; do not FK-remap or world-transform for display.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phi0.data.xperience import DEFAULT_HDF5
from phi0.viz.skeleton import apply_scene_limits, draw_skeleton
from phi0.viz.xperience_viz_frame import compute_keypoints_viz_bounds, hdf5_keypoints_for_viz
    p = argparse.ArgumentParser(description="Dataset GT keypoints skeleton (HDF5 as-is)")
    p.add_argument("--hdf5", type=str, default=str(DEFAULT_HDF5))
    p.add_argument("--start", type=int, default=16, help="First mocap frame index")
    p.add_argument("--count", type=int, default=32, help="Number of frames")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "experiments/xperience_dataset_keypoints_viz"),
    )
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--gif-duration", type=float, default=0.15)
    p.add_argument("--no-gif", action="store_true")
    return p.parse_args()


def parse_args():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    frame_indices: list[int] = []
    keypoints_seq: list[np.ndarray] = []

    with h5py.File(args.hdf5, "r") as f:
        n_total = int(f["full_body_mocap/keypoints"].shape[0])
        end = min(n_total, args.start + args.count * args.stride)
        for t in range(args.start, end, max(1, args.stride)):
            raw = f["full_body_mocap/keypoints"][t].astype(np.float32)
            keypoints_seq.append(hdf5_keypoints_for_viz(raw))
            frame_indices.append(t)

    if not keypoints_seq:
        raise SystemExit("no frames selected")

    stack = np.stack(keypoints_seq, axis=0)
    center, radius = compute_keypoints_viz_bounds(stack)

    png_paths: list[Path] = []
    for t, kp in zip(frame_indices, keypoints_seq):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        draw_skeleton(ax, kp, color="darkgreen", alpha=0.95, linewidth=1.3)
        apply_scene_limits(ax, center, radius)
        ax.view_init(elev=15, azim=-70)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(f"Dataset keypoints GT  t={t}")
        fig.suptitle(
            f"Xperience HDF5 keypoints as-is  start={args.start} stride={args.stride}  "
            f"[fixed scene bounds, no FK/world remap]"
        )
        fig.tight_layout()
        png = out_dir / f"keypoints_{t:05d}.png"
        fig.savefig(png, dpi=args.dpi)
        plt.close(fig)
        png_paths.append(png)

    if not args.no_gif and png_paths:
        try:
            import imageio.v2 as imageio

            gif_path = out_dir / "dataset_keypoints_gt.gif"
            imageio.mimsave(gif_path, [imageio.imread(p) for p in png_paths], duration=args.gif_duration)
            print(f"Wrote {gif_path}")
        except ImportError:
            print("imageio not installed; skipped GIF")

    summary = out_dir / "summary.txt"
    summary.write_text(
        "\n".join(
            [
                f"hdf5={args.hdf5}",
                f"frames={len(frame_indices)}",
                f"indices={frame_indices[0]}..{frame_indices[-1]} stride={args.stride}",
                f"scene_center={center.tolist()}",
                f"scene_radius={radius:.4f}",
                "method=full_body_mocap/keypoints as-is (hdf5_keypoints_for_viz)",
            ]
        ),
        encoding="utf-8",
    )
    print(summary.read_text())
    print(f"Wrote {len(png_paths)} PNGs to {out_dir}")


if __name__ == "__main__":
    main()
