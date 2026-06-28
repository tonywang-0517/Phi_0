#!/usr/bin/env python3
"""Visualize unified action GT from Xperience: official SMPL-X mesh vs quat reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phi0.data.xperience import DEFAULT_HDF5
from phi0.data.xperience_unified_gt import (
    pack_xperience_unified_frame_gt,
    read_xperience_betas,
    read_xperience_root_trans_world,
    validate_unified_gt_fk_matches_hdf5_quat,
)
from phi0.schema.unified_action_schema import root_trans_world_from_unified
from phi0.viz.skeleton import (
    apply_scene_limits,
    configure_mpl3d_skeleton_axes,
    draw_ground_plane,
)
from phi0.viz.smplh_fk import load_skeleton_constants
from phi0.viz.smplx_mesh import (
    compute_mesh_viz_bounds,
    draw_smplx_mesh,
    hdf5_quat_frame_to_smplx_inputs,
    hdf5_quat_transl_world,
    smplx_forward_mesh,
    unified_action_to_smplx_inputs,
)


def parse_args():
    p = argparse.ArgumentParser(description="FK-validate Xperience unified action GT (SMPL-X mesh)")
    p.add_argument("--hdf5", type=str, default=str(DEFAULT_HDF5))
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=32)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--state-t-offset", type=int, default=0, help="state_t = t - offset")
    p.add_argument("--output-dir", type=str, default="./viz_xperience_unified_gt")
    p.add_argument("--atol", type=float, default=2e-3)
    p.add_argument("--make-gif", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    constants = load_skeleton_constants()

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    frames: list[np.ndarray] = []
    errors: list[float] = []
    viz_rows: list[dict] = []
    with h5py.File(args.hdf5, "r") as f:
        end = min(int(f["full_body_mocap/body_quats"].shape[0]), args.start + args.count * args.stride)
        for t in range(args.start, end, args.stride):
            state_t = max(0, t - args.state_t_offset)
            gt = pack_xperience_unified_frame_gt(f, t, state_t=state_t)
            metrics = validate_unified_gt_fk_matches_hdf5_quat(
                f, t, state_t=state_t, atol=args.atol, constants=constants
            )
            betas = read_xperience_betas(f, t)
            state_root = read_xperience_root_trans_world(f, state_t)
            gt_inputs = unified_action_to_smplx_inputs(gt.action, betas=betas)
            ref_inputs = hdf5_quat_frame_to_smplx_inputs(f, t, betas)
            gt_trans = root_trans_world_from_unified(gt.action, state_root)
            ref_trans = hdf5_quat_transl_world(f, t)
            gt_v, mesh_faces = smplx_forward_mesh(gt_inputs, transl_world=gt_trans)
            ref_v, _ = smplx_forward_mesh(ref_inputs, transl_world=ref_trans)
            errors.append(metrics["max_abs_m"])
            viz_rows.append(
                {
                    "t": t,
                    "state_t": state_t,
                    "gt_v": gt_v,
                    "ref_v": ref_v,
                    "metrics": metrics,
                }
            )

    all_verts: list[np.ndarray] = []
    for row in viz_rows:
        all_verts.extend((row["gt_v"], row["ref_v"]))

    bounds_center, bounds_radius = compute_mesh_viz_bounds(*all_verts)

    for row in viz_rows:
        t = row["t"]
        state_t = row["state_t"]
        metrics = row["metrics"]
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        draw_ground_plane(ax, bounds_center, bounds_radius, z=0.0)
        draw_smplx_mesh(ax, row["ref_v"], mesh_faces, color="tab:orange", alpha=0.35)
        draw_smplx_mesh(ax, row["gt_v"], mesh_faces, color="tab:blue", alpha=0.45)
        apply_scene_limits(ax, bounds_center, bounds_radius, ground_at_z0=True)
        configure_mpl3d_skeleton_axes(ax)
        ax.legend(
            handles=[
                Line2D([0], [0], color="tab:blue", lw=2, label="GT unified SMPL-X"),
                Line2D(
                    [0],
                    [0],
                    color="tab:orange",
                    lw=2,
                    label=f"HDF5 quat SMPL-X (FK err {metrics['max_abs_m']:.2e} m)",
                ),
            ],
            loc="upper left",
            fontsize=8,
        )
        fig.suptitle(
            f"Xperience unified GT validation  t={t}  state_t={state_t}  "
            f"world_FK_err={metrics['max_abs_m']:.2e}m  [Isaac world mesh → standing display]"
        )
        fig.tight_layout()
        png = out_dir / f"frame_{t:05d}.png"
        fig.savefig(png, dpi=120)
        plt.close(fig)
        frames.append(plt.imread(png))

    summary = out_dir / "fk_error_summary.txt"
    summary.write_text(
        "\n".join(
            [
                f"frames={len(errors)}",
                f"max_error_m={max(errors) if errors else 0:.6f}",
                f"mean_error_m={float(np.mean(errors)) if errors else 0:.6f}",
                f"atol={args.atol}",
            ]
        ),
        encoding="utf-8",
    )
    print(summary.read_text())

    if args.make_gif and frames:
        try:
            import imageio.v2 as imageio

            gif_path = out_dir / "unified_gt_fk_validation.gif"
            imageio.mimsave(gif_path, frames, duration=0.12)
            print(f"Wrote {gif_path}")
        except ImportError:
            print("imageio not installed; skipped GIF")


if __name__ == "__main__":
    main()
