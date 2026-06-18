#!/usr/bin/env python3
"""Side-by-side skeleton viz comparison for ab2 vs ab3 (GT | ab2 | ab3)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phi0.schema.action_schema import unpack_keypoints_52
from phi0.viz.skeleton import (
    apply_scene_limits,
    load_gt_from_hdf5,
    load_jsonl_predictions,
)
from phi0.viz.xperience_viz_frame import hdf5_keypoints_for_viz
from phi0.data.xperience import DEFAULT_HDF5

AB2 = ROOT / "experiments/ablation2_dit4dit_query_400step"
AB3 = ROOT / "experiments/ablation3_both_400step"
OUT = ROOT / "experiments/loss_comparison/ab2_vs_ab3_viz"

GT_COLOR = "darkgreen"
AB2_COLOR = "#1f77b4"
AB3_COLOR = "#d62728"


def _bone_pairs(keypoints: np.ndarray):
    from phi0.viz.skeleton import SMPLH_PARENTS

    for j in range(len(SMPLH_PARENTS)):
        p = int(SMPLH_PARENTS[j])
        if p >= 0:
            yield keypoints[p], keypoints[j]


def run_pipeline(exp_dir: Path, config_name: str, gpu: str) -> None:
    ckpt = exp_dir / f"{exp_dir.name}_latest.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    jsonl = exp_dir / "benchmark_deploy.jsonl"
    viz_dir = exp_dir / "viz_skeleton_5s"
    env = {"CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": f"{ROOT}/src:{ROOT}/../FastWAM/src"}

    if not jsonl.is_file():
        print(f"[{exp_dir.name}] benchmark_deploy...")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/benchmark_deploy.py"),
                "--checkpoint",
                str(ckpt),
                "--config-name",
                config_name,
                "--device",
                "cuda",
                "--deploy-seconds",
                "5",
                "--output",
                str(jsonl),
            ],
            check=True,
            cwd=ROOT,
            env={**dict(__import__("os").environ), **env},
        )
    if not (viz_dir / "summary.json").is_file():
        print(f"[{exp_dir.name}] visualize_skeleton...")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/visualize_skeleton.py"),
                "--predictions",
                str(jsonl),
                "--output-dir",
                str(viz_dir),
                "--max-frames",
                "100",
                "--fps",
                "15",
            ],
            check=True,
            cwd=ROOT,
        )


def load_keypoints(jsonl: Path, max_frames: int) -> tuple[np.ndarray, list[dict]]:
    d_raw, frames = load_jsonl_predictions(jsonl)
    n = min(max_frames, len(d_raw))
    return unpack_keypoints_52(d_raw[:n]), frames[:n]


def build_triple_animation(
    gt_kp: np.ndarray,
    ab2_kp: np.ndarray,
    ab3_kp: np.ndarray,
    out_dir: Path,
    fps: int,
) -> list[str]:
    n = min(len(gt_kp), len(ab2_kp), len(ab3_kp))
    gt_kp = gt_kp[:n]
    ab2_kp = ab2_kp[:n]
    ab3_kp = ab3_kp[:n]
    all_pts = np.concatenate([gt_kp.reshape(-1, 3), ab2_kp.reshape(-1, 3), ab3_kp.reshape(-1, 3)], axis=0)
    center = all_pts.mean(axis=0)
    radius = float(np.max(np.linalg.norm(all_pts - center, axis=1))) * 1.15 + 1e-3

    fig = plt.figure(figsize=(15, 5))
    axes = []
    labels = [("GT", GT_COLOR), ("ab2 DiT4DiT query", AB2_COLOR), ("ab3 both", AB3_COLOR)]
    keypoints_list = [gt_kp, ab2_kp, ab3_kp]
    line_groups = []
    for i, (label, color) in enumerate(labels):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        apply_scene_limits(ax, center, radius)
        ax.set_title(label)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        axes.append(ax)
        lines = []
        for _ in range(52):
            (line,) = ax.plot([], [], [], color=color, alpha=0.9, linewidth=1.3)
            lines.append(line)
        line_groups.append(lines)

    title = fig.suptitle("")

    def _update_lines(lines, kp, fi):
        for line, (p, c) in zip(lines, _bone_pairs(kp[fi])):
            line.set_data([p[0], c[0]], [p[1], c[1]])
            line.set_3d_properties([p[2], c[2]])

    def init():
        for lines, kp in zip(line_groups, keypoints_list):
            _update_lines(lines, kp, 0)
        title.set_text(f"GT vs ab2 vs ab3 — frame 0 / {n - 1}")
        return [ln for g in line_groups for ln in g] + [title]

    def update(fi):
        for lines, kp in zip(line_groups, keypoints_list):
            _update_lines(lines, kp, fi)
        title.set_text(f"GT vs ab2 vs ab3 — frame {fi} / {n - 1}")
        return [ln for g in line_groups for ln in g] + [title]

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=n, interval=max(1, int(1000 / fps)), blit=False
    )
    written = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, writer in [("gif", "pillow"), ("mp4", "ffmpeg")]:
        path = out_dir / f"skeleton_gt_ab2_ab3_sidebyside.{ext}"
        try:
            if ext == "gif":
                anim.save(path, writer=writer, fps=fps)
            else:
                anim.save(path, writer=writer, fps=fps, bitrate=1800)
            written.append(str(path))
        except Exception as exc:
            print(f"Warning: failed to save {path}: {exc}", file=sys.stderr)
    plt.close(fig)
    return written


def plot_l2_overlay(gt_kp: np.ndarray, ab2_kp: np.ndarray, ab3_kp: np.ndarray, out_path: Path) -> None:
    n = min(len(gt_kp), len(ab2_kp), len(ab3_kp))
    ab2_l2 = np.linalg.norm(ab2_kp[:n] - gt_kp[:n], axis=-1).mean(axis=1)
    ab3_l2 = np.linalg.norm(ab3_kp[:n] - gt_kp[:n], axis=-1).mean(axis=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ab2_l2, label=f"ab2 mean={ab2_l2.mean():.3f}", color=AB2_COLOR, linewidth=1.5)
    ax.plot(ab3_l2, label=f"ab3 mean={ab3_l2.mean():.3f}", color=AB3_COLOR, linewidth=1.5)
    ax.set_xlabel("control step")
    ax.set_ylabel("skeleton L2 (m)")
    ax.set_title("Per-frame skeleton error vs GT")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bone_length_overlay(
    gt_kp: np.ndarray, ab2_kp: np.ndarray, ab3_kp: np.ndarray, out_path: Path
) -> None:
    from phi0.viz.skeleton import iter_bone_segments

    gt_bl, ab2_bl, ab3_bl = [], [], []
    for (gp, gc), (a2p, a2c), (a3p, a3c) in zip(
        iter_bone_segments(gt_kp[0]),
        iter_bone_segments(ab2_kp[0]),
        iter_bone_segments(ab3_kp[0]),
    ):
        gt_bl.append(float(np.linalg.norm(gc - gp)))
        ab2_bl.append(float(np.linalg.norm(a2c - a2p)))
        ab3_bl.append(float(np.linalg.norm(a3c - a3p)))
    x = np.arange(len(gt_bl))
    w = 0.25
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(x - w, gt_bl, w, label="GT", color=GT_COLOR, alpha=0.85)
    ax.bar(x, ab2_bl, w, label="ab2", color=AB2_COLOR, alpha=0.85)
    ax.bar(x + w, ab3_bl, w, label="ab3", color=AB3_COLOR, alpha=0.85)
    ax.set_xlabel("bone index")
    ax.set_ylabel("length (m)")
    ax.set_title("Bone lengths frame 0 — GT vs ab2 vs ab3")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu-ab2", default="2")
    p.add_argument("--gpu-ab3", default="3")
    p.add_argument("--max-frames", type=int, default=100)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--skip-benchmark", action="store_true")
    args = p.parse_args()

    if not args.skip_benchmark:
        run_pipeline(AB2, "train_ablation2_dit4dit_query", args.gpu_ab2)
        run_pipeline(AB3, "train_ablation3_both", args.gpu_ab3)

    ab2_jsonl = AB2 / "benchmark_deploy.jsonl"
    ab3_jsonl = AB3 / "benchmark_deploy.jsonl"
    ab2_kp, ab2_frames = load_keypoints(ab2_jsonl, args.max_frames)
    ab3_kp, ab3_frames = load_keypoints(ab3_jsonl, args.max_frames)
    n = min(len(ab2_kp), len(ab3_kp))
    ab2_kp = ab2_kp[:n]
    ab3_kp = ab3_kp[:n]
    frames = ab2_frames[:n]

    start = int(frames[0].get("source_frame_index", 0))
    gt = load_gt_from_hdf5(DEFAULT_HDF5, start, n)
    gt_kp = hdf5_keypoints_for_viz(gt["keypoints_hdf5"])

    OUT.mkdir(parents=True, exist_ok=True)
    anim_paths = build_triple_animation(gt_kp, ab2_kp, ab3_kp, OUT, args.fps)
    l2_path = OUT / "skeleton_l2_ab2_vs_ab3.png"
    bone_path = OUT / "bone_length_gt_ab2_ab3.png"
    plot_l2_overlay(gt_kp, ab2_kp, ab3_kp, l2_path)
    plot_bone_length_overlay(gt_kp, ab2_kp, ab3_kp, bone_path)

    ab2_sum = json.loads((AB2 / "viz_skeleton_5s/summary.json").read_text())
    ab3_sum = json.loads((AB3 / "viz_skeleton_5s/summary.json").read_text())
    ab2_l2 = float(np.linalg.norm(ab2_kp - gt_kp, axis=-1).mean(axis=1).mean())
    ab3_l2 = float(np.linalg.norm(ab3_kp - gt_kp, axis=-1).mean(axis=1).mean())

    summary = {
        "ab2_dir": str(AB2),
        "ab3_dir": str(AB3),
        "n_frames": n,
        "ab2_skeleton_l2_mean": ab2_l2,
        "ab3_skeleton_l2_mean": ab3_l2,
        "ab2_viz_summary_l2": ab2_sum.get("skeleton_l2_mean"),
        "ab3_viz_summary_l2": ab3_sum.get("skeleton_l2_mean"),
        "ab2_bone_length_l1_frame0": ab2_sum.get("bone_length_l1_mean_frame0"),
        "ab3_bone_length_l1_frame0": ab3_sum.get("bone_length_l1_mean_frame0"),
        "outputs": {
            "triple_animation": anim_paths,
            "l2_overlay": str(l2_path),
            "bone_length_overlay": str(bone_path),
            "ab2_viz_dir": str(AB2 / "viz_skeleton_5s"),
            "ab3_viz_dir": str(AB3 / "viz_skeleton_5s"),
        },
    }
    summary_path = OUT / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Saved comparison to {OUT}")


if __name__ == "__main__":
    main()
