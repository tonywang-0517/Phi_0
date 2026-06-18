#!/usr/bin/env python3
"""Side-by-side skeleton viz: GT | baseline | DiT4DiT (704×1280 ablation)."""

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

from phi0.data.xperience import DEFAULT_HDF5
from phi0.schema.action_schema import unpack_keypoints_52
from phi0.viz.skeleton import apply_scene_limits, load_gt_from_hdf5, load_jsonl_predictions
from phi0.viz.xperience_viz_frame import hdf5_keypoints_for_viz

BASELINE = ROOT / "experiments/ablation_baseline_704_400step"
DIT4DIT = ROOT / "experiments/ablation_dit4dit_query_704_400step"
OUT = ROOT / "experiments/loss_comparison/ablation_704_viz"

GT_COLOR = "darkgreen"
BASE_COLOR = "#1f77b4"
DIT_COLOR = "#ff7f0e"


def _bone_pairs(keypoints: np.ndarray):
    from phi0.viz.skeleton import SMPLH_PARENTS

    for j in range(len(SMPLH_PARENTS)):
        p = int(SMPLH_PARENTS[j])
        if p >= 0:
            yield keypoints[p], keypoints[j]


def ensure_viz(exp_dir: Path, config_name: str, gpu: str) -> None:
    ckpt = exp_dir / f"{exp_dir.name}_latest.pt"
    jsonl = exp_dir / "benchmark_deploy.jsonl"
    viz_dir = exp_dir / "viz_skeleton_5s"
    env = {**dict(__import__("os").environ), "CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": f"{ROOT}/src:{ROOT}/../FastWAM/src:{ROOT}/../vggt-omega"}

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
            env=env,
        )
    if not (viz_dir / "skeleton_gt_vs_pred_sidebyside.gif").is_file():
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
    base_kp: np.ndarray,
    dit_kp: np.ndarray,
    out_dir: Path,
    fps: int,
) -> list[str]:
    n = min(len(gt_kp), len(base_kp), len(dit_kp))
    gt_kp = gt_kp[:n]
    base_kp = base_kp[:n]
    dit_kp = dit_kp[:n]
    all_pts = np.concatenate(
        [gt_kp.reshape(-1, 3), base_kp.reshape(-1, 3), dit_kp.reshape(-1, 3)], axis=0
    )
    center = all_pts.mean(axis=0)
    radius = float(np.max(np.linalg.norm(all_pts - center, axis=1))) * 1.15 + 1e-3

    fig = plt.figure(figsize=(15, 5))
    axes = []
    labels = [
        ("GT", GT_COLOR),
        ("baseline (linear+proprio)", BASE_COLOR),
        ("DiT4DiT prefix/query", DIT_COLOR),
    ]
    keypoints_list = [gt_kp, base_kp, dit_kp]
    line_groups = []
    for i, (label, color) in enumerate(labels):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        apply_scene_limits(ax, center, radius)
        ax.set_title(label, fontsize=10)
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
        title.set_text(f"704 ablation — frame 0 / {n - 1}")
        return [ln for g in line_groups for ln in g] + [title]

    def update(fi):
        for lines, kp in zip(line_groups, keypoints_list):
            _update_lines(lines, kp, fi)
        title.set_text(f"704 ablation — frame {fi} / {n - 1}")
        return [ln for g in line_groups for ln in g] + [title]

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=n, interval=max(1, int(1000 / fps)), blit=False
    )
    written = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, writer in [("gif", "pillow"), ("mp4", "ffmpeg")]:
        path = out_dir / f"skeleton_gt_baseline_dit4dit_sidebyside.{ext}"
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


def plot_l2_overlay(gt_kp: np.ndarray, base_kp: np.ndarray, dit_kp: np.ndarray, out_path: Path) -> None:
    n = min(len(gt_kp), len(base_kp), len(dit_kp))
    base_l2 = np.linalg.norm(base_kp[:n] - gt_kp[:n], axis=-1).mean(axis=1)
    dit_l2 = np.linalg.norm(dit_kp[:n] - gt_kp[:n], axis=-1).mean(axis=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(base_l2, label=f"baseline mean={base_l2.mean():.3f}", color=BASE_COLOR, linewidth=1.5)
    ax.plot(dit_l2, label=f"DiT4DiT mean={dit_l2.mean():.3f}", color=DIT_COLOR, linewidth=1.5)
    ax.set_xlabel("control step")
    ax.set_ylabel("skeleton L2 (m)")
    ax.set_title("Per-frame skeleton error vs GT (704 ablation)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu-baseline", default="2")
    p.add_argument("--gpu-dit4dit", default="3")
    p.add_argument("--max-frames", type=int, default=100)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--skip-pipeline", action="store_true", help="Use existing benchmark/viz only")
    args = p.parse_args()

    if not args.skip_pipeline:
        ensure_viz(BASELINE, "train_ablation_baseline_704_400", args.gpu_baseline)
        ensure_viz(DIT4DIT, "train_ablation_dit4dit_query_704_400", args.gpu_dit4dit)

    base_jsonl = BASELINE / "benchmark_deploy.jsonl"
    dit_jsonl = DIT4DIT / "benchmark_deploy.jsonl"
    base_kp, base_frames = load_keypoints(base_jsonl, args.max_frames)
    dit_kp, _ = load_keypoints(dit_jsonl, args.max_frames)
    n = min(len(base_kp), len(dit_kp))
    base_kp = base_kp[:n]
    dit_kp = dit_kp[:n]
    frames = base_frames[:n]

    start = int(frames[0].get("source_frame_index", 0))
    gt = load_gt_from_hdf5(DEFAULT_HDF5, start, n)
    gt_kp = hdf5_keypoints_for_viz(gt["keypoints_hdf5"])

    OUT.mkdir(parents=True, exist_ok=True)
    anim_paths = build_triple_animation(gt_kp, base_kp, dit_kp, OUT, args.fps)
    l2_path = OUT / "skeleton_l2_baseline_vs_dit4dit.png"
    plot_l2_overlay(gt_kp, base_kp, dit_kp, l2_path)

    base_sum = json.loads((BASELINE / "viz_skeleton_5s/summary.json").read_text())
    dit_sum = json.loads((DIT4DIT / "viz_skeleton_5s/summary.json").read_text())
    base_l2 = float(np.linalg.norm(base_kp - gt_kp, axis=-1).mean(axis=1).mean())
    dit_l2 = float(np.linalg.norm(dit_kp - gt_kp, axis=-1).mean(axis=1).mean())

    summary = {
        "baseline_dir": str(BASELINE),
        "dit4dit_dir": str(DIT4DIT),
        "n_frames": n,
        "baseline_skeleton_l2_mean": base_l2,
        "dit4dit_skeleton_l2_mean": dit_l2,
        "baseline_viz_summary_l2": base_sum.get("skeleton_l2_mean"),
        "dit4dit_viz_summary_l2": dit_sum.get("skeleton_l2_mean"),
        "outputs": {
            "triple_animation": anim_paths,
            "l2_overlay": str(l2_path),
            "baseline_sidebyside": str(BASELINE / "viz_skeleton_5s/skeleton_gt_vs_pred_sidebyside.gif"),
            "dit4dit_sidebyside": str(DIT4DIT / "viz_skeleton_5s/skeleton_gt_vs_pred_sidebyside.gif"),
        },
    }
    summary_path = OUT / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved comparison to {OUT}")


if __name__ == "__main__":
    main()
