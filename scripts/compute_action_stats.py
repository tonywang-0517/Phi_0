#!/usr/bin/env python3
"""Pre-compute Phi_0 action z-score statistics (FastWAM / DiT4DiT-style)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phi0.data.action_stats import compute_action_stats_from_datasets, save_action_stats
from phi0.data.processor import build_overfit_datasets


def parse_args():
    p = argparse.ArgumentParser(description="Compute Phi_0 D_raw action normalization stats")
    p.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "checkpoints/phi0_action_stats.json"),
        help="Output JSON path (mean/std per D_raw dim)",
    )
    p.add_argument("--xperience-max-frames", type=int, default=256)
    p.add_argument("--egodex-max-frames", type=int, default=256)
    p.add_argument("--xperience-video", type=str, default=None)
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    return p.parse_args()


def main():
    args = parse_args()
    video_path = args.xperience_video
    if video_path is not None and str(video_path).lower() in {"", "null", "none"}:
        video_path = None

    datasets = build_overfit_datasets(
        xperience_max_frames=int(args.xperience_max_frames),
        egodex_max_frames=int(args.egodex_max_frames),
        xperience_video=video_path,
        cache_video=False,
    ).datasets

    total_frames = sum(len(ds) for ds in datasets)
    names = [getattr(ds, "DATASET_NAME", type(ds).__name__) for ds in datasets]
    print(
        f"Scanning {len(datasets)} dataset(s): {', '.join(names)} ({total_frames} frames total)",
        flush=True,
    )

    stats = compute_action_stats_from_datasets(
        list(datasets),
        show_progress=not args.no_progress,
    )
    out = save_action_stats(stats, args.output)

    supervised = sum(stats.get("supervised_mask", []))
    kp_std = stats["std"][:156]
    print(
        f"Done: {stats['num_frames']} frames, {supervised}/{stats['action_dim']} supervised dims, "
        f"keypoints std range [{min(kp_std):.4f}, {max(kp_std):.4f}]",
        flush=True,
    )
    print(json.dumps({"written": str(out.resolve()), "num_frames": stats["num_frames"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
