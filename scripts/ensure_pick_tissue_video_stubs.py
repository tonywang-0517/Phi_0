#!/usr/bin/env python3
"""Touch empty mp4 placeholders so LeRobot local init passes with partial video sync."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def ensure_stubs(dataset_root: Path, *, repo_id: str) -> int:
    from phi0.data.simple_lerobot import _import_lerobot

    LeRobotDataset, LeRobotDatasetMetadata = _import_lerobot()
    root = dataset_root.expanduser().resolve()
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"LeRobot dataset meta missing: {root / 'meta' / 'info.json'}")

    meta = LeRobotDatasetMetadata(repo_id, str(root))
    created = 0
    for ep in range(meta.total_episodes):
        for vid_key in meta.video_keys:
            path = root / meta.get_video_file_path(ep, vid_key)
            if path.is_file():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            created += 1
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "Isaac-GR00T/data/pick_tissue_xperience_unified",
    )
    parser.add_argument("--repo-id", default="pick_tissue_xperience_unified")
    args = parser.parse_args()
    n = ensure_stubs(args.dataset_root, repo_id=args.repo_id)
    print(f"created {n} video stub(s) under {args.dataset_root}")


if __name__ == "__main__":
    main()
