#!/usr/bin/env python3
"""Offline-decode LeRobot episode MP4s to mmap-friendly uint8 npy for training."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from phi0.data.predecoded_video import (  # noqa: E402
    PREDECODED_VERSION,
    PredecodedVideoMeta,
    predecode_one_episode,
    predecoded_root,
    read_episodes_jsonl,
    validate_predecoded_store,
    write_store_meta,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path("/mnt/data2/wpy/workspace/Isaac-GR00T/data/pick_tissue_xperience_unified")
DEFAULT_KEYS = (
    "observation.images.ego_view",
    "observation.images.left_wrist",
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    p.add_argument(
        "--native",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store source resolution; Psi0 Resize+CenterCrop runs at VLM time (default: on)",
    )
    p.add_argument("--image-size", type=int, nargs=2, default=(180, 320), metavar=("H", "W"))
    p.add_argument("--video-keys", nargs="*", default=list(DEFAULT_KEYS))
    p.add_argument(
        "--backend",
        choices=("cv2", "torchcodec"),
        default="cv2",
        help="cv2: fast sequential full-episode decode (offline). torchcodec: random timestamps (training).",
    )
    p.add_argument("--max-episodes", type=int, default=0, help="0 = all episodes")
    p.add_argument("--start-episode", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument(
        "--skip-store-meta",
        action="store_true",
        help="decode only; skip meta.json (for parallel shard workers)",
    )
    args = p.parse_args()

    dataset_root = args.dataset_root
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    chunks_size = int(info.get("chunks_size", 1000))
    if args.native:
        from phi0.data.psi0_image import read_lerobot_video_hw

        image_size = read_lerobot_video_hw(dataset_root, args.video_keys[0])
        logger.info("native predecode (Psi0): store at %sx%s", image_size[0], image_size[1])
    else:
        image_size = (int(args.image_size[0]), int(args.image_size[1]))
        logger.warning(
            "legacy resized predecode %sx%s; prefer --native for Psi0-aligned training",
            image_size[0],
            image_size[1],
        )
    store_root = predecoded_root(dataset_root, image_size)
    episodes = read_episodes_jsonl(dataset_root / "meta")
    episodes = [e for e in episodes if int(e["episode_index"]) >= int(args.start_episode)]
    if args.max_episodes > 0:
        episodes = episodes[: int(args.max_episodes)]

    if args.validate_only:
        max_val = int(args.max_episodes) if args.max_episodes > 0 else None
        errs = validate_predecoded_store(
            dataset_root,
            store_root,
            video_keys=args.video_keys,
            max_episodes=max_val,
        )
        if errs:
            logger.error("validation failed (%d issues):", len(errs))
            for e in errs[:20]:
                logger.error("  %s", e)
            if len(errs) > 20:
                logger.error("  ... and %d more", len(errs) - 20)
            raise SystemExit(1)
        logger.info("validation OK: %s", store_root)
        return

    store_root.mkdir(parents=True, exist_ok=True)
    records = []
    for row in episodes:
        ep = int(row["episode_index"])
        length = int(row["length"])
        for key in args.video_keys:
            rec = predecode_one_episode(
                dataset_root=dataset_root,
                store_root=store_root,
                episode_index=ep,
                episode_length=length,
                video_key=key,
                image_size=image_size,
                fps=fps,
                chunks_size=chunks_size,
                backend=args.backend,
                tolerance_s=1.0 / fps,
                overwrite=bool(args.overwrite),
            )
            records.append(rec)
            logger.info(
                "ep %06d %s -> %s shape=%s",
                ep,
                key.split(".")[-1],
                rec.path,
                rec.shape,
            )

    if not args.skip_store_meta:
        write_store_meta(
            store_root,
            PredecodedVideoMeta(
                version=PREDECODED_VERSION,
                image_size=image_size,
                layout="THWC",
                dtype="uint8",
                fps=fps,
                video_keys=tuple(args.video_keys),
                total_episodes=len(read_episodes_jsonl(dataset_root / "meta")),
                backend=str(args.backend),
            ),
        )
        errs = validate_predecoded_store(dataset_root, store_root, video_keys=args.video_keys)
        if errs:
            raise RuntimeError(f"post-decode validation failed: {errs[0]}")
    logger.info("done: %d episodes x %d keys -> %s", len(episodes), len(args.video_keys), store_root)


if __name__ == "__main__":
    main()
