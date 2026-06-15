#!/usr/bin/env python3
"""Preprocess raw EgoDex HDF5 episodes into sparse Phi_0 D_raw (256-d) + dim masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phi0.data.egodex_smplh import (  # noqa: E402
    default_processed_path,
    iter_raw_egodex_hdf5,
    preprocess_egodex_file,
)
from phi0.schema.draw_schema import D_RAW  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EgoDex -> sparse SMPL+H D_raw preprocessor")
    p.add_argument(
        "input",
        nargs="?",
        default="/mnt/data2/wpy/workspace/Isaac-GR00T/demo_data/egodex",
        help="Raw EgoDex .hdf5 file or directory tree",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory (default: alongside each source file as <stem>_smplh.hdf5)",
    )
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--min-confidence", type=float, default=0.25)
    p.add_argument(
        "--world-frame",
        action="store_true",
        help="Keep ARKit world frame instead of camera frame (default: camera frame)",
    )
    p.add_argument(
        "--tactile-proxy",
        action="store_true",
        help="Fill tactile slice with fingertip-distance proxy (still sparse if tips missing)",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing processed files")
    return p.parse_args()


def _output_path(src: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return default_processed_path(src)
    return output_dir / f"{src.stem}_smplh.hdf5"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    sources = [input_path] if input_path.is_file() else list(iter_raw_egodex_hdf5(input_path))
    if not sources:
        raise SystemExit(f"No raw EgoDex HDF5 files under {input_path}")

    summaries = []
    for src in sources:
        out = _output_path(src, output_dir)
        if out.is_file() and not args.force:
            print(f"Skip (exists): {out}")
            continue
        info = preprocess_egodex_file(
            src,
            out,
            camera_frame=not args.world_frame,
            min_confidence=args.min_confidence,
            include_tactile_proxy=args.tactile_proxy,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
        )
        summaries.append(info)
        print(
            f"OK {src.name} -> {out.name} "
            f"frames={info['num_frames']} dims={info['dim_available_count']}/{D_RAW} "
            f"({info['dim_available_ratio']:.1%})"
        )

    if summaries:
        summary_path = (output_dir or sources[0].parent) / "preprocess_egodex_smplh_summary.json"
        summary_path.write_text(json.dumps(summaries, indent=2))
        print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
