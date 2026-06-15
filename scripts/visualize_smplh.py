#!/usr/bin/env python3
"""OPTIONAL: Visualize SMPL+H predictions (disabled by default; requires manual SMPL-H license download)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Optional dependency on GR00T-WholeBodyControl smplx utils
SMPLX_UTILS = Path("/mnt/data1/wpy/workspace/GR00T-WholeBodyControl/gear_sonic/trl/utils/smplx")
if str(SMPLX_UTILS.parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(SMPLX_UTILS.parent.parent.parent.parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=str, required=True, help="JSONL from deploy_g1.py")
    p.add_argument("--output-dir", type=str, default="./viz_out")
    p.add_argument("--body-model-path", type=str, default="data/body_models")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    with open(args.predictions) as f:
        for line in f:
            frames.append(json.loads(line))
    summary = {
        "n_frames": len(frames),
        "note": "Full mesh rendering requires SMPL-H body models under body_model_path.",
        "body_model_path": args.body_model_path,
        "sample_root_trans": frames[0]["root_trans"] if frames else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.save(out_dir / "root_trans.npy", np.stack([f["root_trans"] for f in frames]))
    print(f"Wrote visualization summary to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
