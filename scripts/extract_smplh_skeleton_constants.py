#!/usr/bin/env python3
"""Extract minimal SMPL-H skeleton constants for FK visualization (~11 KB).

Requires one-time download of SMPLX_NEUTRAL.npz (HuggingFace mirror) unless you
already have SMPL-H/SMPL-X body models locally.

Output: data/body_models/smplh_skeleton_constants.npz
  - J_template (52, 3)
  - J_shapedirs (52, 3, 16)
  - parents (52,)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phi0.viz.skeleton import SMPLH_PARENTS, SMPLX_TO_XPERIENCE_JOINT_REMAP  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "body_models" / "smplh_skeleton_constants.npz"


def extract_from_npz(npz_path: Path, n_joints: int = 52, num_betas: int = 16) -> dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    J_regressor = data["J_regressor"]
    if hasattr(J_regressor, "toarray"):
        J_regressor = J_regressor.toarray()
    J_regressor = np.asarray(J_regressor, dtype=np.float32)
    v_template = np.asarray(data["v_template"], dtype=np.float32)
    shapedirs = np.asarray(data["shapedirs"], dtype=np.float32)

    # SMPL-X first 55 joints, then remap to Xperience / GR00T 52-joint layout.
    smplx_joints = n_joints + 3  # need indices up to 54 for hand remap
    J_template_full = J_regressor[:smplx_joints] @ v_template
    J_shapedirs_full = np.einsum(
        "jv,vcd->jcd", J_regressor[:smplx_joints], shapedirs[:, :, :num_betas]
    )
    remap = SMPLX_TO_XPERIENCE_JOINT_REMAP
    J_template = J_template_full[remap]
    J_shapedirs = J_shapedirs_full[remap]
    return {
        "J_template": J_template.astype(np.float32),
        "J_shapedirs": J_shapedirs.astype(np.float32),
        "parents": SMPLH_PARENTS.astype(np.int32),
        "joint_remap": remap,
        "source": str(npz_path),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Extract SMPL-H skeleton FK constants")
    p.add_argument(
        "--smplx-npz",
        type=str,
        default=None,
        help="Path to SMPLX_NEUTRAL.npz (or download from HuggingFace if omitted)",
    )
    p.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--hf-repo", type=str, default="fffiloni/PSHuman-SMPL-related")
    p.add_argument(
        "--hf-filename",
        type=str,
        default="models/smplx/SMPLX_NEUTRAL.npz",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.smplx_npz:
        npz_path = Path(args.smplx_npz)
        if not npz_path.is_file():
            raise SystemExit(f"Not found: {npz_path}")
    else:
        from huggingface_hub import hf_hub_download

        cache = out_path.parent / "_hf_cache"
        npz_path = Path(
            hf_hub_download(
                repo_id=args.hf_repo,
                filename=args.hf_filename,
                local_dir=str(cache),
            )
        )
        print(f"Downloaded {npz_path}")

    payload = extract_from_npz(npz_path)
    np.savez_compressed(out_path, **payload)
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
