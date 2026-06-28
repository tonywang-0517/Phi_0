#!/usr/bin/env python3
"""Assert unified 360:396 qpos labels match pick_tissue_valid observation.state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from phi0.data.g1_qpos_from_wbc import body_dof29_from_wbc43
from phi0.data.pick_tissue_episode_map import dst_ep_to_unified_episode_index
from phi0.schema.unified_action_schema import unpack_g1_body_qpos_36


def verify_episode(
    valid_root: Path,
    unified_root: Path,
    *,
    unified_ep: int,
) -> None:
    uni_pq = unified_root / "data" / "chunk-000" / f"episode_{unified_ep:06d}.parquet"
    files = sorted((valid_root / "data").rglob("episode_*.parquet"))
    # reverse: unified index -> valid file used at rebuild (same sorted order)
    valid_pq = files[unified_ep]
    uni = pd.read_parquet(uni_pq)
    valid = pd.read_parquet(valid_pq)
    n = min(len(uni), len(valid))
    max_dof = 0.0
    max_quat = 0.0
    max_xyz = 0.0
    for i in range(n):
        q = unpack_g1_body_qpos_36(np.asarray(uni.iloc[i]["unified_action"], np.float32))
        st = np.asarray(valid.iloc[i]["observation.state"], np.float32)
        ro = np.asarray(valid.iloc[i]["observation.root_orientation"], np.float32)
        max_dof = max(max_dof, float(np.max(np.abs(q[7:] - body_dof29_from_wbc43(st)))))
        qs, qr = q[3:7], ro
        dot = min(1.0, abs(float(np.dot(qs / np.linalg.norm(qs), qr / np.linalg.norm(qr)))))
        max_quat = max(max_quat, float(np.degrees(2 * np.arccos(dot))))
        if "observation.base_trans" in valid.columns:
            bt = np.asarray(valid.iloc[i]["observation.base_trans"], np.float32).reshape(3)
            max_xyz = max(max_xyz, float(np.max(np.abs(q[:3] - bt))))
    msg = (
        f"unified_ep={unified_ep} valid_file={valid_pq.name} frames={n} "
        f"max_dof_err={max_dof:.2e} max_root_quat_err_deg={max_quat:.2e}"
    )
    if "observation.base_trans" in valid.columns:
        msg += f" max_root_xyz_err={max_xyz:.2e}"
    print(msg)
    if max_dof > 1e-4 or max_quat > 1e-3:
        raise SystemExit(f"label check FAILED ep {unified_ep}")
    if "observation.base_trans" in valid.columns and max_xyz > 1e-3:
        raise SystemExit(f"root xyz vs base_trans FAILED ep {unified_ep}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--valid-root", type=Path, default=Path("/mnt/data2/wpy/workspace/Isaac-GR00T/data/pick_tissue_valid"))
    p.add_argument("--unified-root", type=Path, default=Path("/mnt/data2/wpy/workspace/Isaac-GR00T/data/pick_tissue_xperience_unified"))
    p.add_argument("--unified-ep", type=int, default=448)
    p.add_argument("--dst-ep", type=int, default=None, help="resolve unified ep from manifest dst_ep filename")
    args = p.parse_args()
    uni_ep = args.unified_ep
    if args.dst_ep is not None:
        uni_ep = dst_ep_to_unified_episode_index(args.valid_root, args.dst_ep)
        print(f"dst_ep {args.dst_ep} -> unified_ep {uni_ep}")
    verify_episode(args.valid_root, args.unified_root, unified_ep=uni_ep)


if __name__ == "__main__":
    main()
