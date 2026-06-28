#!/usr/bin/env python3
"""Patch observation.base_trans column into pick-tissue LeRobot parquets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tyro


def _ensure_info_json(info_path: Path) -> None:
    if not info_path.is_file():
        return
    info = json.loads(info_path.read_text(encoding="utf-8"))
    feats = info.setdefault("features", {})
    if "observation.base_trans" in feats:
        return
    feats["observation.base_trans"] = {
        "dtype": "float64",
        "shape": [3],
        "names": ["base_x", "base_y", "base_z"],
    }
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"[patch] updated {info_path}")


def patch_one(parquet: Path, base_trans: np.ndarray) -> None:
    df = pd.read_parquet(parquet)
    n = min(len(df), len(base_trans))
    if n == 0:
        return
    col = [base_trans[i].astype(np.float64).tolist() for i in range(n)]
    df = df.iloc[:n].copy()
    df["observation.base_trans"] = col
    df.to_parquet(parquet, index=False)
    span = base_trans[:n].max(0) - base_trans[:n].min(0)
    print(f"[patch] {parquet.name} frames={n} xyz_span={span}")


def main(
    parquet: Path,
    base_trans_npy: Path,
    info_json: Path | None = None,
) -> None:
    base = np.load(base_trans_npy)
    patch_one(parquet, base)
    if info_json is not None:
        _ensure_info_json(info_json)


if __name__ == "__main__":
    tyro.cli(main)
