#!/usr/bin/env python3
"""Aggregate LeRobot parquet columns into meta/stats.json (+ optional norm_stats.json)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _running_agg(acc: dict[str, np.ndarray] | None, batch: np.ndarray) -> dict[str, np.ndarray]:
    batch = np.asarray(batch, dtype=np.float64)
    if batch.ndim == 1:
        batch = batch.reshape(1, -1)
    if acc is None:
        return {
            "count": batch.shape[0],
            "sum": batch.sum(0),
            "sumsq": (batch**2).sum(0),
            "min": batch.min(0),
            "max": batch.max(0),
        }
    acc["count"] += batch.shape[0]
    acc["sum"] += batch.sum(0)
    acc["sumsq"] += (batch**2).sum(0)
    acc["min"] = np.minimum(acc["min"], batch.min(0))
    acc["max"] = np.maximum(acc["max"], batch.max(0))
    return acc


def _finalize(acc: dict[str, np.ndarray]) -> dict[str, list[float]]:
    count = int(acc["count"])
    mean = acc["sum"] / count
    var = np.maximum(acc["sumsq"] / count - mean**2, 0.0)
    std = np.sqrt(var)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": acc["min"].tolist(),
        "max": acc["max"].tolist(),
        "q01": acc["min"].tolist(),
        "q99": acc["max"].tolist(),
        "count": [count],
    }


def compute_stats(dataset_root: Path, fields: list[str]) -> dict[str, dict]:
    parquets = sorted((dataset_root / "data").rglob("episode_*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No episode parquet under {dataset_root / 'data'}")
    acc: dict[str, dict[str, np.ndarray] | None] = {f: None for f in fields}
    total = 0
    for pq in parquets:
        df = pd.read_parquet(pq, columns=fields)
        n = len(df)
        total += n
        for field in fields:
            values = np.stack([np.asarray(x, dtype=np.float64) for x in df[field].to_numpy()])
            acc[field] = _running_agg(acc[field], values)
    out = {field: _finalize(acc[field]) for field in fields}
    out["num_frames"] = total
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["states", "action"],
        help="Parquet column names to aggregate",
    )
    parser.add_argument(
        "--write-norm-stats",
        action="store_true",
        help="Also write norm_stats.json for openpi (state/actions keys)",
    )
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    stats = compute_stats(root, args.fields)
    num_frames = int(stats.pop("num_frames"))
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    payload = dict(stats)
    payload["num_frames"] = num_frames
    with open(meta_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    if args.write_norm_stats:
        state_key = "states" if "states" in stats else "state"
        action_key = "action" if "action" in stats else "actions"
        norm = {
            "norm_stats": {
                "state": stats[state_key],
                "actions": stats[action_key],
            }
        }
        with open(root / "norm_stats.json", "w", encoding="utf-8") as f:
            json.dump(norm, f, indent=2)
            f.write("\n")

    dims = {k: len(v["mean"]) for k, v in stats.items()}
    print(f"Wrote {meta_dir / 'stats.json'}: frames={num_frames}, dims={dims}")


if __name__ == "__main__":
    main()
