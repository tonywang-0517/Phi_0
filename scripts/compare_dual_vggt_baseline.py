#!/usr/bin/env python3
"""Compare dual VGGT vs ACT+proprio baseline experiment metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def fm_summary(report: dict) -> dict:
    clips = report.get("fm_chunk_eval", {}).get("clips", [])
    if not clips:
        return {}
    mses = [c["masked_mse"] for c in clips if "masked_mse" in c]
    return {
        "n_clips": len(mses),
        "masked_mse_mean": sum(mses) / len(mses),
        "masked_mse_min": min(mses),
        "masked_mse_max": max(mses),
    }


def deploy_summary(report: dict) -> dict:
    deploy = report.get("deploy_eval", {})
    if not deploy:
        return {}
    out = {}
    for key in ("latency_ms_mean", "latency_ms_p95", "throughput_fps"):
        if key in deploy:
            out[key] = deploy[key]
    if "skeleton_l2_mean" in deploy:
        out["skeleton_l2_mean"] = deploy["skeleton_l2_mean"]
    return out


def summarize(name: str, exp_dir: Path) -> dict:
    eval_report = load_json(exp_dir / "eval_report_5s.json")
    viz_summary = load_json(exp_dir / "viz_skeleton_5s" / "summary.json")
    ckpt = exp_dir / f"{exp_dir.name}_latest.pt"
    return {
        "name": name,
        "dir": str(exp_dir),
        "checkpoint_exists": ckpt.is_file(),
        "checkpoint_step": load_json(exp_dir / "train_meta.json").get("step") if (exp_dir / "train_meta.json").is_file() else None,
        "fm_chunk": fm_summary(eval_report),
        "deploy_eval": deploy_summary(eval_report),
        "viz": {
            "skeleton_l2_mean": viz_summary.get("skeleton_l2_mean"),
            "gt_aligned": viz_summary.get("gt_aligned"),
            "n_frames": viz_summary.get("n_frames"),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, default=ROOT / "experiments/phi0_act_proprio_800step")
    p.add_argument("--dual", type=Path, default=ROOT / "experiments/phi0_act_dual_vggt_800step")
    p.add_argument("--output", type=Path, default=ROOT / "experiments/loss_comparison/eval_comparison_800step.json")
    args = p.parse_args()

    baseline = summarize("ACT+proprio 800step", args.baseline)
    dual = summarize("ACT+proprio+VGGT 800step", args.dual)

    def delta(a, b):
        if a is None or b is None:
            return None
        return b - a

    comparison = {
        "baseline": baseline,
        "dual_vggt": dual,
        "delta_dual_minus_baseline": {
            "viz_skeleton_l2_mean": delta(
                baseline["viz"].get("skeleton_l2_mean"),
                dual["viz"].get("skeleton_l2_mean"),
            ),
            "fm_masked_mse_mean": delta(
                baseline["fm_chunk"].get("masked_mse_mean"),
                dual["fm_chunk"].get("masked_mse_mean"),
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
