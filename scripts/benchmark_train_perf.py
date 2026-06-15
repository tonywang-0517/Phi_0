#!/usr/bin/env python3
"""Compare training step latency across perf configurations."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda:2")
    p.add_argument("--output", type=str, default=str(ROOT / "experiments/benchmark_perf/compare.json"))
    return p.parse_args()


def run_train(label: str, extra_args: list[str], steps: int) -> dict:
    cmd = [
        "conda",
        "run",
        "-n",
        "Phi-0-wpy",
        "python",
        str(ROOT / "scripts/train.py"),
        f"max_steps={steps}",
        "batch_size=1",
        "save_every_steps=0",
        "checkpoint_overwrite=true",
        f"output_dir={ROOT / 'experiments' / 'benchmark_perf' / label}",
    ] + extra_args
    env = {
        **dict(subprocess.os.environ),
        "PYTHONPATH": f"{ROOT / 'src'}:/mnt/data2/wpy/workspace/FastWAM/src",
        "CUDA_VISIBLE_DEVICES": "2",
    }
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - t0
    log = proc.stdout + proc.stderr
    step_times = []
    for m in re.finditer(r"step=(\d+) loss=([\d.]+)", log):
        step_times.append({"step": int(m.group(1)), "loss": float(m.group(2))})
    ckpt_dir = ROOT / "experiments" / "benchmark_perf" / label
    ckpt_files = list(ckpt_dir.glob("*.pt")) if ckpt_dir.is_dir() else []
    ckpt_mb = sum(f.stat().st_size for f in ckpt_files) / (1024 * 1024) if ckpt_files else 0
    per_step = wall / max(steps, 1)
    return {
        "label": label,
        "exit_code": proc.returncode,
        "wall_s": round(wall, 2),
        "per_step_s": round(per_step, 3),
        "steps_logged": len(step_times),
        "checkpoint_mb": round(ckpt_mb, 1),
        "last_log_tail": log.strip().splitlines()[-3:] if log.strip() else [],
        "error_tail": log.strip().splitlines()[-8:] if proc.returncode != 0 else [],
    }


def extra_args_device(args: list[str]) -> str | None:
    for a in args:
        if a.startswith("device="):
            return a.split("=", 1)[1].replace("cuda", "").strip(":") or "0"
    return None


def main():
    args = parse_args()
    steps = int(args.steps)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    scenarios = [
        (
            "legacy_video_gc",
            [
                "model.loss.lambda_video=1.0",
                "model.enable_cosmos_gradient_checkpointing=true",
                "model.action_dit_config.use_gradient_checkpointing=true",
                "save_action_expert_only=true",
            ],
        ),
        (
            "optimized_action_only",
            [
                "model.loss.lambda_video=0.0",
                "model.enable_cosmos_gradient_checkpointing=false",
                "model.action_dit_config.use_gradient_checkpointing=false",
                "save_action_expert_only=true",
                "save_optimizer=false",
            ],
        ),
    ]

    results = []
    for label, extra in scenarios:
        print(f"\n=== Running {label} ({steps} steps) ===", flush=True)
        results.append(run_train(label, extra, steps))

    base = results[1]["per_step_s"] if len(results) > 1 and results[1]["exit_code"] == 0 else None
    legacy = results[0]["per_step_s"] if results[0]["exit_code"] == 0 else None
    summary = {
        "steps_per_run": steps,
        "runs": results,
        "speedup_vs_legacy": {
            "optimized_action_only": round(legacy / base, 2) if legacy and base else None,
        },
    }
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if any(r["exit_code"] != 0 for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
