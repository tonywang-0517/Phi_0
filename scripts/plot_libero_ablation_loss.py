#!/usr/bin/env python3
"""Plot loss_action moving average for LIBERO spatial ablation train.log files."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEP_RE = re.compile(
    r"step=(\d+) loss=([\d.eE+-]+)(?:\s+\{(?P<extra>.+)\})?"
)

DEFAULT_LOGS: dict[str, str] = {
    "vlm_only 35k lr=1e-4 (DDP x4)": "experiments/libero_spatial_vlm_only_35k_ddp4/train.log",
    "act3 0.5x blocks 35k lr=1e-4 (DDP x4)": "experiments/libero_spatial_vlm_act3_35k_ddp4/train.log",
    "act12 2x blocks 35k lr=5e-5 (DDP x4)": "experiments/libero_spatial_vlm_act12_35k_ddp4/train.log",
    "wrist 35k lr=1e-4 (DDP x4)": "experiments/libero_spatial_vlm_wrist_35k_ddp4/train.log",
    "wrist 35k bs32 (DDP x8)": "experiments/libero_spatial_vlm_wrist_35k_ddp8_bs32/train.log",
    "dual 35k lr=1e-4 (DDP x4)": "experiments/libero_spatial_vlm_dual_35k_ddp4/train.log",
    "vlm_only bs128 15k lr=1e-4 (ref)": "experiments/libero_spatial_vlm_only_15k_single_bs128/train.log",
}


def parse_log(path: Path) -> tuple[list[int], np.ndarray]:
    seen: dict[int, float] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        loss = float(m.group(2))
        extra = m.group("extra")
        if extra:
            try:
                payload = ast.literal_eval("{" + extra + "}")
                if isinstance(payload, dict) and "loss_action" in payload:
                    loss = float(payload["loss_action"])
            except (SyntaxError, ValueError):
                pass
        seen[int(m.group(1))] = loss
    steps = sorted(seen)
    return steps, np.array([seen[s] for s in steps], dtype=np.float64)


DEFAULT_SMOOTH_WIN = 200


def smooth(steps: list[int], losses: np.ndarray, win: int = DEFAULT_SMOOTH_WIN) -> tuple[np.ndarray, np.ndarray]:
    win = min(win, max(1, len(losses)))
    kernel = np.ones(win, dtype=np.float64) / win
    sm = np.convolve(losses, kernel, mode="valid")
    ss = np.array(steps[win - 1 :], dtype=np.int64)
    return ss, sm


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LIBERO spatial ablation loss curves.")
    parser.add_argument(
        "--smooth-win",
        type=int,
        default=DEFAULT_SMOOTH_WIN,
        help=f"moving-average window (default: {DEFAULT_SMOOTH_WIN})",
    )
    args = parser.parse_args()
    smooth_win = max(1, int(args.smooth_win))

    out = ROOT / "experiments" / "libero_spatial_vlm_ablation_loss.png"
    colors = ["C0", "C5", "C4", "C3", "C2", "C1"]

    fig, ax = plt.subplots(figsize=(11, 5))
    plotted = 0
    for i, (label, rel_path) in enumerate(DEFAULT_LOGS.items()):
        path = ROOT / rel_path
        if not path.is_file():
            print(f"skip {label}: missing {path}")
            continue
        steps, losses = parse_log(path)
        if len(steps) == 0:
            print(f"skip {label}: no step= lines in {path}")
            continue
        ss, sm = smooth(steps, losses, win=smooth_win)
        c = colors[i % len(colors)]
        ax.plot(steps, losses, color=c, alpha=0.04, linewidth=0.3)
        ax.plot(ss, sm, color=c, linewidth=1.2, alpha=0.72, label=f"{label} MA({smooth_win})")
        print(f"{label}: step {steps[0]}..{steps[-1]}  MA {sm[0]:.4f}->{sm[-1]:.4f}")
        plotted += 1

    if plotted == 0:
        raise SystemExit("No curves plotted; check train.log paths.")

    ax.set(xlabel="step", ylabel="loss_action", title=f"LIBERO spatial ablation (MA{smooth_win})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
