#!/usr/bin/env python3
"""Plot smoothed loss_action from Phi_0 train.log."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STEP_RE = re.compile(
    r"step=(\d+) loss=([\d.]+)(?:\s+\{(?P<extra>.+)\})?"
)


def parse_train_log(path: Path) -> tuple[list[int], list[float]]:
    seen: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        action_loss = float(m.group(2))
        extra = m.group("extra")
        if extra:
            try:
                d = ast.literal_eval("{" + extra + "}")
                if "loss_action" in d:
                    action_loss = float(d["loss_action"])
            except (SyntaxError, ValueError):
                pass
        seen[step] = action_loss

    steps = sorted(seen)
    return steps, [seen[s] for s in steps]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: same dir as log, loss_curve.png)",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        metavar=("YMIN", "YMAX"),
        default=(0.0, 0.08),
        help="Y-axis limits (default: 0 0.08)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=0,
        help="Moving-average window (0 = auto from log length)",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Overlay faint raw per-step loss_action",
    )
    args = parser.parse_args()
    log_path = args.log_path.resolve()
    out_path = args.output or (log_path.parent / "loss_curve.png")

    steps, action_losses = parse_train_log(log_path)
    if not steps:
        raise SystemExit(f"No step= lines found in {log_path}")

    win = int(args.smooth_window)
    if win <= 0:
        win = min(50, max(5, len(action_losses) // 50))
    arr = np.asarray(action_losses, dtype=np.float64)
    smooth = np.convolve(arr, np.ones(win) / win, mode="valid")
    smooth_steps = steps[win - 1 :]

    ymin, ymax = args.ylim
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if args.show_raw:
        ax.plot(steps, action_losses, color="C1", linewidth=0.6, alpha=0.25, label="raw")
    ax.plot(smooth_steps, smooth, color="C3", linewidth=2.0, label=f"MA({win})")
    ax.set_xlabel("step")
    ax.set_ylabel("loss_action")
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"loss_action — {log_path.name} (step {steps[0]}–{steps[-1]})")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path} ({len(steps)} points, MA window={win}, y={ymin}–{ymax})")


if __name__ == "__main__":
    main()
