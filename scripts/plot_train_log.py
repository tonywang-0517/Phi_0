#!/usr/bin/env python3
"""Parse Phi_0 train.log and plot loss curves."""

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


def parse_train_log(path: Path) -> tuple[list[int], list[float], list[float]]:
    steps: list[int] = []
    losses: list[float] = []
    action_losses: list[float] = []
    seen: dict[int, float] = {}

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        loss = float(m.group(2))
        action_loss = loss
        extra = m.group("extra")
        if extra:
            try:
                d = ast.literal_eval("{" + extra + "}")
                if "loss_action" in d:
                    action_loss = float(d["loss_action"])
            except (SyntaxError, ValueError):
                pass
        # Keep last entry per step (log may contain restarts).
        seen[step] = (loss, action_loss)

    for step in sorted(seen):
        loss, action_loss = seen[step]
        steps.append(step)
        losses.append(loss)
        action_losses.append(action_loss)
    return steps, losses, action_losses


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
        "--action-ylim",
        type=float,
        nargs=2,
        metavar=("YMIN", "YMAX"),
        default=(0.0, 0.1),
        help="Y-axis limits for action-loss zoom panel (default: 0 0.1)",
    )
    parser.add_argument(
        "--smooth-only",
        action="store_true",
        help="Plot smoothed loss_action only (no raw per-step line)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=0,
        help="Moving-average window (0 = auto from log length)",
    )
    parser.add_argument(
        "--single-panel",
        action="store_true",
        help="Single action-loss panel only (no log-scale total loss)",
    )
    args = parser.parse_args()
    log_path = args.log_path.resolve()
    out_path = args.output or (log_path.parent / "loss_curve.png")

    steps, losses, action_losses = parse_train_log(log_path)
    if not steps:
        raise SystemExit(f"No step= lines found in {log_path}")

    win = int(args.smooth_window)
    if win <= 0:
        win = min(50, max(5, len(action_losses) // 50))
    arr = np.asarray(action_losses, dtype=np.float64)
    kernel = np.ones(win) / win
    smooth = np.convolve(arr, kernel, mode="valid")
    smooth_steps = steps[win - 1 :]

    if args.single_panel:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
        axes_action = [ax]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(steps, losses, linewidth=0.8, alpha=0.7, label="total loss")
        axes[0].set_ylabel("loss (log)")
        axes[0].set_yscale("log")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_title(f"Training loss — {log_path.name} (n={len(steps)} steps)")
        axes_action = [axes[1]]

    ymin, ymax = args.action_ylim
    ax = axes_action[0]
    if not args.smooth_only:
        ax.plot(steps, action_losses, color="C1", linewidth=0.6, label="loss_action", alpha=0.35)
    ax.plot(
        smooth_steps,
        smooth,
        color="C3",
        linewidth=2.0,
        label=f"loss_action MA({win})",
    )
    ax.set_xlabel("step")
    ax.set_ylabel(f"loss_action ({ymin}–{ymax})")
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)
    ax.legend()
    if args.single_panel:
        ax.set_title(
            f"loss_action (smoothed) — {log_path.name} "
            f"(step {steps[0]}–{steps[-1]}, n={len(steps)})"
        )

    if not args.single_panel:
        # Mark batch-size change region (steps 0-199 batch=2, 200+ batch=4) if present.
        if max(steps) >= 200:
            for ax in axes:
                ax.axvline(200, color="gray", linestyle="--", alpha=0.5, linewidth=1)
            axes[0].text(200, axes[0].get_ylim()[1], " batch 2→4", fontsize=8, va="top")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path} ({len(steps)} points, step {steps[0]}–{steps[-1]})")


if __name__ == "__main__":
    main()
