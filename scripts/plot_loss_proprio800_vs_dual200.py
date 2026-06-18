#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments'
OUT_DIR = EXP / 'loss_comparison'
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_A = EXP / 'phi0_act_proprio_800step_train.log'
LOG_B = EXP / 'phi0_act_dual_vggt_200step_train.log'
OUT_PNG = OUT_DIR / 'training_loss_proprio800_vs_dual200_y0_1.png'
OUT_PDF = OUT_DIR / 'training_loss_proprio800_vs_dual200_y0_1.pdf'

STEP_RE = re.compile(r"step=(\d+) loss=([\d.]+)(?:\s+\{(.+)\})?")
COMP_RE = re.compile(r"'(\w+)': ([\d.eE+-]+)")


def parse_log(path: Path):
    steps, total, action, bone = [], [], [], []
    for line in path.read_text(errors='replace').splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        comps = dict(COMP_RE.findall(m.group(3) or ''))
        steps.append(int(m.group(1)))
        total.append(float(m.group(2)))
        action.append(float(comps.get('loss_action', float('nan'))))
        bone.append(float(comps.get('loss_bone', float('nan'))))
    return steps, total, action, bone


def smooth(y, w: int = 15):
    arr = np.array(y, dtype=float)
    out = np.copy(arr)
    half = w // 2
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        out[i] = np.nanmean(arr[lo:hi])
    return out


def main():
    series = [
        ('ACT+proprio 800step',) + parse_log(LOG_A),
        ('ACT+proprio+VGGT 200step',) + parse_log(LOG_B),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ['#2ca02c', '#9467bd']
    for (label, steps, total, action, bone), c in zip(series, colors):
        axes[0, 0].plot(steps, total, alpha=0.15, color=c)
        axes[0, 0].plot(steps, smooth(total), label=label, color=c, linewidth=1.8)
        axes[0, 1].plot(steps, smooth(action), label=label, color=c, linewidth=1.6)
        axes[1, 0].plot(steps, smooth(bone), label=label, color=c, linewidth=1.6)
        axes[1, 1].plot(steps, smooth(total), label=label, color=c, linewidth=1.8)

    for ax, title in zip(axes.ravel(), ['Total loss', 'Action loss', 'Bone loss', 'Total overlay']):
        ax.set_title(title)
        ax.set_xlabel('step')
        ax.set_ylabel('loss')
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    fig.savefig(OUT_PDF)
    plt.close(fig)

    for label, steps, total, *_ in series:
        tail = total[-20:] if len(total) >= 20 else total
        print(
            f"{label}: start={steps[0]} end={steps[-1]} "
            f"end_total={total[-1]:.4f} min_total={min(total):.4f} "
            f"tail20_mean={float(np.mean(tail)):.4f} tail20_std={float(np.std(tail)):.4f}"
        )
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == '__main__':
    main()
