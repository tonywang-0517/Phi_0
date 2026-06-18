#!/usr/bin/env python3
"""Plot action/bone loss: dual_vggt_800step (first 400) vs ablation2_dit4dit_query_400step."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT_DIR = EXP / "loss_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEP_RE = re.compile(r"step=(\d+) loss=([\d.]+)(?:\s+\{(.+)\})?")
COMP_RE = re.compile(r"'(\w+)': ([\d.eE+-]+)")

LOG_DUAL = EXP / "phi0_act_dual_vggt_800step_train.log"
LOG_AB2 = EXP / "ablation2_dit4dit_query_400step_train.log"
OUT_PNG = OUT_DIR / "training_loss_dual800_vs_ab2_400step_y0_0.1.png"
OUT_PDF = OUT_DIR / "training_loss_dual800_vs_ab2_400step_y0_0.1.pdf"
MAX_STEP = 400  # exclusive upper bound: steps 0..399


def parse_log(path: Path, max_step: int | None = MAX_STEP):
    steps, action, bone = [], [], []
    for line in path.read_text(errors="replace").splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        if max_step is not None and step >= max_step:
            continue
        comps = dict(COMP_RE.findall(m.group(3) or ""))
        steps.append(step)
        action.append(float(comps.get("loss_action", float("nan"))))
        bone.append(float(comps.get("loss_bone", float("nan"))))
    return steps, action, bone


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
        ("dual_vggt 800step (0–399)",) + parse_log(LOG_DUAL),
        ("ab2 DiT4DiT query 400step",) + parse_log(LOG_AB2),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#9467bd", "#ff7f0e"]

    for (label, steps, action, bone), c in zip(series, colors):
        axes[0].plot(steps, action, alpha=0.12, color=c)
        axes[0].plot(steps, smooth(action), label=label, color=c, linewidth=1.8)
        axes[1].plot(steps, bone, alpha=0.12, color=c)
        axes[1].plot(steps, smooth(bone), label=label, color=c, linewidth=1.8)

    axes[0].set_title("Action loss")
    axes[1].set_title("Bone loss")
    for ax in axes:
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_xlim(0, MAX_STEP - 1)
        ax.set_ylim(0.0, 0.1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("dual_vggt 800step vs ab2 — first 400 steps", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    for label, steps, action, bone in series:
        tail = slice(-20, None)
        print(
            f"{label}: n={len(steps)} "
            f"end_action={action[-1]:.4f} end_bone={bone[-1]:.4f} "
            f"tail20_action_mean={float(np.mean(action[tail])):.4f} "
            f"tail20_bone_mean={float(np.mean(bone[tail])):.4f}"
        )
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
