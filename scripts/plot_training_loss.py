#!/usr/bin/env python3
"""Plot Phi_0 training loss curves from experiment train logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEP_RE = re.compile(r"step=(\d+) loss=([\d.]+)(?:\s+\{(.+)\})?")
COMP_RE = re.compile(r"'(\w+)': ([\d.eE+-]+)")


def parse_log(path: Path) -> tuple[list[int], list[float], list[float], list[float]]:
    steps, total, action, bone = [], [], [], []
    if not path.is_file():
        return steps, total, action, bone
    for line in path.read_text(errors="replace").splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        comps = dict(COMP_RE.findall(m.group(3) or ""))
        steps.append(int(m.group(1)))
        total.append(float(m.group(2)))
        action.append(float(comps.get("loss_action", float("nan"))))
        bone.append(float(comps.get("loss_bone", float("nan"))))
    return steps, total, action, bone


def parse_log_text(text: str) -> tuple[list[int], list[float], list[float], list[float]]:
    steps, total, action, bone = [], [], [], []
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        comps = dict(COMP_RE.findall(m.group(3) or ""))
        steps.append(int(m.group(1)))
        total.append(float(m.group(2)))
        action.append(float(comps.get("loss_action", float("nan"))))
        bone.append(float(comps.get("loss_bone", float("nan"))))
    return steps, total, action, bone


def extract_resume_section(train_log: Path, marker: str) -> str:
    lines = train_log.read_text(errors="replace").splitlines()
    start = next(i for i, line in enumerate(lines) if marker in line)
    end = next(
        i
        for i, line in enumerate(lines)
        if i > start and "Saved checkpoint (overwrite): experiments/phi0_act_proprio_800step" in line and "step=800" in line
    )
    return "\n".join(line for line in lines[start : end + 1] if STEP_RE.search(line))


def build_800step_log(experiments: Path, train_log: Path) -> Path:
    log_400 = experiments / "phi0_act_proprio_400step_train.log"
    resume_text = extract_resume_section(train_log, "Training steps 400 -> 800")
    s0, t0, a0, b0 = parse_log(log_400)
    s1, t1, a1, b1 = parse_log_text(resume_text)
    out = experiments / "phi0_act_proprio_800step_train.log"
    steps = s0 + s1
    totals = t0 + t1
    actions = a0 + a1
    bones = b0 + b1
    out.write_text(
        "\n".join(
            f"step={s} loss={t:.4f} {{'loss_action': {a}, 'loss_bone': {b}}}"
            for s, t, a, b in zip(steps, totals, actions, bones)
        )
    )
    return out


def smooth(y: list[float], w: int = 15) -> list[float]:
    arr = np.array(y, dtype=float)
    out = np.copy(arr)
    half = w // 2
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        out[i] = np.nanmean(arr[lo:hi])
    return out.tolist()


def default_experiments(experiments: Path, train_log: Path) -> list[tuple[str, Path]]:
    log_800 = build_800step_log(experiments, train_log)
    series = [
        ("FM 200step", experiments / "phi0_fm_200step_train.log"),
        ("ACT 200step", experiments / "phi0_act_200step_train.log"),
        ("ACT+proprio 200step", experiments / "phi0_act_proprio_200step_train.log"),
        ("ACT+proprio 400step", experiments / "phi0_act_proprio_400step_train.log"),
        ("ACT+proprio 800step", log_800),
    ]
    dual_log = experiments / "phi0_act_dual_vggt_800step_train.log"
    if dual_log.is_file():
        series.append(("ACT+proprio+VGGT 800step", dual_log))
    return series


def plot_losses(
    series: list[tuple[str, list[int], list[float], list[float], list[float]]],
    out_dir: Path,
    *,
    y_max: float = 1.0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    titles = ["Total loss", "Action loss", "Bone loss", "Total overlay"]
    for i, (label, steps, total, action, bone) in enumerate(series):
        c = colors[i % len(colors)]
        axes[0, 0].plot(steps, total, alpha=0.15, color=c)
        axes[0, 0].plot(steps, smooth(total), label=label, color=c, linewidth=1.8)
        axes[0, 1].plot(steps, smooth(action), label=label, color=c, linewidth=1.5)
        axes[1, 0].plot(steps, smooth(bone), label=label, color=c, linewidth=1.5)
        axes[1, 1].plot(steps, smooth(total), label=label, color=c, linewidth=1.5)

    for ax, title in zip(axes.ravel(), titles):
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_ylim(0.0, y_max)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "training_loss_comparison.png", dpi=150)
    fig.savefig(out_dir / "training_loss_comparison.pdf")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for i, (label, steps, total, *_rest) in enumerate(series):
        c = colors[i % len(colors)]
        ax2.plot(steps, total, alpha=0.12, color=c)
        ax2.plot(steps, smooth(total), label=label, color=c, linewidth=2)
    ax2.set_xlabel("step")
    ax2.set_ylabel("total loss")
    ax2.set_ylim(0.0, y_max)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title("Phi_0 training total loss comparison")
    fig2.tight_layout()
    fig2.savefig(out_dir / "training_loss_total.png", dpi=150)
    plt.close(fig2)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Phi_0 training loss comparison")
    p.add_argument("--experiments-dir", type=str, default=str(ROOT / "experiments"))
    p.add_argument("--train-log", type=str, default=str(ROOT / "train.log"))
    p.add_argument("--output-dir", type=str, default=str(ROOT / "experiments" / "loss_comparison"))
    p.add_argument("--y-max", type=float, default=1.0, help="Upper y-axis limit (default: 1.0)")
    args = p.parse_args()

    experiments = Path(args.experiments_dir)
    series = []
    for label, path in default_experiments(experiments, Path(args.train_log)):
        steps, total, action, bone = parse_log(path)
        if not steps:
            print(f"skip {label}: missing {path}")
            continue
        series.append((label, steps, total, action, bone))
        print(
            f"{label}: {steps[0]}-{steps[-1]}, end={total[-1]:.3f}, min={min(total):.3f}"
        )

    plot_losses(series, Path(args.output_dir), y_max=float(args.y_max))
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
