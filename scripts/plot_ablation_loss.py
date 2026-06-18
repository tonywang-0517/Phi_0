#!/usr/bin/env python3
"""Plot 400-step ablation loss curves (total / action / bone) vs dual_200 baseline."""

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

SERIES = [
    ("dual_200 (baseline)", EXP / "phi0_act_dual_vggt_200step_train.log"),
    ("ab1 VGGT full", EXP / "ablation1_vggt_full_400step_train.log"),
    ("ab2 DiT4DiT query", EXP / "ablation2_dit4dit_query_400step_train.log"),
    ("ab3 both", EXP / "ablation3_both_400step_train.log"),
]

SERIES_704 = [
    ("baseline (linear+proprio)", EXP / "ablation_baseline_704_400step/train.log"),
    ("DiT4DiT prefix/query", EXP / "ablation_dit4dit_query_704_400step/train.log"),
]

OUT_PNG = OUT_DIR / "training_loss_ablation_400step_y0_1.png"
OUT_PDF = OUT_DIR / "training_loss_ablation_400step_y0_1.pdf"
OUT_704_PNG = OUT_DIR / "training_loss_ablation_704_400step.png"
OUT_704_PDF = OUT_DIR / "training_loss_ablation_704_400step.pdf"
OUT_704_ZOOM_PNG = OUT_DIR / "training_loss_ablation_704_400step_zoom.png"


def parse_log(path: Path) -> tuple[list[int], dict[str, list[float]]]:
    if not path.is_file():
        return [], {}
    steps: list[int] = []
    metrics: dict[str, list[float]] = {}
    for line in path.read_text(errors="replace").splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        comps = dict(COMP_RE.findall(m.group(3) or ""))
        steps.append(int(m.group(1)))
        metrics.setdefault("total", []).append(float(m.group(2)))
        for key, val in comps.items():
            metrics.setdefault(key, []).append(float(val))
    return steps, metrics


def legacy_parse_log(path: Path):
    steps, metrics = parse_log(path)
    return (
        steps,
        metrics.get("total", []),
        metrics.get("loss_action", []),
        metrics.get("loss_bone", []),
    )


def smooth(y, w: int = 15):
    arr = np.array(y, dtype=float)
    out = np.copy(arr)
    half = w // 2
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        out[i] = np.nanmean(arr[lo:hi])
    return out


def plot_series_grid(
    parsed: list[tuple[str, list[int], dict[str, list[float]]]],
    *,
    out_png: Path,
    out_pdf: Path | None = None,
    ylim: tuple[float, float] | None = None,
    title_suffix: str = "",
):
    metric_keys = [
        ("total", "Total loss"),
        ("loss_action", "Action loss"),
        ("loss_bone", "Bone loss"),
        ("loss_bone_hand", "Hand bone loss"),
        ("loss_hand_mse", "Hand MSE"),
    ]
    present = [mk for mk, _ in metric_keys if any(mk in metrics for _, _, metrics in parsed)]
    n = len(present)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for ax_idx, (key, title) in enumerate(metric_keys):
        if key not in present:
            continue
        ax = axes[ax_idx]
        for (label, steps, metrics), c in zip(parsed, colors):
            if key not in metrics:
                continue
            y = metrics[key]
            ax.plot(steps, y, alpha=0.12, color=c)
            ax.plot(steps, smooth(y), label=label, color=c, linewidth=1.8)
        ax.set_title(f"{title}{title_suffix}")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for j in range(len(present), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    if out_pdf is not None:
        fig.savefig(out_pdf)
    plt.close(fig)


def plot_704_ablation():
    parsed: list[tuple[str, list[int], dict[str, list[float]]]] = []
    for label, path in SERIES_704:
        steps, metrics = parse_log(path)
        if not steps:
            print(f"SKIP (no data): {label} -> {path}")
            continue
        parsed.append((label, steps, metrics))

    if len(parsed) < 2:
        raise SystemExit("Need both 704 ablation train.log files.")

    plot_series_grid(
        parsed,
        out_png=OUT_704_PNG,
        out_pdf=OUT_704_PDF,
        ylim=None,
        title_suffix=" (704×1280)",
    )
    plot_series_grid(
        parsed,
        out_png=OUT_704_ZOOM_PNG,
        out_pdf=None,
        ylim=(0.0, 1.5),
        title_suffix=" (704×1280, zoom)",
    )

    for label, steps, metrics in parsed:
        total = metrics["total"]
        tail = total[-20:] if len(total) >= 20 else total
        print(
            f"{label}: steps={steps[0]}..{steps[-1]} "
            f"end_total={total[-1]:.4f} min_total={min(total):.4f} "
            f"tail20_mean={float(np.mean(tail)):.4f} tail20_std={float(np.std(tail)):.4f}"
        )
    print(f"Saved: {OUT_704_PNG}")
    print(f"Saved: {OUT_704_PDF}")
    print(f"Saved: {OUT_704_ZOOM_PNG}")


def main():
    plot_704_ablation()

    parsed = []
    for label, path in SERIES:
        steps, total, action, bone = legacy_parse_log(path)
        if not steps:
            print(f"SKIP (no data): {label} -> {path}")
            continue
        parsed.append((label, steps, total, action, bone))

    if not parsed:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ["#9467bd", "#1f77b4", "#ff7f0e", "#d62728"]
    for (label, steps, total, action, bone), c in zip(parsed, colors[: len(parsed)]):
        axes[0, 0].plot(steps, total, alpha=0.15, color=c)
        axes[0, 0].plot(steps, smooth(total), label=label, color=c, linewidth=1.8)
        axes[0, 1].plot(steps, smooth(action), label=label, color=c, linewidth=1.6)
        axes[1, 0].plot(steps, smooth(bone), label=label, color=c, linewidth=1.6)
        axes[1, 1].plot(steps, smooth(total), label=label, color=c, linewidth=1.8)

    for ax, title in zip(
        axes.ravel(), ["Total loss", "Action loss", "Bone loss", "Total overlay"]
    ):
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    fig.savefig(OUT_PDF)
    plt.close(fig)

    for label, steps, total, *_ in parsed:
        tail = total[-20:] if len(total) >= 20 else total
        print(
            f"{label}: steps={steps[0]}..{steps[-1]} "
            f"end_total={total[-1]:.4f} min_total={min(total):.4f} "
            f"tail20_mean={float(np.mean(tail)):.4f} tail20_std={float(np.std(tail)):.4f}"
        )
    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
