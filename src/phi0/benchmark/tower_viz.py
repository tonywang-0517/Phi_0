"""Visualize VGGT register tokens and related tower debug outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def _to_uint8_rgb(frame_chw: torch.Tensor) -> np.ndarray:
    x = frame_chw.detach().float().cpu()
    if x.min() < -0.01:
        x = (x.clamp(-1, 1) + 1.0) * 0.5
    x = (x.clamp(0, 1) * 255.0).byte()
    return x.permute(1, 2, 0).numpy()


def save_vggt_input_frame_grid(
    video_bcthw: torch.Tensor,
    path: Path,
    *,
    control_indices: Sequence[int] | None = None,
    max_control_t: int | None = None,
    image_resolution: int = 512,
) -> None:
    """Save VGGT-balanced input frames (what the tower actually sees)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from phi0.models.vggt.preprocess import video_to_vggt_input

    vggt_in = video_to_vggt_input(video_bcthw, image_resolution=image_resolution)[0]
    t = int(vggt_in.shape[0])
    cols = min(t, 6)
    rows = int(np.ceil(t / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.0))
    axes_flat = np.atleast_1d(axes).ravel()
    padded = set()
    if control_indices is not None and max_control_t is not None:
        padded = {i for i, c in enumerate(control_indices) if int(c) > int(max_control_t)}
    for i in range(t):
        ax = axes_flat[i]
        ax.imshow(_to_uint8_rgb(vggt_in[i]))
        title = f"t={i}"
        if control_indices is not None and i < len(control_indices):
            c = int(control_indices[i])
            title = f"ctrl={c}"
            if c > int(max_control_t or c):
                title += " (pad)"
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    for j in range(t, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle("VGGT input (balanced resize)", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_register_temporal_curve(
    vggt_ctx: torch.Tensor,
    path: Path,
    *,
    num_registers_per_frame: int = 16,
) -> None:
    """Per-frame mean register L2 norm (shows temporal motion vs padding duplicates)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ctx = vggt_ctx.detach().float().cpu()[0]
    norms = ctx.norm(dim=-1).numpy()
    r = int(num_registers_per_frame)
    grid = norms.reshape(-1, r)
    per_frame = grid.mean(axis=1)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(range(len(per_frame)), per_frame, marker="o")
    ax.set_xlabel("temporal subsample idx")
    ax.set_ylabel("mean register L2")
    ax.set_title("VGGT register energy per frame")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_vggt_register_heatmap(
    vggt_ctx: torch.Tensor,
    path: Path,
    *,
    num_registers_per_frame: int = 16,
    title: str = "VGGT scene registers (L2 norm)",
) -> None:
    """Save ``[B, S*R, D]`` register ctx as a frame×register heatmap PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ctx = vggt_ctx.detach().float().cpu()[0]
    norms = ctx.norm(dim=-1).numpy()
    r = int(num_registers_per_frame)
    if r <= 0 or norms.size % r != 0:
        raise ValueError(f"Cannot reshape {norms.size} norms with R={r}")
    grid = norms.reshape(-1, r)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, grid.shape[1] * 0.35), max(3, grid.shape[0] * 0.35)))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xlabel("register idx (not spatial H×W)")
    ax.set_ylabel("temporal subsample idx")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_vggt_register_pca_rgb(
    vggt_ctx: torch.Tensor,
    path: Path,
    *,
    num_registers_per_frame: int = 16,
) -> None:
    """PCA 2048-D registers -> RGB strip per frame (qualitative structure check)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ctx = vggt_ctx.detach().float().cpu()[0].numpy()
    r = int(num_registers_per_frame)
    t = ctx.shape[0] // r
    tokens = ctx.reshape(t, r, -1)

    flat = tokens.reshape(-1, tokens.shape[-1])
    flat = flat - flat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    comp3 = flat @ vt[:3].T
    comp3 = comp3.reshape(t, r, 3)
    lo = comp3.min(axis=(0, 1), keepdims=True)
    hi = comp3.max(axis=(0, 1), keepdims=True)
    rgb = (comp3 - lo) / np.clip(hi - lo, 1e-6, None)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(r * 0.5, t * 0.5))
    ax.imshow(np.clip(rgb, 0, 1), aspect="auto")
    ax.set_xlabel("register (global token, not pixel)")
    ax.set_ylabel("frame")
    ax.set_title("VGGT registers PCA→RGB")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_cosmos_cond_vs_pred_frame(
    cond_chw: torch.Tensor,
    pred_tchw: torch.Tensor,
    path: Path,
) -> float:
    """Side-by-side conditioning frame vs predict_video frame 0; return MSE in [0,1]."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cond = _to_uint8_rgb(cond_chw)
    pred0 = _to_uint8_rgb(pred_tchw[0])
    mse = float(
        np.mean(
            (cond.astype(np.float32) / 255.0 - pred0.astype(np.float32) / 255.0) ** 2
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(cond)
    axes[0].set_title("I2W cond (last clip frame)")
    axes[0].axis("off")
    axes[1].imshow(pred0)
    axes[1].set_title(f"predict frame 0 (MSE={mse:.4f})")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return mse
