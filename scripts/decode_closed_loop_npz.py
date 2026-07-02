#!/usr/bin/env python3
"""Decode closed-loop observations.npz + outputs.npz into viewable media."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

logger = logging.getLogger(__name__)


def _write_rgb_mp4(frames: np.ndarray, path: Path, *, fps: float) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=float(fps))
    for frame in frames:
        writer.append_data(np.asarray(frame, dtype=np.uint8))
    writer.close()


def _annotate_frame(
    img: np.ndarray,
    *,
    title: str,
    control_idx: int,
    inference_i: int,
) -> np.ndarray:
    import cv2

    out = np.asarray(img, dtype=np.uint8).copy()
    cv2.putText(
        out,
        f"{title}  infer={inference_i}  ctrl={control_idx}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return out


def decode_observations(
    obs_path: Path,
    out_dir: Path,
    *,
    fps: float = 2.0,
    hold_s: float = 0.5,
) -> dict:
    data = np.load(obs_path)
    ego = np.asarray(data["ego"], dtype=np.uint8)
    control_idx = np.asarray(data["control_idx"], dtype=np.int32)
    timestamps = np.asarray(data["timestamp"], dtype=np.float64)
    n = int(data["num_inferences"]) if "num_inferences" in data.files else int(ego.shape[0])
    wrist = np.asarray(data["wrist"], dtype=np.uint8) if "wrist" in data.files else None
    inference_elapsed_s = (
        np.asarray(data["inference_elapsed_s"], dtype=np.float64)
        if "inference_elapsed_s" in data.files
        else None
    )

    hold_frames = max(1, int(round(float(hold_s) * float(fps))))
    ego_frames: list[np.ndarray] = []
    wrist_frames: list[np.ndarray] = []
    pair_frames: list[np.ndarray] = []
    for i in range(n):
        ego_ann = _annotate_frame(
            ego[i], title="ego", control_idx=int(control_idx[i]), inference_i=i
        )
        for _ in range(hold_frames):
            ego_frames.append(ego_ann)
        if wrist is not None:
            wrist_ann = _annotate_frame(
                wrist[i], title="wrist", control_idx=int(control_idx[i]), inference_i=i
            )
            for _ in range(hold_frames):
                wrist_frames.append(wrist_ann)
                pair = np.concatenate([ego_ann, wrist_ann], axis=1)
                pair_frames.append(pair)

    out_dir.mkdir(parents=True, exist_ok=True)
    ego_mp4 = out_dir / "observations_ego.mp4"
    _write_rgb_mp4(np.stack(ego_frames, axis=0), ego_mp4, fps=fps)
    written = {"ego_mp4": str(ego_mp4), "num_inferences": n}
    if wrist_frames:
        wrist_mp4 = out_dir / "observations_wrist.mp4"
        pair_mp4 = out_dir / "observations_ego_wrist.mp4"
        _write_rgb_mp4(np.stack(wrist_frames, axis=0), wrist_mp4, fps=fps)
        _write_rgb_mp4(np.stack(pair_frames, axis=0), pair_mp4, fps=fps)
        written["wrist_mp4"] = str(wrist_mp4)
        written["pair_mp4"] = str(pair_mp4)

    summary = {
        "control_idx": control_idx.tolist(),
        "timestamp": timestamps.tolist(),
        "duration_s": float(timestamps[-1] - timestamps[0]) if n > 1 else 0.0,
        **written,
    }
    if inference_elapsed_s is not None and inference_elapsed_s.size:
        summary["inference_elapsed_s"] = inference_elapsed_s.tolist()
        summary["inference_elapsed_mean_s"] = float(np.mean(inference_elapsed_s))
        summary["inference_elapsed_p95_s"] = float(np.percentile(inference_elapsed_s, 95))
        summary["inference_elapsed_max_s"] = float(np.max(inference_elapsed_s))
    (out_dir / "observations_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _render_output_panel(
    height: int,
    width: int,
    frame_idx: int,
    num_frames: int,
    token: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    control_idx: int | None = None,
    chunk_idx: int | None = None,
) -> np.ndarray:
    import cv2

    panel = np.zeros((height, width, 3), dtype=np.uint8)
    y = 28
    lines = [
        "closed-loop outputs",
        f"frame {frame_idx + 1}/{num_frames}",
    ]
    if control_idx is not None:
        lines.append(f"control_idx={control_idx}")
    if chunk_idx is not None:
        lines.append(f"chunk_idx={chunk_idx}")
    lines.extend(
        [
            f"token dim={token.size}",
            f"  [0]={token[0]:+.3f}  [1]={token[1]:+.3f}  [2]={token[2]:+.3f}",
            f"  [3]={token[3]:+.3f}  [4]={token[4]:+.3f}",
        ]
    )
    for text in lines:
        cv2.putText(
            panel,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        y += 22

    def _hand_bars(label: str, vals: np.ndarray, y0: int) -> int:
        cv2.putText(
            panel,
            label,
            (12, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (120, 200, 255),
            1,
            cv2.LINE_AA,
        )
        y0 += 18
        vals = np.asarray(vals, dtype=np.float64).reshape(-1)
        bar_w = max(width - 40, 80)
        for j, v in enumerate(vals[:7]):
            frac = float(np.clip(abs(v), 0.0, 1.0))
            x0, x1 = 20, 20 + int(bar_w * frac)
            cv2.rectangle(panel, (x0, y0), (x1, y0 + 10), (80, 180, 255), -1)
            cv2.putText(
                panel,
                f"{j}:{v:+.2f}",
                (20 + bar_w + 6, y0 + 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            y0 += 16
        return y0 + 6

    y = _hand_bars("L hand", left, y)
    _hand_bars("R hand", right, y)
    return panel


def decode_outputs(
    output_path: Path,
    out_dir: Path,
    *,
    control_fps: float = 50.0,
    panel_size: tuple[int, int] = (480, 640),
) -> dict:
    data = np.load(output_path)
    tokens = np.asarray(data["tokens"], dtype=np.float32)
    left = np.asarray(data["left"], dtype=np.float32)
    right = np.asarray(data["right"], dtype=np.float32)
    n = int(data["num_frames"]) if "num_frames" in data.files else int(tokens.shape[0])
    chunk_idx = np.asarray(data["chunk_idx"], dtype=np.int32) if "chunk_idx" in data.files else None
    control_idx = (
        np.asarray(data["control_idx"], dtype=np.int32) if "control_idx" in data.files else None
    )
    hand_ramp = (
        np.asarray(data["hand_ramp"], dtype=np.float32) if "hand_ramp" in data.files else None
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "outputs_arrays.npz",
        tokens=tokens,
        left=left,
        right=right,
        chunk_idx=chunk_idx,
        hand_ramp=hand_ramp,
    )

    h, w = int(panel_size[0]), int(panel_size[1])
    plan_frames = np.stack(
        [
            _render_output_panel(
                h,
                w,
                i,
                n,
                tokens[i],
                left[i],
                right[i],
                control_idx=None if control_idx is None else int(control_idx[i]),
                chunk_idx=None if chunk_idx is None else int(chunk_idx[i]),
            )
            for i in range(n)
        ],
        axis=0,
    )
    plan_mp4 = out_dir / "outputs_plan.mp4"
    _write_rgb_mp4(plan_frames, plan_mp4, fps=control_fps)

    try:
        import matplotlib.pyplot as plt

        t = np.arange(n, dtype=np.float64) / float(control_fps)
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(t, tokens[:, 0], label="token[0]")
        axes[0].plot(t, tokens[:, 1], label="token[1]")
        axes[0].set_ylabel("tokens")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        for j in range(left.shape[1]):
            axes[1].plot(t, left[:, j], alpha=0.8, label=f"L{j}")
        axes[1].set_ylabel("left hand")
        axes[1].grid(True, alpha=0.3)

        for j in range(right.shape[1]):
            axes[2].plot(t, right[:, j], alpha=0.8, label=f"R{j}")
        axes[2].set_ylabel("right hand")
        axes[2].set_xlabel("time (s)")
        axes[2].grid(True, alpha=0.3)

        if chunk_idx is not None:
            for ax in axes:
                changes = np.where(np.diff(chunk_idx) != 0)[0] + 1
                for c in changes:
                    ax.axvline(t[c], color="red", alpha=0.15, linewidth=1)

        fig.suptitle(f"closed-loop outputs ({n} frames @ {control_fps:.0f} Hz)")
        fig.tight_layout()
        plot_path = out_dir / "outputs_trajectory.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
    except ImportError:
        plot_path = None
        logger.warning("matplotlib not installed; skipped outputs_trajectory.png")

    summary = {
        "num_frames": n,
        "duration_s": round(n / float(control_fps), 3),
        "token_shape": list(tokens.shape),
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "token0_range": [float(tokens[:, 0].min()), float(tokens[:, 0].max())],
        "plan_mp4": str(plan_mp4),
        "plot": str(plot_path) if plot_path is not None else None,
    }
    (out_dir / "outputs_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


@dataclass
class DecodeConfig:
    record_dir: Path
    out_dir: Path | None = None
    control_fps: float = 50.0
    obs_fps: float = 2.0
    obs_hold_s: float = 0.5


def main(config: DecodeConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    record_dir = config.record_dir.expanduser().resolve()
    out_dir = (config.out_dir or record_dir / "decoded").expanduser().resolve()

    obs_path = record_dir / "observations.npz"
    output_path = record_dir / "outputs.npz"
    meta_path = record_dir / "record_meta.json"
    if meta_path.is_file():
        logger.info("record_meta: %s", json.loads(meta_path.read_text(encoding="utf-8")))

    if obs_path.is_file():
        obs_summary = decode_observations(
            obs_path, out_dir, fps=config.obs_fps, hold_s=config.obs_hold_s
        )
        logger.info("observations: %s", obs_summary)
    else:
        logger.warning("missing %s", obs_path)

    if output_path.is_file():
        out_summary = decode_outputs(output_path, out_dir, control_fps=config.control_fps)
        logger.info("outputs: %s", out_summary)
        logger.info(
            "robot sim replay: bash scripts/run_closed_loop_outputs_sim_replay.sh %s",
            output_path,
        )
    else:
        logger.warning("missing %s", output_path)

    logger.info("decoded media -> %s", out_dir)


if __name__ == "__main__":
    main(tyro.cli(DecodeConfig))
