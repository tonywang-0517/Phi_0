#!/usr/bin/env python3
"""Visualize Phi_0 SMPL-H skeleton trajectories (GT + predicted keypoints).

Loads deploy JSONL (256-d d_raw) and optionally overlays GT from Xperience HDF5
keypoints (joint 0 is root quat xyz anchor — use as-is for consistent bones).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phi0.viz.skeleton import (  # noqa: E402
    apply_scene_limits,
    compute_scene_bounds,
    draw_skeleton,
    load_gt_from_hdf5,
    load_jsonl_predictions,
    subsample_frame_indices,
)
from phi0.schema.action_schema import get_action_schema, unpack_keypoints_52
from phi0.data.xperience import DEFAULT_HDF5
from phi0.viz.xperience_viz_frame import hdf5_keypoints_for_viz

PRED_COLOR = "royalblue"
PRED_ROOT_COLOR = "darkorange"
VIDEO_PANEL_SIZE = (480, 640)  # H, W — matches deploy resize


def resolve_viz_action_chunk_size(
    predictions_path: Path,
    *,
    override: int | None = None,
    default: int = 29,
) -> int:
    """Match deploy segment size (``inference_benchmark.json`` or default 29)."""
    if override is not None:
        return max(1, int(override))
    bench = predictions_path.parent / "inference_benchmark.json"
    if bench.is_file():
        try:
            data = json.loads(bench.read_text())
            if "action_chunk_size" in data:
                return max(1, int(data["action_chunk_size"]))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return max(1, int(default))


def deploy_model_context_native_indices(
    frames: list[dict],
    action_chunk_size: int,
) -> list[int]:
    """Native frame index of the last video-tower refresh before each control step."""
    chunk = max(1, int(action_chunk_size))
    out: list[int] = []
    for fi in range(len(frames)):
        seg_start = (fi // chunk) * chunk
        out.append(int(frames[seg_start].get("source_frame_index", seg_start)))
    return out


def _read_video_rgb_at_indices(
    source: Path,
    indices: list[int],
    panel_size: tuple[int, int] = VIDEO_PANEL_SIZE,
) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv required for video panel (pip install opencv-python)") from exc

    h, w = panel_size
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {source}")

    cache: dict[int, np.ndarray] = {}
    out: list[np.ndarray] = []
    try:
        for idx in indices:
            idx = int(idx)
            if idx not in cache:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, bgr = cap.read()
                if not ok:
                    print(f"Warning: failed to read video frame {idx} from {source}", file=sys.stderr)
                    cache[idx] = np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    cache[idx] = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            out.append(cache[idx])
    finally:
        cap.release()
    return np.stack(out, axis=0)


def load_deploy_video_frames(
    frames: list[dict],
    *,
    video_path: str | Path | None = None,
    panel_size: tuple[int, int] = VIDEO_PANEL_SIZE,
) -> tuple[np.ndarray | None, str | None]:
    """Load per-control-step RGB frames indexed by ``source_frame_index`` in JSONL."""
    if not frames:
        return None, None
    source = video_path or frames[0].get("source")
    if not source:
        return None, None
    source = Path(source)
    if not source.is_file():
        print(f"Warning: ego video not found: {source}", file=sys.stderr)
        return None, None

    indices = [
        int(fr.get("source_frame_index", fr.get("frame", i)))
        for i, fr in enumerate(frames)
    ]
    try:
        return _read_video_rgb_at_indices(source, indices, panel_size), str(source)
    except ImportError:
        print("Warning: opencv required for video panel (pip install opencv-python)", file=sys.stderr)
        return None, None
    except RuntimeError as exc:
        print(f"Warning: {exc}", file=sys.stderr)
        return None, None


def load_model_context_video_frames(
    frames: list[dict],
    action_chunk_size: int,
    *,
    video_path: str | Path | None = None,
    panel_size: tuple[int, int] = VIDEO_PANEL_SIZE,
) -> tuple[np.ndarray | None, list[int] | None]:
    """RGB at deploy video-refresh frame (one refresh per action chunk)."""
    if not frames:
        return None, None
    source = video_path or frames[0].get("source")
    if not source:
        return None, None
    source = Path(source)
    ctx_indices = deploy_model_context_native_indices(frames, action_chunk_size)
    try:
        return _read_video_rgb_at_indices(source, ctx_indices, panel_size), ctx_indices
    except (ImportError, RuntimeError) as exc:
        print(f"Warning: model context video unavailable: {exc}", file=sys.stderr)
        return None, None


def parse_args():
    p = argparse.ArgumentParser(description="Phi_0 GT + predicted SMPL-H skeleton viz")
    p.add_argument("--predictions", type=str, required=True, help="deploy smplh_out.jsonl")
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "experiments" / "viz"),
        help="Output directory (default: experiments/viz/)",
    )
    p.add_argument(
        "--hdf5",
        type=str,
        default=str(DEFAULT_HDF5) if DEFAULT_HDF5.is_file() else None,
        help="Xperience annotation.hdf5 for GT overlay (auto when predictions use xperience video)",
    )
    p.add_argument(
        "--skeleton-constants",
        type=str,
        default=None,
        help="Unused (kept for CLI compat with older runs)",
    )
    p.add_argument("--start-frame", type=int, default=0, help="HDF5 frame offset when aligning GT")
    p.add_argument("--max-frames", type=int, default=120, help="Cap frames for animation/plots")
    p.add_argument("--fps", type=int, default=15, help="Animation FPS")
    p.add_argument(
        "--format",
        choices=("gif", "mp4", "png_frames", "all"),
        default="all",
        help="Animation export format",
    )
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument(
        "--static-panels",
        type=int,
        default=4,
        help="Number of static snapshot panels (evenly spaced)",
    )
    p.add_argument(
        "--video",
        type=str,
        default=None,
        help="Override ego video path (default: ``source`` field in predictions JSONL)",
    )
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Disable ego-video panel in skeleton_animation",
    )
    p.add_argument(
        "--action-chunk-size",
        type=int,
        default=None,
        help="Deploy action chunk / video refresh period (default: inference_benchmark.json or 29)",
    )
    return p.parse_args()


def resolve_hdf5_path(hdf5_arg: str | None, frames: list[dict]) -> Path | None:
    if hdf5_arg:
        path = Path(hdf5_arg)
        if path.is_file():
            return path
        raise FileNotFoundError(f"HDF5 not found: {path}")
    for fr in frames:
        src = str(fr.get("source", "")).lower()
        if "xperience" in src or "stereo_left" in src:
            if DEFAULT_HDF5.is_file():
                return DEFAULT_HDF5
            break
    return None


def compute_scene_bounds_with_pred(
    gt_keypoints: np.ndarray | None,
    pred_keypoints: np.ndarray,
    gt_root: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Framing bounds from full pred skeleton (+ GT when available).

    Do not use root trajectory alone: joint 0 moves ~0.1 m while limbs span ~1 m,
    which clips bones and produces spiky \"star\" artifacts when GT is absent.
    """
    chunks: list[np.ndarray] = [pred_keypoints.reshape(-1, 3)]
    if gt_root is not None:
        chunks.append(gt_root.reshape(-1, 3))
    if gt_keypoints is not None:
        chunks.append(gt_keypoints.reshape(-1, 3))
    pts = np.concatenate(chunks, axis=0)
    center = pts.mean(axis=0)
    radius = float(np.max(np.linalg.norm(pts - center, axis=1))) * 1.15 + 1e-3
    return center, radius


def plot_static_snapshots(
    gt_keypoints: np.ndarray | None,
    pred_keypoints: np.ndarray,
    frame_indices: np.ndarray,
    out_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    n = len(frame_indices)
    fig = plt.figure(figsize=(4 * n, 4))
    center, radius = compute_scene_bounds_with_pred(
        gt_keypoints[frame_indices] if gt_keypoints is not None else None,
        pred_keypoints[frame_indices],
    )

    for i, fi in enumerate(frame_indices):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        if gt_keypoints is not None:
            draw_skeleton(ax, gt_keypoints[fi], color="darkgreen", alpha=0.95)
        draw_skeleton(ax, pred_keypoints[fi], color=PRED_COLOR, alpha=0.9, linewidth=1.4)
        ax.scatter(
            pred_keypoints[fi, 0, 0],
            pred_keypoints[fi, 0, 1],
            pred_keypoints[fi, 0, 2],
            c=PRED_ROOT_COLOR,
            s=28,
            depthshade=False,
        )
        apply_scene_limits(ax, center, radius)
        ax.set_title(f"frame {fi}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")

    title = "GT (green) + pred keypoints (blue)" if gt_keypoints is not None else "Pred keypoints skeleton"
    fig.suptitle(f"{title} (static snapshots)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_bone_length_histogram(
    gt_keypoints: np.ndarray,
    pred_keypoints: np.ndarray,
    out_path: Path,
    dpi: int,
) -> None:
    """Debug: compare bone lengths along SMPLH_PARENTS edges (GT vs pred FK)."""
    import matplotlib.pyplot as plt

    from phi0.viz.skeleton import iter_bone_segments

    gt_lens = []
    pred_lens = []
    labels = []
    for j, ((gp, gc), (pp, pc)) in enumerate(
        zip(iter_bone_segments(gt_keypoints[0]), iter_bone_segments(pred_keypoints[0]))
    ):
        gt_lens.append(float(np.linalg.norm(gc - gp)))
        pred_lens.append(float(np.linalg.norm(pc - pp)))
        labels.append(str(j))

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(x - width / 2, gt_lens, width, label="GT FK", color="darkgreen", alpha=0.85)
    ax.bar(x + width / 2, pred_lens, width, label="pred FK", color=PRED_COLOR, alpha=0.85)
    ax.set_ylabel("bone length (m)")
    ax.set_xlabel("bone index (SMPLH_PARENTS edge)")
    ax.set_title("Bone lengths — frame 0 (impossible long edges indicate FK/parent mismatch)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_root_trajectory(
    pred_root_trans: np.ndarray,
    gt_root: np.ndarray | None,
    out_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        pred_root_trans[:, 0],
        pred_root_trans[:, 1],
        pred_root_trans[:, 2],
        "b-o",
        markersize=3,
        label="pred root_trans (d_raw)",
    )
    ax.scatter(
        pred_root_trans[0, 0],
        pred_root_trans[0, 1],
        pred_root_trans[0, 2],
        c="cyan",
        s=40,
        label="pred start",
    )
    ax.scatter(
        pred_root_trans[-1, 0],
        pred_root_trans[-1, 1],
        pred_root_trans[-1, 2],
        c="blue",
        s=40,
        label="pred end",
    )
    if gt_root is not None:
        ax.plot(gt_root[:, 0], gt_root[:, 1], gt_root[:, 2], "r--", alpha=0.85, label="GT root_trans")
        ax.scatter(gt_root[0, 0], gt_root[0, 1], gt_root[0, 2], c="orange", s=40, label="GT start")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("World root translation (Ts_world_root[:3])")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _save_animation(anim, path: Path, fps: int) -> bool:
    """Save FuncAnimation to gif/mp4; return False if writer unavailable."""
    from matplotlib import animation

    try:
        if path.suffix == ".gif":
            writer = animation.PillowWriter(fps=fps)
        elif path.suffix == ".mp4":
            writer = animation.FFMpegWriter(fps=fps)
        else:
            return False
        anim.save(str(path), writer=writer)
        return path.is_file()
    except Exception as exc:
        print(f"Warning: could not save {path}: {exc}", file=sys.stderr)
        return False


def build_animation(
    gt_keypoints: np.ndarray | None,
    pred_keypoints: np.ndarray,
    out_dir: Path,
    fps: int,
    fmt: str,
    *,
    video_frames: np.ndarray | None = None,
    video_native_indices: list[int] | None = None,
    model_context_frames: np.ndarray | None = None,
    model_context_native_indices: list[int] | None = None,
    action_chunk_size: int | None = None,
) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib import animation

    n = len(pred_keypoints)
    center, radius = compute_scene_bounds_with_pred(gt_keypoints, pred_keypoints)

    has_ego = video_frames is not None and len(video_frames) >= n
    has_model_ctx = model_context_frames is not None and len(model_context_frames) >= n
    n_panels = 1 + int(has_ego) + int(has_model_ctx)

    if n_panels > 1:
        fig = plt.figure(figsize=(5 * n_panels, 5.5))
        col = 1
        ax_ego = None
        im_ego = None
        ax_model = None
        im_model = None
        if has_ego:
            ax_ego = fig.add_subplot(1, n_panels, col)
            ax_ego.axis("off")
            im_ego = ax_ego.imshow(video_frames[0])
            col += 1
        if has_model_ctx:
            ax_model = fig.add_subplot(1, n_panels, col)
            ax_model.axis("off")
            im_model = ax_model.imshow(model_context_frames[0])
            col += 1
        ax = fig.add_subplot(1, n_panels, col, projection="3d")
        title = fig.suptitle("")
    else:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax_ego = ax_model = None
        im_ego = im_model = None
        title = ax.set_title("")

    apply_scene_limits(ax, center, radius)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    gt_lines = []
    if gt_keypoints is not None:
        for _ in range(52):
            (line,) = ax.plot([], [], [], color="darkgreen", alpha=0.95, linewidth=1.2)
            gt_lines.append(line)

    pred_lines = []
    for _ in range(52):
        (line,) = ax.plot([], [], [], color=PRED_COLOR, alpha=0.9, linewidth=1.4)
        pred_lines.append(line)

    pred_root_scatter = ax.scatter([], [], [], c=PRED_ROOT_COLOR, s=36, depthshade=False)

    def _ego_native(fi: int) -> int:
        if video_native_indices and fi < len(video_native_indices):
            return int(video_native_indices[fi])
        return fi

    def _ctx_native(fi: int) -> int:
        if model_context_native_indices and fi < len(model_context_native_indices):
            return int(model_context_native_indices[fi])
        chunk = max(1, int(action_chunk_size or 29))
        return (fi // chunk) * chunk

    def _update_lines(lines, keypoints, fi: int) -> None:
        for line, (p, c) in zip(lines, _bone_pairs(keypoints[fi])):
            line.set_data([p[0], c[0]], [p[1], c[1]])
            line.set_3d_properties([p[2], c[2]])

    def _set_frame(fi: int) -> list:
        if gt_keypoints is not None:
            _update_lines(gt_lines, gt_keypoints, fi)
        _update_lines(pred_lines, pred_keypoints, fi)
        pred_root_scatter._offsets3d = (
            [pred_keypoints[fi, 0, 0]],
            [pred_keypoints[fi, 0, 1]],
            [pred_keypoints[fi, 0, 2]],
        )
        skel_title = f"control {fi} / {n - 1}  (green=GT, blue=pred)"
        if has_ego and ax_ego is not None and im_ego is not None:
            ax_ego.set_title(f"timeline ego @ native {_ego_native(fi)}", fontsize=9)
            im_ego.set_data(video_frames[fi])
        if has_model_ctx and ax_model is not None and im_model is not None:
            ctx_n = _ctx_native(fi)
            chunk = max(1, int(action_chunk_size or 29))
            seg = (fi // chunk) * chunk
            ax_model.set_title(
                f"model video refresh @ native {ctx_n}  (chunk {seg}-{min(seg + chunk - 1, n - 1)})",
                fontsize=9,
            )
            im_model.set_data(model_context_frames[fi])
        if n_panels > 1:
            title.set_text(skel_title)
        else:
            title.set_text(skel_title)
        artists: list = gt_lines + pred_lines + [pred_root_scatter, title]
        if im_ego is not None:
            artists.append(im_ego)
        if im_model is not None:
            artists.append(im_model)
        return artists

    def init():
        return _set_frame(0)

    def update(fi: int):
        return _set_frame(fi)

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n,
        interval=max(1, int(1000 / fps)),
        blit=False,
    )

    written: list[str] = []
    formats = ["gif", "mp4"] if fmt == "all" else ([fmt] if fmt != "png_frames" else [])
    if fmt in ("png_frames", "all"):
        frames_dir = out_dir / "skeleton_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for fi in range(n):
            _set_frame(fi)
            frame_path = frames_dir / f"frame_{fi:04d}.png"
            fig.savefig(frame_path, dpi=100, bbox_inches="tight")
        written.append(str(frames_dir.relative_to(out_dir)))

    for ext in formats:
        path = out_dir / f"skeleton_animation.{ext}"
        if _save_animation(anim, path, fps):
            written.append(path.name)

    plt.close(fig)
    return written


def build_sidebyside_animation(
    gt_keypoints: np.ndarray,
    pred_keypoints: np.ndarray,
    out_dir: Path,
    fps: int,
    fmt: str,
) -> list[str]:
    """Side-by-side GT (left) vs pred (right) skeleton animation."""
    import matplotlib.pyplot as plt
    from matplotlib import animation

    n = len(pred_keypoints)
    center, radius = compute_scene_bounds_with_pred(gt_keypoints, pred_keypoints)

    fig = plt.figure(figsize=(12, 5))
    ax_gt = fig.add_subplot(121, projection="3d")
    ax_pred = fig.add_subplot(122, projection="3d")
    for ax, label in ((ax_gt, "GT"), (ax_pred, "Pred")):
        apply_scene_limits(ax, center, radius)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(label)

    gt_lines = []
    for _ in range(52):
        (line,) = ax_gt.plot([], [], [], color="darkgreen", alpha=0.95, linewidth=1.2)
        gt_lines.append(line)

    pred_lines = []
    for _ in range(52):
        (line,) = ax_pred.plot([], [], [], color=PRED_COLOR, alpha=0.9, linewidth=1.4)
        pred_lines.append(line)

    pred_root_scatter = ax_pred.scatter([], [], [], c=PRED_ROOT_COLOR, s=36, depthshade=False)
    title = fig.suptitle("")

    def _update_lines(lines, keypoints, fi: int) -> None:
        for line, (p, c) in zip(lines, _bone_pairs(keypoints[fi])):
            line.set_data([p[0], c[0]], [p[1], c[1]])
            line.set_3d_properties([p[2], c[2]])

    def init():
        _update_lines(gt_lines, gt_keypoints, 0)
        _update_lines(pred_lines, pred_keypoints, 0)
        pred_root_scatter._offsets3d = (
            [pred_keypoints[0, 0, 0]],
            [pred_keypoints[0, 0, 1]],
            [pred_keypoints[0, 0, 2]],
        )
        title.set_text(f"GT vs Pred — frame 0 / {n - 1}")
        return gt_lines + pred_lines + [pred_root_scatter, title]

    def update(fi: int):
        _update_lines(gt_lines, gt_keypoints, fi)
        _update_lines(pred_lines, pred_keypoints, fi)
        pred_root_scatter._offsets3d = (
            [pred_keypoints[fi, 0, 0]],
            [pred_keypoints[fi, 0, 1]],
            [pred_keypoints[fi, 0, 2]],
        )
        title.set_text(f"GT vs Pred — frame {fi} / {n - 1}")
        return gt_lines + pred_lines + [pred_root_scatter, title]

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n,
        interval=max(1, int(1000 / fps)),
        blit=False,
    )

    written: list[str] = []
    formats = ["gif", "mp4"] if fmt == "all" else ([fmt] if fmt != "png_frames" else [])
    for ext in formats:
        path = out_dir / f"skeleton_gt_vs_pred_sidebyside.{ext}"
        if _save_animation(anim, path, fps):
            written.append(path.name)

    plt.close(fig)
    return written


def _bone_pairs(keypoints: np.ndarray):
    from phi0.viz.skeleton import iter_bone_segments

    return list(iter_bone_segments(keypoints))


def write_summary(
    out_dir: Path,
    d_raw: np.ndarray,
    pred_fk: np.ndarray | None,
    pred_keypoints: np.ndarray,
    frames: list[dict],
    gt: dict[str, np.ndarray] | None,
    written: list[str],
    constants_path: Path,
    *,
    pred_method: str,
    video_source: str | None = None,
    action_chunk_size: int | None = None,
) -> None:
    schema = get_action_schema()
    summary = {
        "n_frames": len(frames),
        "d_raw_shape": list(d_raw.shape),
        "action_rep": schema.rep,
        "gt_aligned": gt is not None,
        "pred_skeleton_method": pred_method,
        "skeleton_constants": str(constants_path),
        "outputs": written,
    }
    if pred_fk is not None:
        summary["pred_root_fk_vs_root_trans_mean_l2"] = float(
            np.mean(np.linalg.norm(pred_fk[:, 0, :] - d_raw[:, 0:3], axis=1))
        )
    if gt is not None:
        gt_kp_viz = gt["keypoints_viz"]
        summary["gt_skeleton_method"] = "HDF5 keypoints as-is (joint 0 = root quat xyz anchor)"
        summary["skeleton_l2_per_frame"] = np.linalg.norm(
            pred_keypoints - gt_kp_viz, axis=-1
        ).mean(axis=1).tolist()
        summary["skeleton_l2_mean"] = float(np.mean(summary["skeleton_l2_per_frame"]))
        from phi0.viz.skeleton import iter_bone_segments

        gt_bl = []
        pred_bl = []
        for (gp, gc), (pp, pc) in zip(
            iter_bone_segments(gt_kp_viz[0]),
            iter_bone_segments(pred_keypoints[0]),
        ):
            gt_bl.append(float(np.linalg.norm(gc - gp)))
            pred_bl.append(float(np.linalg.norm(pc - pp)))
        summary["bone_length_l1_mean_frame0"] = float(np.mean(np.abs(np.array(pred_bl) - np.array(gt_bl))))
        summary["bone_length_pred_max_frame0"] = float(max(pred_bl))
        summary["bone_length_gt_max_frame0"] = float(max(gt_bl))
    if video_source:
        summary["ego_video"] = video_source
        summary["animation_includes_ego_video"] = True
    if action_chunk_size is not None:
        summary["deploy_action_chunk_size"] = int(action_chunk_size)
        summary["animation_includes_model_video_context"] = True
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        raise SystemExit("matplotlib required: pip install matplotlib")

    pred_path = Path(args.predictions)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d_raw, frames = load_jsonl_predictions(pred_path)
    if args.max_frames and len(d_raw) > args.max_frames:
        d_raw = d_raw[: args.max_frames]
        frames = frames[: args.max_frames]

    pred_keypoints = unpack_keypoints_52(d_raw)
    pred_method = "direct keypoints_52 from d_raw[0:156]"

    start = int(frames[0].get("source_frame_index", args.start_frame))
    gt = None
    gt_keypoints = None
    gt_root = None
    hdf5_path = resolve_hdf5_path(args.hdf5, frames)
    if hdf5_path is not None:
        gt = load_gt_from_hdf5(hdf5_path, start, len(frames))
        gt_root = gt["root_trans"]
        gt_keypoints = hdf5_keypoints_for_viz(gt["keypoints_hdf5"])
        gt["keypoints_viz"] = gt_keypoints

    written: list[str] = []

    panel_idx = subsample_frame_indices(len(d_raw), max(1, args.static_panels))
    static_path = out_dir / "skeleton_static.png"
    plot_static_snapshots(gt_keypoints, pred_keypoints, panel_idx, static_path, args.dpi)
    written.append(static_path.name)

    traj_path = out_dir / "root_trajectory.png"
    pred_root = pred_keypoints[:, 0, :]
    plot_root_trajectory(pred_root, gt_root, traj_path, args.dpi)
    written.append(traj_path.name)

    bone_path = out_dir / "bone_length_gt_vs_pred.png"
    if gt_keypoints is not None:
        plot_bone_length_histogram(gt_keypoints, pred_keypoints, bone_path, args.dpi)
        written.append(bone_path.name)

    video_frames = None
    video_native_indices: list[int] | None = None
    model_context_frames = None
    model_context_indices: list[int] | None = None
    video_source: str | None = None
    action_chunk = resolve_viz_action_chunk_size(
        pred_path, override=args.action_chunk_size
    )
    if not args.no_video:
        video_frames, video_source = load_deploy_video_frames(
            frames, video_path=args.video
        )
        if video_frames is not None:
            video_native_indices = [
                int(fr.get("source_frame_index", fr.get("frame", i)))
                for i, fr in enumerate(frames)
            ]
            model_context_frames, model_context_indices = load_model_context_video_frames(
                frames,
                action_chunk,
                video_path=args.video or video_source,
            )

    anim_written = build_animation(
        gt_keypoints,
        pred_keypoints,
        out_dir,
        fps=args.fps,
        fmt=args.format,
        video_frames=video_frames,
        video_native_indices=video_native_indices,
        model_context_frames=model_context_frames,
        model_context_native_indices=model_context_indices,
        action_chunk_size=action_chunk,
    )
    written.extend(anim_written)

    if gt_keypoints is not None:
        sidebyside_written = build_sidebyside_animation(
            gt_keypoints,
            pred_keypoints,
            out_dir,
            fps=args.fps,
            fmt=args.format,
        )
        written.extend(sidebyside_written)

    np.save(out_dir / "pred_d_raw.npy", d_raw)
    np.save(out_dir / "pred_keypoints_viz.npy", pred_keypoints)
    if gt is not None:
        np.save(out_dir / "gt_d_raw.npy", gt["d_raw"])
        np.save(out_dir / "gt_keypoints_viz.npy", gt_keypoints)

    write_summary(
        out_dir,
        d_raw,
        None,
        pred_keypoints,
        frames,
        gt,
        written,
        Path("n/a"),
        pred_method=pred_method,
        video_source=video_source,
        action_chunk_size=action_chunk if not args.no_video else None,
    )
    print(f"Wrote skeleton viz to {out_dir}")
    for name in written:
        print(f"  - {out_dir / name}")


if __name__ == "__main__":
    main()
