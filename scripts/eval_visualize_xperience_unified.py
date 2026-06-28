#!/usr/bin/env python3
"""Evaluate + FK-visualize Phi_0 Xperience unified (512-d) action checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from phi0.checkpoint_utils import merge_saved_cfg
from phi0.data.cosmos_video_size import cosmos_video_size_from_cfg, round_hw_to_multiple
from phi0.data.processor import Phi0Processor
from phi0.data.sequence import SequenceDataset, sequence_dataset_from_cfg
from phi0.data.xperience import DEFAULT_HDF5
from phi0.data.xperience_unified_gt import (
    read_xperience_root_trans_world,
    reference_joints_world_from_hdf5_quat,
)
from phi0.inference.session import ClipInputsCache, PromptEmbedCache
from phi0.runtime import (
    activate_cuda_device,
    apply_processor_stats_from_checkpoint,
    build_base_dataset,
    build_processor,
    create_phi0,
    resolve_inference_device,
    sync_model_action_norm,
)
from phi0.schema.unified_action_schema import (
    get_unified_action_schema,
    joints_world_52_from_unified,
)
from phi0.viz.skeleton import (
    apply_scene_limits,
    configure_mpl3d_skeleton_axes,
    draw_ground_plane,
    draw_skeleton,
)
from phi0.viz.smplh_fk import load_skeleton_constants
from phi0.viz.xperience_viz_frame import (
    align_fk_joints_to_keypoints_frame,
    compute_keypoints_viz_bounds,
    hdf5_keypoints_for_viz,
)

# Reuse masked MSE from eval_action.py
from eval_action import compute_masked_mse, fm_chunk_eval

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Eval + visualize Xperience unified checkpoint")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "experiments/xperience_unified_act_1k_ddp4/xperience_unified_act_latest.pt"),
    )
    p.add_argument("--config-dir", type=str, default=str(ROOT / "configs"))
    p.add_argument("--config-name", type=str, default="train_xperience_unified")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--min-free-gb", type=float, default=12.0)
    p.add_argument("--max-clips", type=int, default=16, help="Clips for numeric MSE eval")
    p.add_argument(
        "--viz-clips",
        type=int,
        nargs="*",
        default=[0, 8, 16],
        help="Clip indices to render skeleton panels",
    )
    p.add_argument(
        "--viz-stride",
        type=int,
        default=2,
        help="Subsample future horizon frames in visualization",
    )
    p.add_argument("--hdf5", type=str, default=str(DEFAULT_HDF5))
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Default: <checkpoint_parent>/eval_viz",
    )
    p.add_argument("--make-gif", action="store_true", default=True)
    p.add_argument("--no-gif", action="store_false", dest="make_gif")
    return p.parse_args()


def unified_component_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    action_is_pad: Optional[torch.Tensor],
    action_dim_is_pad: Optional[torch.Tensor],
) -> Dict[str, float]:
    schema = get_unified_action_schema()
    out: Dict[str, float] = {}
    for name, (s, e) in schema.slices.items():
        dim_pad = None
        if action_dim_is_pad is not None:
            dim_pad = action_dim_is_pad[..., s:e]
            if dim_pad.ndim == 1:
                dim_pad = dim_pad.unsqueeze(0).unsqueeze(0)
            if dim_pad.all():
                out[name] = float("nan")
                continue
        out[name] = float(
            compute_masked_mse(
                pred[..., s:e],
                target[..., s:e],
                action_is_pad,
                dim_pad,
            ).item()
        )
    return out


def control_index_to_native(native_start: int, control_idx: int, *, native_fps: float, control_fps: float) -> int:
    return int(native_start) + int(round(float(control_idx) * float(native_fps) / float(control_fps)))


def fk_world_joints(
    action: np.ndarray,
    *,
    state_root_trans_world: np.ndarray,
    betas: np.ndarray,
    constants: dict[str, np.ndarray],
) -> np.ndarray:
    """52 joint positions in SMPL-H world frame (same as visualize_xperience_unified_gt.py)."""
    return joints_world_52_from_unified(
        action,
        state_root_trans_world=state_root_trans_world,
        betas=betas,
        constants=constants,
    )


def joint_l2_mean(pred_j: np.ndarray, gt_j: np.ndarray) -> float:
    return float(np.linalg.norm(pred_j - gt_j, axis=-1).mean())


def joint_l2_max(pred_j: np.ndarray, gt_j: np.ndarray) -> float:
    return float(np.linalg.norm(pred_j - gt_j, axis=-1).max())


@torch.no_grad()
def render_clip_skeletons(
    model,
    processor: Phi0Processor,
    seq: SequenceDataset,
    clip_idx: int,
    *,
    hdf5_path: Path,
    out_dir: Path,
    constants: dict[str, np.ndarray],
    clip_cache: ClipInputsCache,
    prompt_cache: PromptEmbedCache,
    stride: int = 2,
    make_gif: bool = True,
) -> Dict[str, Any]:
    clip_item = seq[clip_idx]
    native_start = int(clip_item.get("native_start", clip_item["idx"]))
    ds_name = str(clip_item["dataset"])
    native_fps = float(seq.native_fps[ds_name])
    control_fps = float(clip_item.get("control_fps", seq.control_fps))
    proprio_w = int(model.past_action_window_size)

    inputs = clip_cache.get_or_build(
        model, processor, seq, clip_idx, prompt_cache=prompt_cache, cache_action_context=True
    )
    pred_norm = model.predict_action_chunk(inputs)
    _, target_norm, target_pad, target_dim_pad = model._action_proprio_future(
        inputs["action"], inputs.get("action_is_pad"), inputs.get("action_dim_is_pad")
    )

    pred_denorm = processor.postprocess(pred_norm.float()).cpu().numpy()[0]
    target_denorm = processor.postprocess(target_norm.float()).cpu().numpy()[0]
    mse = float(
        compute_masked_mse(pred_norm, target_norm, target_pad, target_dim_pad).item()
    )
    comps = unified_component_losses(pred_norm, target_norm, target_pad, target_dim_pad)

    clip_dir = out_dir / f"clip_{clip_idx:04d}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    pred_errors: List[float] = []
    pred_kp_errors: List[float] = []
    gt_ref_errors: List[float] = []
    png_paths: List[Path] = []
    viz_frames: List[Dict[str, Any]] = []

    with h5py.File(hdf5_path, "r") as f:
        anchor_root = read_xperience_root_trans_world(f, native_start)
        n_future = pred_denorm.shape[0]
        for k in range(0, n_future, max(1, stride)):
            control_idx = proprio_w + k
            native_t = control_index_to_native(
                native_start, control_idx, native_fps=native_fps, control_fps=control_fps
            )
            native_t = min(native_t, int(f["full_body_mocap/body_quats"].shape[0]) - 1)
            betas = f["full_body_mocap/betas"][native_t].astype(np.float32)

            pred_j = fk_world_joints(
                pred_denorm[k],
                state_root_trans_world=anchor_root,
                betas=betas,
                constants=constants,
            )
            gt_j = fk_world_joints(
                target_denorm[k],
                state_root_trans_world=anchor_root,
                betas=betas,
                constants=constants,
            )
            ref_j = reference_joints_world_from_hdf5_quat(f, native_t, constants=constants)
            kp_ref = hdf5_keypoints_for_viz(f["full_body_mocap/keypoints"][native_t])
            pred_j_kp = align_fk_joints_to_keypoints_frame(pred_j, kp_ref)[0]
            gt_j_kp = align_fk_joints_to_keypoints_frame(gt_j, kp_ref)[0]

            pred_err = joint_l2_mean(pred_j, gt_j)
            pred_err_kp = joint_l2_mean(pred_j_kp[1:], gt_j_kp[1:])
            gt_ref_err = joint_l2_max(gt_j, ref_j)
            gt_kp_ref_err = joint_l2_max(gt_j_kp[1:], kp_ref[1:])
            pred_errors.append(pred_err)
            pred_kp_errors.append(pred_err_kp)
            gt_ref_errors.append(gt_ref_err)
            viz_frames.append(
                {
                    "k": k,
                    "native_t": native_t,
                    "kp_ref": kp_ref,
                    "pred_j_kp": pred_j_kp,
                    "gt_j_kp": gt_j_kp,
                    "pred_err": pred_err,
                    "pred_err_kp": pred_err_kp,
                    "gt_ref_err": gt_ref_err,
                    "gt_kp_ref_err": gt_kp_ref_err,
                }
            )


    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    all_pts: list[np.ndarray] = []
    for vf in viz_frames:
        all_pts.extend((vf["kp_ref"], vf["pred_j_kp"], vf["gt_j_kp"]))

    bounds_center, bounds_radius = compute_keypoints_viz_bounds(*all_pts)

    for vf in viz_frames:
        k = vf["k"]
        native_t = vf["native_t"]
        pred_err = vf["pred_err"]
        pred_err_kp = vf["pred_err_kp"]
        gt_kp_ref_err = vf["gt_kp_ref_err"]
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        draw_ground_plane(ax, bounds_center, bounds_radius, z=float(bounds_center[2] - bounds_radius))
        draw_skeleton(ax, vf["kp_ref"], color="tab:orange", alpha=0.85, linewidth=1.2)
        draw_skeleton(ax, vf["gt_j_kp"], color="tab:green", alpha=0.75, linewidth=1.1)
        draw_skeleton(ax, vf["pred_j_kp"], color="royalblue", alpha=0.85, linewidth=1.2)
        apply_scene_limits(ax, bounds_center, bounds_radius, ground_at_z0=False)
        configure_mpl3d_skeleton_axes(ax)
        ax.legend(
            handles=[
                Line2D([0], [0], color="royalblue", lw=2, label=f"Pred (kp L2 vs GT {pred_err_kp:.3f} m)"),
                Line2D([0], [0], color="tab:green", lw=2, label=f"GT unified (world L2 {pred_err:.3f} m)"),
                Line2D(
                    [0],
                    [0],
                    color="tab:orange",
                    lw=2,
                    label=f"HDF5 keypoints (GT kp err {gt_kp_ref_err:.2e} m)",
                ),
            ],
            loc="upper left",
            fontsize=8,
        )
        fig.suptitle(
            f"clip={clip_idx} future_k={k} native_t={native_t} anchor={native_start} "
            f"masked_mse={mse:.4f} world_L2={pred_err:.3f}m kp_L2={pred_err_kp:.3f}m  "
            f"[per-frame Sim(3) FK skeleton → keypoints frame]"
        )
        fig.tight_layout()
        png = clip_dir / f"future_{k:03d}_t{native_t:05d}.png"
        fig.savefig(png, dpi=120)
        plt.close(fig)
        png_paths.append(png)

    summary = {
        "clip_idx": clip_idx,
        "native_start": native_start,
        "masked_mse": mse,
        "component_mse": comps,
        "pred_vs_gt_joint_l2_mean": float(np.mean(pred_errors)) if pred_errors else float("nan"),
        "pred_vs_gt_joint_l2_max": float(np.max(pred_errors)) if pred_errors else float("nan"),
        "pred_vs_gt_keypoints_sim3_l2_mean": float(np.mean(pred_kp_errors)) if pred_kp_errors else float("nan"),
        "pred_vs_gt_keypoints_sim3_l2_max": float(np.max(pred_kp_errors)) if pred_kp_errors else float("nan"),
        "gt_vs_quat_ref_max_m": float(np.max(gt_ref_errors)) if gt_ref_errors else float("nan"),
        "num_viz_frames": len(pred_errors),
        "png_dir": str(clip_dir),
    }

    if make_gif and png_paths:
        try:
            import imageio.v2 as imageio

            frames = [imageio.imread(p) for p in png_paths]
            gif_path = clip_dir / "skeleton_pred_gt.gif"
            imageio.mimsave(gif_path, frames, duration=0.15)
            summary["gif"] = str(gif_path)
        except ImportError:
            logger.warning("imageio not installed; skipped GIF for clip %d", clip_idx)

    return summary


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ckpt_path = Path(args.checkpoint).resolve()
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "eval_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(version_base="1.3", config_dir=args.config_dir):
        cfg = compose(config_name=args.config_name)

    device = resolve_inference_device(args.device, min_free_gb=float(args.min_free_gb))
    activate_cuda_device(device)
    cfg.device = device

    logger.info("Loading checkpoint %s", ckpt_path)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and payload.get("cfg"):
        cfg = merge_saved_cfg(cfg, payload["cfg"])

    model = create_phi0(cfg, smoke=bool(cfg.get("smoke_action_only", False)))
    if isinstance(payload, dict) and ("model" in payload or "action_expert" in payload):
        model.load_checkpoint(str(ckpt_path))
    processor = build_processor(cfg).eval()
    if isinstance(payload, dict):
        apply_processor_stats_from_checkpoint(processor, payload, cfg)
    sync_model_action_norm(model, processor)
    model.eval()

    cosmos_hw = round_hw_to_multiple(*cosmos_video_size_from_cfg(cfg.data))
    base = build_base_dataset(cfg)
    seq = sequence_dataset_from_cfg(base, cfg.data)
    prompt_cache = PromptEmbedCache()
    clip_cache = ClipInputsCache()
    constants = load_skeleton_constants()

    max_clips = max(1, int(args.max_clips))
    logger.info("Chunk eval on up to %d clips (action_head=%s)...", max_clips, model.action_head)
    chunk_report = fm_chunk_eval(
        model,
        processor,
        cfg,
        max_clips=max_clips,
        seq=seq,
        prompt_cache=prompt_cache,
        clip_cache=clip_cache,
    )
    for clip in chunk_report.get("clips", []):
        clip_idx = int(clip["clip_idx"])
        inputs = clip_cache.get_or_build(
            model, processor, seq, clip_idx, prompt_cache=prompt_cache, cache_action_context=True
        )
        pred_norm = model.predict_action_chunk(inputs)
        _, target_norm, target_pad, target_dim_pad = model._action_proprio_future(
            inputs["action"], inputs.get("action_is_pad"), inputs.get("action_dim_is_pad")
        )
        clip["unified_component_mse"] = unified_component_losses(
            pred_norm, target_norm, target_pad, target_dim_pad
        )

    viz_summaries: List[Dict[str, Any]] = []
    viz_indices = [i for i in args.viz_clips if 0 <= i < len(seq)]
    for clip_idx in tqdm(viz_indices, desc="viz-clips", unit="clip"):
        viz_summaries.append(
            render_clip_skeletons(
                model,
                processor,
                seq,
                clip_idx,
                hdf5_path=Path(args.hdf5),
                out_dir=out_dir,
                constants=constants,
                clip_cache=clip_cache,
                prompt_cache=prompt_cache,
                stride=int(args.viz_stride),
                make_gif=bool(args.make_gif),
            )
        )

    report: Dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "config_name": args.config_name,
        "action_head": str(model.action_head),
        "raw_action_dim": int(cfg.model.get("raw_action_dim", 512)),
        "device": device,
        "num_clips_in_dataset": len(seq),
        "chunk_eval": chunk_report,
        "viz_clips": viz_summaries,
        "hdf5": str(Path(args.hdf5).resolve()),
        "output_dir": str(out_dir.resolve()),
        "notes": {
            "loss": "masked MSE on normalized future actions (training-aligned)",
            "skeleton": "FK skeleton only; per-frame Sim(3) Procrustes → HDF5 keypoints frame",
            "panels": "pred | GT unified | HDF5 keypoints (orange skeleton)",
            "joint_metrics": "world_L2 in mocap FK; kp_L2 after per-frame Sim(3) to keypoints",
        },
    }
    report_path = out_dir / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "Eval done: chunk masked_mse mean=%.4f | pred vs GT joint_l2 mean=%.4f",
        chunk_report.get("masked_mse_mean", float("nan")),
        float(np.mean([v["pred_vs_gt_joint_l2_mean"] for v in viz_summaries]))
        if viz_summaries
        else float("nan"),
    )
    logger.info("Wrote %s", report_path)
    print(json.dumps({k: report[k] for k in ("chunk_eval", "viz_clips", "output_dir")}, indent=2))


if __name__ == "__main__":
    main()
