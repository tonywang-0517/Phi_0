#!/usr/bin/env python3
"""Evaluate Phi_0 action prediction: FM chunk MSE + deploy FM metrics vs GT."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm import tqdm

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phi0.checkpoint_utils import merge_saved_cfg
from phi0.data.egodex import EgoDexDataset
from phi0.data.processor import Phi0Processor, build_overfit_datasets
from phi0.data.sequence import SequenceDataset, sequence_dataset_from_cfg
from phi0.data.temporal_align import (
    DEFAULT_DATASET_NATIVE_FPS,
    native_span_frames,
    resample_action_sequence,
)
from phi0.data.xperience import XperienceDataset
from phi0.inference.deploy_align import (
    build_deploy_video_tensor,
    deploy_proprio_control_indices,
    deploy_subsampled_video_control_indices,
)
from phi0.inference.session import (
    ActionInferenceSession,
    ClipInputsCache,
    PromptEmbedCache,
    resolve_deploy_action_chunk_size,
)
from phi0.runtime import (
    activate_cuda_device,
    apply_processor_stats_from_checkpoint,
    create_phi0,
    list_cuda_memory,
    prepare_model_batch,
    resolve_inference_device,
    sync_model_action_norm,
)
from phi0.schema.draw_schema import D_RAW
from phi0.schema.action_schema import get_action_schema, unpack_keypoints_52
from phi0.viz.skeleton import load_gt_from_hdf5
from phi0.viz.xperience_viz_frame import hdf5_keypoints_for_viz


def parse_args():
    p = argparse.ArgumentParser(description="Phi_0 action evaluation")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "experiments/phi0_full/phi0_smoke.pt"),
    )
    p.add_argument("--config-dir", type=str, default=str(ROOT / "configs"))
    p.add_argument("--config-name", type=str, default="train_full")
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda/auto: pick GPU with most free VRAM; cuda:N: explicit GPU; cpu",
    )
    p.add_argument(
        "--min-free-gb",
        type=float,
        default=18.0,
        help="When auto-picking GPU, require at least this much free VRAM",
    )
    p.add_argument("--num-deploy-frames", type=int, default=None, help="Override frame count (else deploy-seconds * fps)")
    p.add_argument("--deploy-seconds", type=float, default=5.0, help="Deploy FM eval duration in seconds")
    p.add_argument("--deploy-fps", type=float, default=None, help="Control Hz (default: cfg.data.control_fps or 20)")
    p.add_argument(
        "--video-refresh-interval",
        type=int,
        default=4,
        help="Forward video tower every N deploy frames to refresh latent/action context",
    )
    p.add_argument("--deploy-start-frame", type=int, default=0)
    p.add_argument(
        "--max-clips",
        type=int,
        default=32,
        help="Cap FM chunk-eval clips (full dataset can be 200+ clips, ~10s each)",
    )
    p.add_argument("--output", type=str, default=None, help="Optional JSON report path")
    return p.parse_args()


def compute_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    action_is_pad: Optional[torch.Tensor],
    action_dim_is_pad: Optional[torch.Tensor],
    per_frame: bool = False,
) -> torch.Tensor:
    """Same masking logic as Phi0._compute_action_loss."""
    loss = F.mse_loss(pred.float(), target.float(), reduction="none")
    if action_dim_is_pad is not None:
        pad = action_dim_is_pad.to(device=loss.device)
        if pad.ndim == 1:
            dim_valid = (~pad).to(dtype=loss.dtype).view(1, 1, -1)
        elif pad.ndim == 2:
            dim_valid = (~pad).to(dtype=loss.dtype).unsqueeze(0)
        else:
            dim_valid = (~pad).to(dtype=loss.dtype)
        loss = loss * dim_valid
    if action_is_pad is not None:
        token_valid = (~action_is_pad).to(device=loss.device, dtype=loss.dtype).unsqueeze(-1)
        loss = loss * token_valid

    if per_frame:
        if action_dim_is_pad is None:
            return loss.mean(dim=-1)
        pad = action_dim_is_pad
        if pad.ndim == 1:
            dim_valid = (~pad).float().view(1, 1, -1)
        elif pad.ndim == 2:
            dim_valid = (~pad).float().unsqueeze(0)
        else:
            dim_valid = (~pad).float()
        if action_is_pad is not None:
            token_valid = (~action_is_pad).float().unsqueeze(-1)
            denom = (dim_valid * token_valid).sum(dim=-1).clamp(min=1.0)
            return loss.sum(dim=-1) / denom
        denom = dim_valid.sum(dim=-1).clamp(min=1.0)
        return loss.sum(dim=-1) / denom

    if action_dim_is_pad is not None and action_is_pad is not None:
        pad = action_dim_is_pad
        if pad.ndim == 1:
            dim_valid = (~pad).float().view(1, 1, -1)
        elif pad.ndim == 2:
            dim_valid = (~pad).float().unsqueeze(0)
        else:
            dim_valid = (~pad).float()
        token_valid = (~action_is_pad).float().unsqueeze(-1)
        denom = (dim_valid * token_valid).sum().clamp(min=1.0)
        return loss.sum() / denom
    return loss.mean()


def component_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    action_is_pad: Optional[torch.Tensor],
    action_dim_is_pad: Optional[torch.Tensor],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    schema = get_action_schema()
    for name, (s, e) in schema.slices.items():
        dim_pad = None
        if action_dim_is_pad is not None:
            dim_pad = action_dim_is_pad[..., s:e]
            if dim_pad.ndim == 1:
                dim_pad = dim_pad.unsqueeze(0).unsqueeze(0)
            # No GT for this slice — report as N/A (not supervised).
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


def l2_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    diff = pred - gt
    per_frame = np.linalg.norm(diff.reshape(len(pred), -1), axis=1)
    return {
        "mean": float(np.mean(per_frame)),
        "max": float(np.max(per_frame)),
        "per_frame": per_frame.tolist(),
    }


def root_l2(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Joint-0 keypoint L2 (anchor position in keypoints frame)."""
    diff = unpack_keypoints_52(pred)[:, 0, :] - unpack_keypoints_52(gt)[:, 0, :]
    per_frame = np.linalg.norm(diff, axis=1)
    return {
        "mean": float(np.mean(per_frame)),
        "max": float(np.max(per_frame)),
        "per_frame": per_frame.tolist(),
    }


def skeleton_l2(pred_d_raw: np.ndarray, gt_d_raw: np.ndarray) -> Dict[str, float]:
    pred_j = unpack_keypoints_52(pred_d_raw)
    gt_j = unpack_keypoints_52(gt_d_raw)
    per_joint = np.linalg.norm(pred_j - gt_j, axis=-1).mean(axis=1)
    return {
        "mean": float(np.mean(per_joint)),
        "max": float(np.max(per_joint)),
        "per_frame": per_joint.tolist(),
    }


def skeleton_metrics_vs_hdf5(
    pred_d_raw: np.ndarray,
    gt_hdf5: dict[str, np.ndarray],
) -> Dict[str, float]:
    """Pred vs GT HDF5 keypoints (direct unpack)."""
    pred_kp = unpack_keypoints_52(pred_d_raw)
    gt_kp = hdf5_keypoints_for_viz(gt_hdf5["keypoints"])
    skel_mean_joint = np.linalg.norm(pred_kp - gt_kp, axis=-1).mean(axis=1)
    return {
        "skeleton_mean_joint_l2_mean": float(np.mean(skel_mean_joint)),
        "skeleton_mean_joint_l2_max": float(np.max(skel_mean_joint)),
    }


@torch.no_grad()
def fm_chunk_eval(
    model,
    processor,
    cfg,
    max_clips: int | None = None,
    *,
    seq: Optional[SequenceDataset] = None,
    prompt_cache: Optional[PromptEmbedCache] = None,
    clip_cache: Optional[ClipInputsCache] = None,
) -> Dict[str, Any]:
    data_cfg = cfg.data
    if seq is None:
        mixed = build_overfit_datasets(
            xperience_max_frames=int(data_cfg.get("xperience_max_frames", 16)),
            egodex_max_frames=int(data_cfg.get("egodex_max_frames", 16)),
        )
        seq = sequence_dataset_from_cfg(mixed, data_cfg)
    processor = processor.eval()
    if prompt_cache is None:
        prompt_cache = PromptEmbedCache()
    if clip_cache is None:
        clip_cache = ClipInputsCache()

    n_clips = len(seq)
    if max_clips is not None and max_clips > 0:
        n_clips = min(n_clips, int(max_clips))

    results: Dict[str, Any] = {
        "clips": [],
        "by_dataset": {},
        "num_clips_evaluated": n_clips,
        "eval_notes": {
            "mode": "FM Euler denoise on full clip (DiT4DiT-style chunk prediction)",
            "vision_context": "uses model.action_context_mode (default first_frame, deploy-aligned)",
            "action_space": "normalized (processor mean=0 std=1 unless stats registered)",
            "caching": "prompt embed + per-clip VAE/Cosmos hook cached (ClipInputsCache)",
        },
        "cache_stats": {},
    }
    all_losses = []
    all_per_frame = []

    clip_iter = tqdm(range(n_clips), desc="fm-chunk", unit="clip", file=sys.stdout)
    for clip_idx in clip_iter:
        inputs = clip_cache.get_or_build(
            model,
            processor,
            seq,
            clip_idx,
            prompt_cache=prompt_cache,
            cache_action_context=True,
        )
        pred = model.predict_action_chunk(inputs)
        _, target, target_pad, target_dim_pad = model._action_proprio_future(
            inputs["action"], inputs.get("action_is_pad"), inputs.get("action_dim_is_pad")
        )
        batch = SequenceDataset.collate_fn([seq[clip_idx]])
        ds = batch["dataset"][0]
        loss = compute_masked_mse(
            pred,
            target,
            target_pad,
            target_dim_pad,
        )
        per_frame = compute_masked_mse(
            pred,
            target,
            target_pad,
            target_dim_pad,
            per_frame=True,
        )
        comps = component_losses(pred, target, target_pad, target_dim_pad)

        ds = batch["dataset"][0]
        clip = {
            "clip_idx": clip_idx,
            "dataset": ds,
            "masked_mse": float(loss.item()),
            "masked_mse_per_frame": per_frame.squeeze(0).cpu().tolist(),
            "component_mse": comps,
        }
        results["clips"].append(clip)
        all_losses.append(float(loss.item()))
        all_per_frame.extend(clip["masked_mse_per_frame"])

        by_ds = results["by_dataset"].setdefault(
            ds, {"losses": [], "component_sums": {k: 0.0 for k in get_action_schema().slices}, "n": 0}
        )
        by_ds["losses"].append(float(loss.item()))
        by_ds["n"] += 1
        for k, v in comps.items():
            if not (isinstance(v, float) and np.isnan(v)):
                by_ds["component_sums"][k] += v
        clip_iter.set_postfix(mse=f"{loss.item():.4f}", ds=ds, cache=clip_cache.hits)

    results["cache_stats"] = {
        "prompt_unique": len(prompt_cache._store),
        **clip_cache.stats(),
    }

    for ds, agg in results["by_dataset"].items():
        n = max(agg["n"], 1)
        agg["masked_mse_mean"] = float(np.mean(agg["losses"]))
        agg["component_mse_mean"] = {k: v / n for k, v in agg["component_sums"].items()}
        del agg["component_sums"]

    results["masked_mse_mean"] = float(np.mean(all_losses))
    results["masked_mse_per_frame_mean"] = float(np.mean(all_per_frame))
    results["masked_mse_per_frame_std"] = float(np.std(all_per_frame))
    return results


@torch.no_grad()
def random_baseline_eval(
    model,
    processor,
    cfg,
    max_clips: int | None = None,
    *,
    seq: Optional[SequenceDataset] = None,
    clip_cache: Optional[ClipInputsCache] = None,
    prompt_cache: Optional[PromptEmbedCache] = None,
) -> Dict[str, Any]:
    """Zero-action baseline with same FM chunk-eval setup."""
    data_cfg = cfg.data
    if seq is None:
        mixed = build_overfit_datasets(
            xperience_max_frames=int(data_cfg.get("xperience_max_frames", 16)),
            egodex_max_frames=int(data_cfg.get("egodex_max_frames", 16)),
        )
        seq = sequence_dataset_from_cfg(mixed, data_cfg)
    processor = processor.eval()
    n_clips = len(seq)
    if max_clips is not None and max_clips > 0:
        n_clips = min(n_clips, int(max_clips))
    losses = []
    for clip_idx in tqdm(range(n_clips), desc="zero-baseline", unit="clip", file=sys.stdout):
        if clip_cache is not None:
            inputs = clip_cache.get_or_build(
                model, processor, seq, clip_idx, prompt_cache=prompt_cache, cache_action_context=True
            )
            _, target, token_pad, dim_pad = model._action_proprio_future(
                inputs["action"], inputs.get("action_is_pad"), inputs.get("action_dim_is_pad")
            )
            mask_pad = dim_pad
        else:
            batch = SequenceDataset.collate_fn([seq[clip_idx]])
            mb = prepare_model_batch(model, processor, batch, prompt_cache=prompt_cache)
            mb = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in mb.items()}
            _, target, token_pad, dim_pad = model._action_proprio_future(
                mb["action"], mb.get("action_is_pad"), mb.get("action_dim_is_pad")
            )
            mask_pad = dim_pad
        pred = torch.zeros_like(target)
        loss = compute_masked_mse(pred, target, token_pad, mask_pad)
        losses.append(float(loss.item()))
    return {"masked_mse_mean": float(np.mean(losses))}


def _round_to_multiple(value: int, base: int = 16) -> int:
    return max(base, (value // base) * base)


def _load_rgb_frame(video_path: Path, frame_index: int, size_hw: tuple[int, int]) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = size_hw
    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    return frame.astype(np.uint8)


def _build_input_tensor(model, rgb: np.ndarray) -> torch.Tensor:
    from PIL import Image

    h, w = rgb.shape[:2]
    if h % 16 != 0 or w % 16 != 0:
        h = _round_to_multiple(h)
        w = _round_to_multiple(w)
        rgb = np.asarray(Image.fromarray(rgb).resize((w, h), resample=Image.BILINEAR), dtype=np.uint8)
    image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(
        device=model.device, dtype=model.torch_dtype,
    )
    return image_tensor * (2.0 / 255.0) - 1.0


def _chw_to_input_tensor(model, frame_chw: torch.Tensor) -> torch.Tensor:
    """Convert cached [C,H,W] float01 or uint8 tensor to model input in [-1, 1]."""
    if frame_chw.ndim == 3:
        frame_chw = frame_chw.unsqueeze(0)
    t = frame_chw.to(device=model.device, dtype=model.torch_dtype)
    if float(t.max()) > 1.5:
        t = t / 255.0
    return t * 2.0 - 1.0


def _action_dim_pad_mask(dataset) -> torch.Tensor:
    """True = dim excluded from loss (matches batch action_dim_is_pad)."""
    avail = np.asarray(getattr(dataset, "action_dim_is_pad"))
    return torch.from_numpy(~avail).unsqueeze(0)


def _normalize_d_raw(processor: Phi0Processor, d_raw: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(d_raw).float().unsqueeze(0)
    return processor._normalize_action(t)


@torch.no_grad()
def clip_hook_eval(
    model,
    processor,
    clip_batch: Dict[str, Any],
) -> Dict[str, float]:
    """Eval aligned with training: full subsampled video + proprio prefix."""
    mb = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in clip_batch.items()}
    inputs = model.build_inputs(mb)
    pred = model.predict_action_chunk(inputs)
    _, target, target_pad, target_dim_pad = model._action_proprio_future(
        inputs["action"], inputs.get("action_is_pad"), inputs.get("action_dim_is_pad")
    )
    loss = compute_masked_mse(
        pred, target, target_pad, target_dim_pad
    )
    loss_f0 = compute_masked_mse(
        pred[:, :1], target[:, :1], target_pad[:, :1] if target_pad is not None else None, target_dim_pad
    )
    return {"masked_mse": float(loss.item()), "masked_mse_frame0": float(loss_f0.item())}


@torch.no_grad()
def deploy_fm_eval(
    model,
    processor,
    num_frames: int,
    start_frame: int,
    *,
    deploy_fps: float = 20.0,
    video_refresh_interval: int = 4,
    native_fps: float = 20.0,
    action_video_freq_ratio: int = 2,
    action_chunk_size: int | None = None,
    seq_len: int = 33,
) -> Dict[str, Any]:
    xp = XperienceDataset(max_frames=start_frame + native_span_frames(num_frames, deploy_fps, native_fps), start_frame=0, cache_video=True)
    video_path = xp.video_path
    if video_path is None or not video_path.exists():
        raise FileNotFoundError("Xperience stereo_left.mp4 not found for deploy eval")

    h, w = xp.image_size
    h, w = _round_to_multiple(h), _round_to_multiple(w)
    prompt = xp.task_text
    prompt_cache = PromptEmbedCache()
    refresh_every = max(1, int(video_refresh_interval))
    action_chunk = (
        int(action_chunk_size)
        if action_chunk_size is not None
        else resolve_deploy_action_chunk_size(model, seq_len=int(seq_len))
    )
    video_ctrl_idx = deploy_subsampled_video_control_indices(
        num_frames - 1,
        seq_len=seq_len,
        action_video_freq_ratio=action_video_freq_ratio,
    )

    native_len = native_span_frames(num_frames, deploy_fps, native_fps)
    gt_native = np.stack(
        [xp._load_frame_action(start_frame + i) for i in range(native_len)],
        axis=0,
    )
    gt_d_raw = resample_action_sequence(
        torch.from_numpy(gt_native).float(), native_len, num_frames
    ).numpy()
    gt_norm_by_control = [
        _normalize_d_raw(processor, gt_d_raw[t]) for t in range(num_frames)
    ]

    session = ActionInferenceSession(
        model,
        processor=processor,
        deploy_seq_len=seq_len,
        action_video_freq_ratio=action_video_freq_ratio,
        use_gt_proprio=True,
    )
    proprio_w = int(getattr(model, "past_action_window_size", 0))

    def _read_chw(control_t: int) -> torch.Tensor:
        native_t = start_frame + int(round(control_t * native_fps / deploy_fps))
        if xp._video_frames is not None and 0 <= native_t < len(xp._video_frames):
            return xp._video_frames[native_t]
        rgb_t = _load_rgb_frame(video_path, native_t, (h, w))
        return torch.from_numpy(rgb_t).permute(2, 0, 1).float() / 255.0

    def _deploy_video_clip(seg_start: int) -> torch.Tensor:
        return build_deploy_video_tensor(
            seg_start,
            _read_chw,
            seq_len=seq_len,
            action_video_freq_ratio=action_video_freq_ratio,
            past_window=proprio_w,
            device=model.device,
            dtype=model.torch_dtype,
        )

    preds_norm = []
    for seg_start in tqdm(range(0, num_frames, action_chunk), desc="deploy FM", unit="seg", file=sys.stdout):
        if seg_start == 0:
            session.prefill_from_video_clip(
                _deploy_video_clip(0), prompt, prompt_cache=prompt_cache,
            )
        else:
            session.refresh_video_context_from_clip(_deploy_video_clip(seg_start))
        if proprio_w > 0:
            proprio_ctrl = deploy_proprio_control_indices(seg_start, proprio_w)
            session.set_proprio_gt(
                torch.stack([gt_norm_by_control[c].reshape(-1) for c in proprio_ctrl], dim=0)
            )
        chunk_len = min(action_chunk, num_frames - seg_start)
        chunk = session.predict(chunk_len)
        for i in range(chunk_len):
            preds_norm.append(chunk[i].detach().float().cpu())

    pred_norm_t = torch.stack(preds_norm, dim=0).unsqueeze(0).to(device=model.device)
    pred_d_raw = np.stack(
        [processor.postprocess(pred_norm_t[:, i]).reshape(-1).detach().cpu().numpy() for i in range(num_frames)],
        axis=0,
    )

    gt_norm_t = torch.stack(gt_norm_by_control, dim=0).unsqueeze(0).to(device=model.device)
    gt_hdf5 = load_gt_from_hdf5(xp.hdf5_path, start_frame, native_len)
    gt_kp_viz = hdf5_keypoints_for_viz(gt_hdf5["keypoints"])
    gt_kp_from_d_raw = unpack_keypoints_52(gt_d_raw)
    gt_packed_vs_hdf5_mean_joint = float(
        np.linalg.norm(gt_kp_from_d_raw - gt_kp_viz, axis=-1).mean(axis=1).mean()
    )
    dim_pad = _action_dim_pad_mask(xp).to(device=model.device)
    token_pad = torch.zeros(1, num_frames, dtype=torch.bool, device=model.device)

    masked_mse = float(compute_masked_mse(pred_norm_t, gt_norm_t, token_pad, dim_pad).item())
    per_frame_mse = compute_masked_mse(
        pred_norm_t, gt_norm_t, token_pad, dim_pad, per_frame=True
    ).squeeze(0).cpu().numpy().tolist()

    comps = component_losses(pred_norm_t, gt_norm_t, token_pad, dim_pad)

    zero_pred = torch.zeros_like(pred_norm_t)
    random_mse = float(compute_masked_mse(zero_pred, gt_norm_t, token_pad, dim_pad).item())

    # Deploy-aligned FM chunk eval on matching clip (first-frame hook only).
    from phi0.data.processor import build_overfit_datasets as _build

    mixed = _build(xperience_max_frames=max(start_frame + native_len, 256), egodex_max_frames=16)
    align_cfg = {
        "seq_len": num_frames,
        "control_fps": deploy_fps,
        "action_video_freq_ratio": action_video_freq_ratio,
        "dataset_native_fps": DEFAULT_DATASET_NATIVE_FPS,
    }
    seq = sequence_dataset_from_cfg(mixed, align_cfg)
    clip_idx = min(start_frame, len(seq) - 1)
    clip_batch = prepare_model_batch(model, processor, SequenceDataset.collate_fn([seq[clip_idx]]), prompt_cache=prompt_cache)
    clip_batch = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in clip_batch.items()}
    deploy_aligned_fm = clip_hook_eval(model, processor, clip_batch)

    return {
        "num_frames": num_frames,
        "deploy_seconds": num_frames / float(deploy_fps),
        "deploy_fps": float(deploy_fps),
        "native_fps": float(native_fps),
        "action_video_freq_ratio": int(action_video_freq_ratio),
        "video_control_indices": video_ctrl_idx,
        "video_tower_refresh_count": session.video_refresh_count,
        "video_refresh_interval_frames": refresh_every,
        "action_chunk_size": action_chunk,
        "deploy_seq_len": int(seq_len),
        "deploy_video_clip_frames": len(video_ctrl_idx),
        "use_gt_proprio": True,
        "deploy_align": "training_clip_gt_proprio",
        "proprio_window": proprio_w,
        "fm_segment_len": action_chunk,
        "start_frame": start_frame,
        "video": str(video_path),
        "prompt": prompt,
        "masked_mse": masked_mse,
        "masked_mse_space": "normalized (same as training loss)",
        "masked_mse_per_frame": per_frame_mse,
        "component_mse": comps,
        "zero_baseline_masked_mse": random_mse,
        "fm_chunk_aligned": deploy_aligned_fm,
        "eval_notes": {
            "fm_segments": "Each segment predicts action_chunk_size frames (training-aligned horizon); "
            "video context refreshed at segment boundaries.",
            "vision_context": f"{len(video_ctrl_idx)}-frame subsampled clip (seq_len={seq_len}, ratio={action_video_freq_ratio}) ending at segment start",
            "proprio": "GT normalized actions at the 4 control steps before each segment start",
            "temporal_align": "GT/action resampled to control_fps; video read at native_fps mapping",
            "deterministic_hook": "capture_stochastic=false uses sigma=0 (config default in phi0_full.yaml)",
        },
        "root_l2_joint0": root_l2(pred_d_raw, gt_d_raw),
        "skeleton_l2_keypoints": skeleton_l2(pred_d_raw, gt_d_raw),
        "skeleton_metrics_vs_hdf5": skeleton_metrics_vs_hdf5(pred_d_raw, gt_hdf5),
        "gt_packed_vs_hdf5_mean_joint_l2": gt_packed_vs_hdf5_mean_joint,
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with initialize_config_dir(version_base="1.3", config_dir=args.config_dir):
        cfg = compose(config_name=args.config_name)
    device = resolve_inference_device(args.device, min_free_gb=float(args.min_free_gb))
    activate_cuda_device(device)
    cfg.device = device

    logger.info("Loading checkpoint %s on %s", args.checkpoint, device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_cfg = payload.get("cfg") if isinstance(payload, dict) else None
    if saved_cfg:
        cfg = merge_saved_cfg(cfg, saved_cfg)

    model = create_phi0(cfg, smoke=bool(cfg.get("smoke_action_only", False)))
    if isinstance(payload, dict) and ("model" in payload or "action_expert" in payload):
        model.load_checkpoint(args.checkpoint)

    processor = Phi0Processor(normalize=True).eval()
    if isinstance(payload, dict):
        apply_processor_stats_from_checkpoint(processor, payload, cfg)
    sync_model_action_norm(model, processor)
    model.eval()

    max_clips = args.max_clips if args.max_clips > 0 else None
    deploy_fps = float(args.deploy_fps if args.deploy_fps is not None else cfg.data.get("control_fps", 20.0))
    deploy_seconds = float(args.deploy_seconds)
    refresh_iv = int(args.video_refresh_interval)
    native_fps_map = OmegaConf.to_container(
        cfg.data.get("dataset_native_fps", DEFAULT_DATASET_NATIVE_FPS), resolve=True
    )
    native_fps = float(dict(native_fps_map).get("xperience", 20.0))
    video_ratio = int(cfg.data.get("action_video_freq_ratio", 2))
    if args.num_deploy_frames is not None:
        deploy_frames = int(args.num_deploy_frames)
    else:
        deploy_frames = max(1, int(round(deploy_seconds * deploy_fps)))

    mixed = build_overfit_datasets(
        xperience_max_frames=int(cfg.data.get("xperience_max_frames", 16)),
        egodex_max_frames=int(cfg.data.get("egodex_max_frames", 16)),
    )
    seq = sequence_dataset_from_cfg(mixed, cfg.data)
    prompt_cache = PromptEmbedCache()
    clip_cache = ClipInputsCache()

    logger.info("FM chunk eval (max_clips=%s)...", max_clips)
    fm_chunk = fm_chunk_eval(
        model,
        processor,
        cfg,
        max_clips=max_clips,
        seq=seq,
        prompt_cache=prompt_cache,
        clip_cache=clip_cache,
    )
    logger.info("Zero baseline eval (reusing clip cache)...")
    zb = random_baseline_eval(
        model,
        processor,
        cfg,
        max_clips=max_clips,
        seq=seq,
        clip_cache=clip_cache,
        prompt_cache=prompt_cache,
    )
    logger.info("Deploy FM eval (%d frames, %.1fs @ %.0fHz, video refresh every %d)...",
                deploy_frames, deploy_seconds, deploy_fps, refresh_iv)
    deploy = deploy_fm_eval(
        model,
        processor,
        num_frames=deploy_frames,
        start_frame=args.deploy_start_frame,
        deploy_fps=deploy_fps,
        video_refresh_interval=refresh_iv,
        native_fps=native_fps,
        action_video_freq_ratio=video_ratio,
        seq_len=int(cfg.data.get("seq_len", 33)),
    )

    report: Dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": device,
        "device_selection": {
            "requested": args.device,
            "resolved": device,
            "min_free_gb": float(args.min_free_gb),
            "visible_gpus": [
                {"index": i, "free_gib": free / (1024**3), "total_gib": total / (1024**3)}
                for i, free, total in list_cuda_memory()
            ],
        },
        "training": {
            "max_steps_in_ckpt_cfg": int(cfg.get("max_steps", 0)),
            "smoke_action_only": bool(cfg.get("smoke_action_only", False)),
            "control_fps": float(cfg.data.get("control_fps", 20.0)),
            "action_video_freq_ratio": int(cfg.data.get("action_video_freq_ratio", 2)),
            "dataset_native_fps": dict(native_fps_map),
            "xperience_max_frames": int(cfg.data.get("xperience_max_frames", 16)),
            "egodex_max_frames": int(cfg.data.get("egodex_max_frames", 16)),
            "action_dit_layers": int(cfg.model.action_dit_config.num_layers),
            "action_context_mode": str(cfg.model.get("action_context_mode", "first_frame")),
            "capture_stochastic": bool(cfg.model.get("capture_stochastic", False)),
            "learning_rate": float(cfg.get("learning_rate", 0)),
            "max_clips_evaluated": max_clips,
        },
        "fm_chunk_eval": fm_chunk,
        "zero_baseline_fm_chunk": zb,
        "deploy": {
            "num_frames": deploy_frames,
            "deploy_seconds": deploy_seconds,
            "deploy_fps": deploy_fps,
            "native_fps": native_fps,
            "video_refresh_interval": refresh_iv,
            "start_frame": args.deploy_start_frame,
        },
        "deploy_fm_xperience": deploy,
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote report to {out}")


if __name__ == "__main__":
    main()
