#!/usr/bin/env python3
"""Benchmark Phi_0 deploy inference speed + write JSONL for viz."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phi0.checkpoint_utils import merge_saved_cfg
from phi0.data.processor import Phi0Processor
from phi0.data.temporal_align import (
    DEFAULT_DATASET_NATIVE_FPS,
    native_span_frames,
)
from phi0.data.xperience import XperienceDataset
from phi0.inference.deploy_align import (
    build_deploy_video_tensor,
    control_step_to_native_frame,
    deploy_proprio_control_indices,
    deploy_subsampled_video_control_indices,
)
from phi0.inference.session import ActionInferenceSession, PromptEmbedCache, resolve_deploy_action_chunk_size
from phi0.runtime import (
    activate_cuda_device,
    apply_processor_stats_from_checkpoint,
    create_phi0,
    resolve_inference_device,
    sync_model_action_norm,
)
from phi0.schema.draw_schema import unpack_action_for_viz, zero_unsupervised_action_dims_np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(ROOT / "experiments/phi0_full/phi0_full_latest.pt"))
    p.add_argument("--config-name", type=str, default="train_full")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--deploy-seconds", type=float, default=5.0)
    p.add_argument(
        "--deploy-fps",
        type=float,
        default=None,
        help="Control Hz (default: cfg.data.control_fps or 20)",
    )
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument(
        "--video-refresh-interval",
        type=int,
        default=4,
        help="Forward video tower every N control steps to refresh latent/action context",
    )
    p.add_argument(
        "--input-video",
        type=str,
        default="/mnt/data2/wpy/workspace/Isaac-GR00T/demo_data/xperience-10m-sample/stereo_left.mp4",
    )
    p.add_argument("--prompt", type=str, default="Making pour-over coffee")
    p.add_argument("--output", type=str, default=str(ROOT / "experiments/phi0_full/smplh_predictions_2k.jsonl"))
    p.add_argument(
        "--benchmark-json",
        type=str,
        default=None,
        help="Defaults to <output-dir>/inference_benchmark.json",
    )
    p.add_argument(
        "--action-chunk-size",
        type=int,
        default=None,
        help="Deploy predict horizon per forward (default: seq_len - past_action_window_size)",
    )
    return p.parse_args()


def _resolve_native_fps(cfg, explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    native_map = cfg.data.get("dataset_native_fps", DEFAULT_DATASET_NATIVE_FPS)
    if hasattr(native_map, "get"):
        return float(native_map.get("xperience", 20.0))
    return float(DEFAULT_DATASET_NATIVE_FPS["xperience"])


def _control_to_native_frame(control_t: int, start_frame: int, deploy_fps: float, native_fps: float) -> int:
    return control_step_to_native_frame(control_t, start_frame, deploy_fps, native_fps)


def main():
    args = parse_args()
    device = resolve_inference_device(args.device)
    activate_cuda_device(device)

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input_video}")
    cap.release()

    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name=args.config_name)

    ckpt = Path(args.checkpoint)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "cfg" in payload:
        cfg = merge_saved_cfg(cfg, payload["cfg"])
    cfg.device = device

    deploy_fps = float(
        args.deploy_fps if args.deploy_fps is not None else cfg.data.get("control_fps", 20.0)
    )
    native_fps = _resolve_native_fps(cfg, None)
    video_ratio = int(cfg.data.get("action_video_freq_ratio", 2))
    seq_len = int(cfg.data.get("seq_len", 33))
    num_frames = int(args.num_frames) if args.num_frames is not None else max(
        1, int(round(float(args.deploy_seconds) * deploy_fps))
    )
    refresh_every = max(1, int(args.video_refresh_interval))
    start_frame = int(args.start_frame)
    video_ctrl_idx = deploy_subsampled_video_control_indices(
        num_frames - 1,
        seq_len=seq_len,
        action_video_freq_ratio=video_ratio,
    )
    native_span = native_span_frames(num_frames, deploy_fps, native_fps)

    t0 = time.perf_counter()
    model = create_phi0(cfg, smoke=False)
    if isinstance(payload, dict) and ("model" in payload or "action_expert" in payload):
        model.load_checkpoint(str(ckpt))
    model.eval()
    load_s = time.perf_counter() - t0

    processor = Phi0Processor(normalize=True).eval()
    if isinstance(payload, dict):
        apply_processor_stats_from_checkpoint(processor, payload, cfg)
    sync_model_action_norm(model, processor)

    xp = XperienceDataset(
        max_frames=start_frame + native_span,
        start_frame=0,
        cache_video=True,
    )

    session = ActionInferenceSession(
        model,
        processor=processor,
        deploy_seq_len=seq_len,
        action_video_freq_ratio=video_ratio,
        use_gt_proprio=True,
    )
    action_chunk = (
        int(args.action_chunk_size)
        if args.action_chunk_size is not None
        else resolve_deploy_action_chunk_size(model, seq_len=seq_len)
    )
    proprio_w = int(getattr(model, "past_action_window_size", 0))
    prompt_cache = PromptEmbedCache()

    def _read_chw(control_t: int) -> torch.Tensor:
        native_t = _control_to_native_frame(control_t, start_frame, deploy_fps, native_fps)
        if xp._video_frames is not None and 0 <= native_t < len(xp._video_frames):
            return xp._video_frames[native_t]
        cap = cv2.VideoCapture(args.input_video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(native_t))
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Cannot read native frame {native_t} from {args.input_video}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (640, 480))
        return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    def _gt_action_norm(control_t: int) -> torch.Tensor:
        native_t = _control_to_native_frame(control_t, start_frame, deploy_fps, native_fps)
        d_raw = xp._load_frame_action(native_t)
        return processor._normalize_action(torch.from_numpy(d_raw).float()).squeeze(0)

    def _deploy_video_clip(seg_start: int) -> torch.Tensor:
        return build_deploy_video_tensor(
            seg_start,
            _read_chw,
            seq_len=seq_len,
            action_video_freq_ratio=video_ratio,
            past_window=proprio_w,
            device=model.device,
            dtype=model.torch_dtype,
        )

    prefill_native = _control_to_native_frame(0, start_frame, deploy_fps, native_fps)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    session.prefill_from_video_clip(
        _deploy_video_clip(0), args.prompt, prompt_cache=prompt_cache,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t1

    step_times: list[float] = []
    refresh_times: list[float] = []
    preds: list[np.ndarray] = []
    for seg_start in range(0, num_frames, action_chunk):
        if seg_start > 0:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            tr = time.perf_counter()
            session.refresh_video_context_from_clip(_deploy_video_clip(seg_start))
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            refresh_times.append(time.perf_counter() - tr)
        if proprio_w > 0:
            proprio_ctrl = deploy_proprio_control_indices(seg_start, proprio_w)
            session.set_proprio_gt(torch.stack([_gt_action_norm(c) for c in proprio_ctrl], dim=0))
        chunk_len = min(action_chunk, num_frames - seg_start)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ts = time.perf_counter()
        chunk = session.predict(chunk_len)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        chunk_ms = (time.perf_counter() - ts) * 1000.0 / chunk_len
        for i in range(chunk_len):
            step_times.append(chunk_ms / 1000.0)
            pred_d_raw = zero_unsupervised_action_dims_np(
                processor.postprocess(chunk[i].float().unsqueeze(0)).reshape(-1).detach().cpu().numpy()
            )
            preds.append(pred_d_raw)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for t, pred_d_raw in enumerate(preds):
            native_t = _control_to_native_frame(t, start_frame, deploy_fps, native_fps)
            viz = unpack_action_for_viz(pred_d_raw)
            rec = {
                "frame": t,
                "control_fps": deploy_fps,
                "native_fps": native_fps,
                "source": args.input_video,
                "source_frame_index": native_t,
                "start_frame": start_frame,
                "d_raw": pred_d_raw.tolist(),
                "keypoints_52": viz["keypoints_52"].reshape(-1).tolist(),
            }
            f.write(json.dumps(rec) + "\n")

    step_ms = np.array(step_times) * 1000.0
    bench = {
        "checkpoint": str(ckpt),
        "checkpoint_step": int(payload.get("step", -1)) if isinstance(payload, dict) else -1,
        "device": str(device),
        "num_deploy_frames": num_frames,
        "deploy_seconds": round(num_frames / deploy_fps, 3),
        "deploy_fps": deploy_fps,
        "native_fps": native_fps,
        "native_span_frames": native_span,
        "action_video_freq_ratio": video_ratio,
        "video_control_indices": video_ctrl_idx,
        "start_frame": start_frame,
        "action_chunk_size": action_chunk,
        "deploy_seq_len": seq_len,
        "deploy_video_clip_frames": len(
            deploy_subsampled_video_control_indices(0, seq_len=seq_len, action_video_freq_ratio=video_ratio)
        ),
        "use_gt_proprio": True,
        "proprio_window": proprio_w,
        "video_refresh_interval_frames": refresh_every,
        "deploy_align": "training_clip_gt_proprio",
        "video_tower_refresh_count": session.video_refresh_count,
        "video_refresh_ms_mean": round(float(np.mean(refresh_times)) * 1000.0, 2) if refresh_times else 0.0,
        "model_load_s": round(load_s, 2),
        "prefill_s": round(prefill_s, 3),
        "prefill_ms": round(prefill_s * 1000.0, 1),
        "prefill_native_frame": prefill_native,
        "ar_step_ms_mean": round(float(step_ms.mean()), 2),
        "ar_step_ms_median": round(float(np.median(step_ms)), 2),
        "ar_step_ms_p95": round(float(np.percentile(step_ms, 95)), 2),
        "ar_step_ms_min": round(float(step_ms.min()), 2),
        "ar_step_ms_max": round(float(step_ms.max()), 2),
        "ar_fps_mean": round(1000.0 / float(step_ms.mean()), 2),
        "total_infer_s": round(prefill_s + float(step_ms.sum()) / 1000.0, 3),
        "predictions_jsonl": str(out_path),
        "temporal_align": {
            "control_to_native": "native_t = start + round(t * native_fps / deploy_fps)",
            "dataset_native_fps": OmegaConf.to_container(
                cfg.data.get("dataset_native_fps", DEFAULT_DATASET_NATIVE_FPS),
                resolve=True,
            ),
        },
    }
    bench_path = Path(args.benchmark_json) if args.benchmark_json else out_path.parent / "inference_benchmark.json"
    bench_path.parent.mkdir(parents=True, exist_ok=True)
    bench_path.write_text(json.dumps(bench, indent=2))
    print(json.dumps(bench, indent=2))


if __name__ == "__main__":
    main()
