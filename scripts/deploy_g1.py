#!/usr/bin/env python3
"""G1 deployment: FM chunk action inference (52×3 keypoints)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phi0.checkpoint_utils import merge_saved_cfg
from phi0.data.cosmos_video_size import cosmos_video_size_from_cfg, round_hw_to_multiple
from phi0.data.processor import Phi0Processor
from phi0.data.temporal_align import DEFAULT_DATASET_NATIVE_FPS
from phi0.inference.session import ActionInferenceSession, PromptEmbedCache
from phi0.runtime import (
    activate_cuda_device,
    apply_processor_stats_from_checkpoint,
    build_processor,
    create_phi0,
    resolve_inference_device,
    sync_model_action_norm,
)
from phi0.schema.draw_schema import unpack_action_for_viz, zero_unsupervised_action_dims_np

logger = logging.getLogger(__name__)

DEFAULT_EGODEX_MP4 = ROOT / "Isaac-GR00T/demo_data/egodex/test/add_remove_lid/0.mp4"
DEFAULT_XPERIENCE_MP4 = ROOT / "Isaac-GR00T/demo_data/xperience-10m-sample/stereo_left.mp4"
DEFAULT_CHECKPOINT = ROOT / "experiments/phi0_full/phi0_smoke.pt"


def parse_args():
    p = argparse.ArgumentParser(description="Phi_0 G1 deploy inference")
    p.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Phi_0 training checkpoint (.pt)")
    p.add_argument(
        "--fastwam-ckpt",
        type=str,
        default=None,
        help="Optional FastWAM MoT checkpoint loaded before Phi_0 weights",
    )
    p.add_argument("--config-dir", type=str, default=str(ROOT / "configs"))
    p.add_argument("--config-name", type=str, default="train_full")
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument(
        "--deploy-fps",
        type=float,
        default=None,
        help="Control Hz for native video frame mapping (default: cfg.data.control_fps or 20)",
    )
    p.add_argument(
        "--native-fps",
        type=float,
        default=None,
        help="Source video Hz (default: cfg.data.dataset_native_fps for xperience or 20)",
    )
    p.add_argument("--output", type=str, default="smplh_predictions.jsonl")
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
    p.add_argument("--input-image", type=str, default=None, help="Single RGB image path")
    p.add_argument(
        "--input-video",
        type=str,
        default=None,
        help="MP4 path; uses --frame-index as starting frame",
    )
    p.add_argument("--frame-index", type=int, default=0)
    p.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    p.add_argument("--prompt", type=str, default=None, help="Task instruction for text encoder")
    p.add_argument(
        "--tiled",
        action="store_true",
        help="Unused (reserved): Cosmos VAE does not use Wan tiled encode; flag kept for CLI compat",
    )
    p.add_argument(
        "--save-d-raw",
        action="store_true",
        help="Save full 256-d d_raw vector per frame (enables euler/tactile viz)",
    )
    return p.parse_args()


def _round_to_multiple(value: int, base: int = 16) -> int:
    return max(base, (value // base) * base)


def _load_rgb_frame(path: Path, frame_index: int, size_hw: tuple[int, int]) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        img = Image.open(path).convert("RGB")
        h, w = size_hw
        img = img.resize((w, h), resample=Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)

    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from {path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = size_hw
    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    return frame.astype(np.uint8)


def _resolve_input_path(args) -> tuple[Path, int]:
    if args.input_image:
        return Path(args.input_image), 0
    if args.input_video:
        return Path(args.input_video), int(args.frame_index)
    for candidate in (DEFAULT_EGODEX_MP4, DEFAULT_XPERIENCE_MP4):
        if candidate.exists():
            return candidate, int(args.frame_index)
    raise FileNotFoundError(
        "No input image/video found. Pass --input-image or --input-video, "
        "or download samples via scripts/download_samples.py"
    )


def _build_input_tensor(model, rgb: np.ndarray) -> torch.Tensor:
    h, w = rgb.shape[:2]
    if h % 16 != 0 or w % 16 != 0:
        h = _round_to_multiple(h)
        w = _round_to_multiple(w)
        pil = Image.fromarray(rgb, mode="RGB")
        rgb = np.asarray(pil.resize((w, h), resample=Image.BILINEAR), dtype=np.uint8)
    image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(
        device=model.device,
        dtype=model.torch_dtype,
    )
    return image_tensor * (2.0 / 255.0) - 1.0


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with initialize_config_dir(version_base="1.3", config_dir=args.config_dir):
        cfg = compose(config_name=args.config_name)
    device = resolve_inference_device(args.device, min_free_gb=float(args.min_free_gb))
    activate_cuda_device(device)
    cfg.device = device

    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_cfg = payload.get("cfg") if isinstance(payload, dict) else None
    if saved_cfg:
        cfg = merge_saved_cfg(cfg, saved_cfg)

    logger.info("Loading model on %s...", device)
    model = create_phi0(cfg, smoke=bool(cfg.get("smoke_action_only", False)))
    if args.fastwam_ckpt:
        model.load_checkpoint(args.fastwam_ckpt)
    if isinstance(payload, dict) and ("model" in payload or "action_expert" in payload):
        model.load_checkpoint(args.checkpoint)
    model.eval()

    processor = build_processor(cfg).eval()
    if isinstance(payload, dict):
        apply_processor_stats_from_checkpoint(processor, payload, cfg)
    sync_model_action_norm(model, processor)

    deploy_fps = float(
        args.deploy_fps if args.deploy_fps is not None else cfg.data.get("control_fps", 20.0)
    )
    if args.native_fps is not None:
        native_fps = float(args.native_fps)
    else:
        native_map = cfg.data.get("dataset_native_fps", DEFAULT_DATASET_NATIVE_FPS)
        native_fps = float(native_map.get("xperience", 20.0) if hasattr(native_map, "get") else 20.0)

    input_path, frame_index = _resolve_input_path(args)
    if args.image_size is not None:
        h, w = int(args.image_size[0]), int(args.image_size[1])
    else:
        h, w = cosmos_video_size_from_cfg(cfg.data)
    h, w = round_hw_to_multiple(h, w)
    rgb = _load_rgb_frame(input_path, frame_index, (h, w))
    input_image = _build_input_tensor(model, rgb)
    prompt = args.prompt or "human egocentric manipulation task"
    prompt_cache = PromptEmbedCache()

    logger.info("Cosmos hook capture (once) + FM chunk predict...")
    video_ratio = int(cfg.data.get("action_video_freq_ratio", 2))
    session = ActionInferenceSession(
        model,
        processor=processor,
        max_rgb_frames=max(33, args.num_frames),
        action_video_freq_ratio=video_ratio,
    )
    session.prefill_from_image(input_image, prompt, prompt_cache=prompt_cache)
    pred_chunk = session.predict(args.num_frames)
    out_path = Path(args.output)
    with out_path.open("w") as f:
        for t in tqdm(range(args.num_frames), desc="deploy FM", unit="frame", file=sys.stdout):
            pred_norm = pred_chunk[t]
            pred_d_raw = zero_unsupervised_action_dims_np(
                processor.postprocess(pred_norm.float().unsqueeze(0))
                .reshape(-1)
                .detach()
                .cpu()
                .numpy()
            )
            viz = unpack_action_for_viz(pred_d_raw)
            is_video = input_path.suffix.lower() == ".mp4"
            native_t = (
                frame_index + int(round(t * native_fps / deploy_fps))
                if is_video
                else frame_index
            )
            rec = {
                "frame": t,
                "control_fps": deploy_fps,
                "native_fps": native_fps if is_video else None,
                "source": str(input_path),
                "source_frame_index": native_t,
                "keypoints_52": viz["keypoints_52"].reshape(-1).tolist(),
            }
            if args.save_d_raw:
                rec["d_raw"] = pred_d_raw.tolist()
            f.write(json.dumps(rec) + "\n")
    print(f"Input: {input_path} frame={frame_index} ({h}x{w})")
    print(f"Wrote {args.num_frames} action frames to {out_path}")


if __name__ == "__main__":
    main()
