#!/usr/bin/env python3
"""Open-loop evaluation for Phi_0 on SIMPLE G1 LeRobot clips."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from phi0.checkpoint_utils import merge_saved_cfg
from phi0.data.simple_action_norm import SIMPLE_G1_DIM
from phi0.data.simple_lerobot import SimpleG1ClipDataset
from phi0.inference.session import ActionInferenceSession
from phi0.models.vlm.preprocess import normalize_vlm_instruction
from phi0.runtime import (
    activate_cuda_device,
    apply_processor_stats_from_checkpoint,
    build_processor,
    create_phi0,
    resolve_inference_device,
    sync_model_action_norm,
)

logger = logging.getLogger(__name__)

PRINT_SPLITS = [14, 28, 31, 32, 33, 34, 35]
LABELS = [
    "hand_joints",
    "arm_joints",
    "torso_roll",
    "torso_pitch",
    "torso_yaw",
    "height",
    "vx",
    "vy",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phi_0 SIMPLE G1 open-loop eval")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config-dir", default=str(ROOT / "configs"))
    p.add_argument("--config-name", default="train_simple_g1_act")
    p.add_argument("--data-root", default=str(ROOT / "data/simple"))
    p.add_argument("--repo-id", default="G1WholebodyBendPick-v0-psi0")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--min-free-gb", type=float, default=8.0)
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--stride", type=int, default=1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = resolve_inference_device(args.device, min_free_gb=args.min_free_gb)
    activate_cuda_device(device)

    with initialize_config_dir(version_base="1.3", config_dir=args.config_dir):
        cfg = compose(config_name=args.config_name)
    cfg.device = device
    cfg.data.simple_root = args.data_root
    cfg.data.simple_repo_id = args.repo_id

    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_cfg = payload.get("cfg") if isinstance(payload, dict) else None
    if saved_cfg:
        cfg = merge_saved_cfg(cfg, saved_cfg)

    model = create_phi0(cfg)
    if isinstance(payload, dict) and ("model" in payload or "action_expert" in payload):
        model.load_checkpoint(args.checkpoint)
    model.eval()

    processor = build_processor(cfg).eval()
    if isinstance(payload, dict):
        apply_processor_stats_from_checkpoint(processor, payload, cfg)
    sync_model_action_norm(model, processor)

    dataset = SimpleG1ClipDataset(
        root_dir=args.data_root,
        repo_id=args.repo_id,
        future_action_steps=int(cfg.data.get("future_action_steps", 30)),
        image_size=tuple(cfg.model.vlm.image_size),
        val=False,
    )
    session = ActionInferenceSession(model, processor=processor)
    future_steps = int(cfg.data.get("future_action_steps", 30))
    errors = []

    indices = list(range(0, min(len(dataset), args.max_samples), max(1, args.stride)))
    for idx in tqdm(indices, desc="open-loop"):
        sample = dataset[idx]
        rgb = (
            (sample["images"]["ego_view"][0, 0].permute(1, 2, 0).numpy() * 255.0)
            .clip(0, 255)
            .astype(np.uint8)
        )
        h, w = rgb.shape[:2]
        target_h, target_w = processor.vlm_image_size
        if (h, w) != (target_h, target_w):
            from PIL import Image

            rgb = np.asarray(
                Image.fromarray(rgb).resize((target_w, target_h), Image.BILINEAR),
                dtype=np.uint8,
            )
        image_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        image_t = (image_t * 2.0 - 1.0).unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        instruction = normalize_vlm_instruction(sample["task"])
        proprio = processor.normalize_robot_nd_tensor(
            sample["robot_proprio_36d"].float(), dim=SIMPLE_G1_DIM, proprio=True
        ).to(device=model.device, dtype=model.torch_dtype)

        session.reset()
        session.prefill_from_image(image_t, instruction)
        session.set_proprio_gt(proprio)
        pred_norm = session.predict(future_steps, denormalize=False)
        if pred_norm.ndim == 3:
            pred_norm = pred_norm.squeeze(0)
        pred = processor.denormalize_robot_nd_future(
            pred_norm.unsqueeze(0), dim=SIMPLE_G1_DIM
        ).squeeze(0)
        gt = sample["robot_future_36d"].float()
        if gt.ndim == 3:
            gt = gt.squeeze(0)
        err = (pred.detach().cpu() - gt).abs().mean(dim=0).numpy()
        errors.append(err)

    mean_err = np.stack(errors, axis=0).mean(axis=0)
    print("\n--- mean per-dim L1 ---")
    for label, seg in zip(LABELS, np.split(mean_err, PRINT_SPLITS)):
        print(f"{label:16s} norm={np.linalg.norm(seg):.6f}")
    print(f"overall mean L1={mean_err.mean():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
