#!/usr/bin/env python3
"""Demo: LangChain robot agent (official Qwen3-VL) + per-skill Phi0 routing."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logger = logging.getLogger(__name__)
DEFAULT_EP = 447


def _load_episode_images(episode_idx: int):
    from phi0.agent.frames import load_pick_tissue_episode_images

    return load_pick_tissue_episode_images(episode_idx)


def parse_args():
    p = argparse.ArgumentParser(description="Phi0 LangChain agent demo")
    p.add_argument("--user-instruction", type=str, default="你可以把沙发上的纸巾拿起来么？")
    p.add_argument("--episode-idx", type=int, default=DEFAULT_EP)
    p.add_argument("--image", type=str, default="")
    p.add_argument("--wrist-image", type=str, default="")
    p.add_argument("--pick-checkpoint", type=str, default="")
    p.add_argument("--throw-checkpoint", type=str, default="")
    p.add_argument("--config-name", type=str, default="train_pick_tissue_xperience_unified_ddp4_3k")
    p.add_argument("--vlm-model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from PIL import Image

    from phi0.agent import Phi0SkillRouter, build_robot_agent
    from phi0.models.vlm.tower import GenerateTextConfig
    from phi0.runtime import resolve_inference_device

    device = resolve_inference_device(args.device, min_free_gb=12.0)

    if args.image:
        ego = Image.open(args.image).convert("RGB")
        wrist = Image.open(args.wrist_image).convert("RGB") if args.wrist_image else None
        clip_row = -1
    else:
        ego, wrist, clip_row, task = _load_episode_images(int(args.episode_idx))
        logger.info("loaded ep%d clip_row=%d task=%r", args.episode_idx, clip_row, task)

    router = None
    dry_run = bool(args.dry_run)
    if not dry_run:
        router = Phi0SkillRouter.from_overrides(
            pick_checkpoint=args.pick_checkpoint or None,
            throw_checkpoint=args.throw_checkpoint or None,
            config_name=args.config_name,
            device=device,
        )

    agent = build_robot_agent(
        model_path=args.vlm_model,
        device=device,
        gen_cfg=GenerateTextConfig(max_new_tokens=int(args.max_new_tokens), do_sample=False),
        phi0_router=router,
    )
    result = agent.run(args.user_instruction, ego, wrist_image=wrist, dry_run=dry_run)

    print("\n=== Agent 回复 ===")
    print(result["output"])
    if result["tool_steps"]:
        print("\n=== 工具调用 ===")
        for step in result["tool_steps"]:
            print(f"  {step['tool']}: {step['result']}")
    print("\n=== JSON ===")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
