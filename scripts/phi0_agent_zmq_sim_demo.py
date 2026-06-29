#!/usr/bin/env python3
"""Agent -> skill -> SONIC latent ZMQ sim (ep447 ego+wrist). stay skips deploy."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

logger = logging.getLogger(__name__)
DEFAULT_EP = 447
ACTION_SKILLS = frozenset({"pick_tissues", "throw_rubbish"})


def _load_episode_images(episode_idx: int):
    from phi0.agent.frames import load_pick_tissue_episode_images

    ego, wrist, clip_row, _task = load_pick_tissue_episode_images(episode_idx)
    return ego, wrist, clip_row


def parse_args():
    p = argparse.ArgumentParser(description="Phi0 agent + SONIC latent sim demo")
    p.add_argument("--user-instruction", type=str, default="你可以把沙发上的纸巾拿起来么？")
    p.add_argument("--episode-idx", type=int, default=DEFAULT_EP)
    p.add_argument("--pick-checkpoint", type=str, default="")
    p.add_argument("--throw-checkpoint", type=str, default="")
    p.add_argument(
        "--config-name",
        type=str,
        default="train_pick_tissue_xperience_unified_ddp4_3k",
    )
    p.add_argument("--vlm-model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--motion-seconds", type=float, default=8.0)
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument(
        "--precompute-in",
        type=str,
        default="",
        help="Optional: reuse saved npz (default = inline VLM infer at publisher)",
    )
    p.add_argument("--skip-agent", action="store_true", help="Force skill (with --force-skill)")
    p.add_argument(
        "--force-skill",
        type=str,
        default="",
        choices=["", "pick_tissues", "throw_rubbish", "stay"],
    )
    return p.parse_args()


def _resolve_precompute_in(explicit: str) -> Path | None:
    """Only use npz when user explicitly passes --precompute-in / PRECOMPUTE_IN."""
    if not explicit.strip():
        return None
    path = Path(explicit).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"--precompute-in not found: {path}")
    return path


def _run_sonic_sim(
    *,
    skill: str,
    episode_idx: int,
    pick_checkpoint: str,
    throw_checkpoint: str,
    config_name: str,
    motion_seconds: float,
    out_dir: Path,
    precompute_in: str = "",
) -> Path:
    from phi0.agent.checkpoints import DEFAULT_SKILL_CHECKPOINTS, resolve_skill_checkpoint

    spec = DEFAULT_SKILL_CHECKPOINTS[skill]
    if skill == "pick_tissues" and pick_checkpoint:
        ckpt = Path(pick_checkpoint)
    elif skill == "throw_rubbish" and throw_checkpoint:
        ckpt = Path(throw_checkpoint)
    else:
        ckpt, _ = resolve_skill_checkpoint(spec, root=ROOT)

    out_mp4 = out_dir / f"agent_{skill}_ep{episode_idx}_sonic_latent_model.mp4"
    env = os.environ.copy()
    env.update(
        {
            "CHECKPOINT": str(ckpt),
            "CONFIG_NAME": config_name,
            "UNIFIED_EP": str(episode_idx),
            "MOTION_SECONDS": str(motion_seconds),
            "OUT_MP4": str(out_mp4),
            "WORK_DIR": str(out_dir),
            "GT_PANEL_LAYOUT": env.get("GT_PANEL_LAYOUT", "top"),
            "ENABLE_G1_DEBUG_OVERLAY": env.get("ENABLE_G1_DEBUG_OVERLAY", "0"),
        }
    )
    pc = _resolve_precompute_in(precompute_in)
    if pc is not None:
        env["PRECOMPUTE_IN"] = str(pc)
        logger.info("reuse precompute %s", pc)
    else:
        logger.info("SONIC publisher will inline-infer (no precompute npz)")
    script = ROOT / "scripts/run_pick_tissue_sonic_latent_eval.sh"
    logger.info("launch SONIC sim skill=%s ckpt=%s ep=%d", skill, ckpt, episode_idx)
    subprocess.run(["bash", str(script)], check=True, env=env, cwd=str(ROOT))
    return out_mp4


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from phi0.agent import Phi0SkillRouter, build_robot_agent
    from phi0.models.vlm.tower import GenerateTextConfig
    from phi0.runtime import resolve_inference_device

    device = resolve_inference_device(args.device, min_free_gb=12.0)
    ego, wrist, clip_row = _load_episode_images(int(args.episode_idx))
    logger.info("ep%d clip_row=%d", args.episode_idx, clip_row)

    ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else WORKSPACE / "logs" / f"agent_sonic_sim_{ts}"
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    skill: str | None = None
    if args.force_skill:
        skill = args.force_skill
        agent_result = {"selected_skill": skill, "output": f"(forced skill={skill})", "tool_steps": []}
    elif args.skip_agent:
        raise SystemExit("--skip-agent requires --force-skill")
    else:
        router = Phi0SkillRouter.from_overrides(
            pick_checkpoint=args.pick_checkpoint or None,
            throw_checkpoint=args.throw_checkpoint or None,
            config_name=args.config_name,
            device=device,
        )
        agent = build_robot_agent(
            model_path=args.vlm_model,
            device=device,
            gen_cfg=GenerateTextConfig(max_new_tokens=256, do_sample=False),
            phi0_router=router,
        )
        agent_result = agent.run(args.user_instruction, ego, wrist_image=wrist)
        skill = agent_result.get("selected_skill")
        if skill is None and agent_result.get("tool_steps"):
            logger.warning(
                "agent called %d tools; only single-tool actions drive Phi0",
                len(agent_result["tool_steps"]),
            )
        elif skill is None:
            logger.warning("agent did not call any tool -> skip Phi0 / SONIC")

    (out_dir / "agent_result.json").write_text(
        json.dumps(agent_result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print("=== Agent ===")
    print(agent_result.get("output", ""))
    print("selected_skill:", skill)
    if agent_result.get("tool_steps"):
        print("tool_steps:", json.dumps(agent_result["tool_steps"], ensure_ascii=False, default=str))

    if skill not in ACTION_SKILLS:
        if skill is None:
            print("agent 未选出可执行技能 -> 不启动 Phi0 / SONIC")
        else:
            print(f"skill={skill!r} -> skip SONIC sim (out_dir={out_dir})")
        return

    mp4 = _run_sonic_sim(
        skill=skill,
        episode_idx=int(args.episode_idx),
        pick_checkpoint=args.pick_checkpoint,
        throw_checkpoint=args.throw_checkpoint,
        config_name=args.config_name,
        motion_seconds=float(args.motion_seconds),
        out_dir=out_dir,
        precompute_in=args.precompute_in,
    )
    print(f"[done] video={mp4}")


if __name__ == "__main__":
    main()
