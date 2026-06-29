#!/usr/bin/env python3
"""Eval demo: pick-tissue clip + optional one-shot VLM agent speech (action path separate)."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phi0.checkpoint_utils import merge_saved_cfg
from phi0.deploy.pick_tissue_gt import clip_dataset_index_for_episode
from phi0.inference.session import ActionInferenceSession
from phi0.models.vlm.tower import GenerateTextConfig
from phi0.runtime import (
    activate_cuda_device,
    build_base_dataset,
    build_processor,
    create_phi0,
    prepare_model_batch_cpu,
    resolve_inference_device,
)

logger = logging.getLogger(__name__)

# ponytail: canonical smoke / regression clip for pick-tissue eval (manifest ep2)
DEFAULT_EVAL_EPISODE_IDX = 447


def _load_pick_tissue_clip(cfg, episode_idx: int):
    from phi0.data.pick_tissue_unified import PickTissueUnifiedClipDataset

    base = build_base_dataset(cfg)
    clip_row = clip_dataset_index_for_episode(base, episode_idx, data_cfg=cfg.data)
    item = base[clip_row]
    batch = PickTissueUnifiedClipDataset.collate_fn([item])
    return batch, str(batch["task"][0]), clip_row


def _frame_to_bcthw(frame: torch.Tensor, *, device, dtype) -> torch.Tensor:
    """[B,C,H,W] in [0,1] → [B,3,1,H,W] in [-1,1]."""
    f = frame.to(device=device, dtype=dtype)
    return (f * 2.0 - 1.0).unsqueeze(2)


def _obs_video_bcthw(cpu_payload, *, device, dtype) -> tuple[torch.Tensor, torch.Tensor | None]:
    obs = cpu_payload["obs_pixel"]  # [B,1,C,H,W] in [0,1]
    video = _frame_to_bcthw(obs[:, 0], device=device, dtype=dtype)
    wrist_video = None
    obs_wrist = cpu_payload.get("obs_wrist_pixel")
    if obs_wrist is not None:
        wrist_video = _frame_to_bcthw(obs_wrist[:, 0], device=device, dtype=dtype)
    return video, wrist_video


def parse_args():
    p = argparse.ArgumentParser(description="Phi-0 eval: pick-tissue clip + optional agent speech")
    p.add_argument("--config-dir", type=str, default=str(ROOT / "configs"))
    p.add_argument(
        "--config-name",
        type=str,
        default="train_pick_tissue_xperience_unified_ddp4_3k",
    )
    p.add_argument("--checkpoint", type=str, default="", help="Optional; VLM-only agent test skips action ckpt")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--min-free-gb", type=float, default=12.0)
    p.add_argument(
        "--episode-idx",
        type=int,
        default=DEFAULT_EVAL_EPISODE_IDX,
        help=f"pick_tissue_xperience_unified episode_index (default {DEFAULT_EVAL_EPISODE_IDX})",
    )
    p.add_argument("--instruction", type=str, default="", help="Override dataset task text")
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument(
        "--enable-agent-speech",
        action="store_true",
        help="One AR decode on first input snapshot (eval-only)",
    )
    p.add_argument(
        "--agent-speech-model-path",
        type=str,
        default="",
        help="Optional HF id/path for agent AR only (action still uses vlm.model_path Psi0)",
    )
    p.add_argument("--skip-action", action="store_true", help="Agent speech only; skip predict()")
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT.parent / "logs/pick_tissue_finetune"),
        help="Write agent_speech_ep{idx}_*.txt here",
    )
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = resolve_inference_device(args.device, min_free_gb=float(args.min_free_gb))
    activate_cuda_device(device)

    with initialize_config_dir(version_base="1.3", config_dir=args.config_dir):
        cfg = compose(config_name=args.config_name)
    cfg.device = device

    ckpt_path = Path(args.checkpoint).resolve() if args.checkpoint else None
    if ckpt_path and ckpt_path.is_file():
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(payload, dict) and payload.get("cfg"):
            cfg = merge_saved_cfg(cfg, payload["cfg"])

    model = create_phi0(cfg, smoke=bool(cfg.get("smoke_action_only", False)))
    if ckpt_path and ckpt_path.is_file():
        model.load_checkpoint(str(ckpt_path))
    model.eval()
    processor = build_processor(cfg).eval()

    episode_idx = int(args.episode_idx)
    batch, task, clip_row = _load_pick_tissue_clip(cfg, episode_idx)
    instruction = args.instruction.strip().lower() if args.instruction else task
    logger.info(
        "episode_idx=%d clip_row=%d task=%r instruction=%r",
        episode_idx,
        clip_row,
        task,
        instruction,
    )

    cpu_payload = prepare_model_batch_cpu(model, processor, batch)
    cpu_payload["sample"]["instruction"] = [instruction]

    session = ActionInferenceSession(
        model,
        processor,
        use_wrist_view=bool(getattr(processor, "use_wrist_view", False)),
        agent_speech_model_path=(
            args.agent_speech_model_path.strip()
            or str(getattr(getattr(cfg.model, "vlm", {}), "agent_speech_model_path", "") or "").strip()
            or None
        ),
    )
    if args.enable_agent_speech:
        session.enable_agent_speech_for_eval(True)

    video, wrist_video = _obs_video_bcthw(
        cpu_payload, device=model.device, dtype=model.torch_dtype
    )
    session.prefill_from_video_clip(
        video,
        instruction,
        wrist_video=wrist_video,
    )

    agent_text = ""
    if args.enable_agent_speech:
        gen_cfg = GenerateTextConfig(
            max_new_tokens=int(args.max_new_tokens),
            do_sample=bool(args.do_sample),
        )
        agent_text = session.run_agent_speech_once(gen_cfg=gen_cfg)
        repeat = session.run_agent_speech_once(gen_cfg=gen_cfg)
        assert repeat == agent_text, "agent speech must be one-shot per session"
        print(f"agent (once): {agent_text}")

    if not args.skip_action and ckpt_path and ckpt_path.is_file():
        dim = int(getattr(model.action_expert, "raw_action_dim", 512))
        session.seed_proprio_from_normalized(torch.zeros(dim, device=model.device))
        action = session.predict(int(args.num_frames))
        print(f"action shape: {tuple(action.shape)}")
    elif not args.skip_action and not ckpt_path:
        logger.info("no --checkpoint; skipped action predict (agent-only)")

    if args.enable_agent_speech and agent_text:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"agent_speech_ep{episode_idx}_{ts}.txt"
        out_path.write_text(
            "\n".join(
                [
                    f"episode_idx={episode_idx}",
                    f"clip_row={clip_row}",
                    f"task={task}",
                    f"instruction={instruction}",
                    f"agent_text={agent_text}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
