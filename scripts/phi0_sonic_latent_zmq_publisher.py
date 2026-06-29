#!/usr/bin/env python3
"""Publish Phi-0 predicted SONIC motion_token (+ gripper hands) over ZMQ pose v4."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zmq
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
_GR00T = ROOT.parent / "GR00T-WholeBodyControl"
for p in (ROOT / "src", ROOT, _GR00T):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gear_sonic.utils.teleop.zmq.v4_latent_replay import (  # noqa: E402
    prebuild_latent_action_messages,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message  # noqa: E402

from phi0.checkpoint_utils import merge_saved_cfg  # noqa: E402
from phi0.deploy.sonic_zmq_io import unified_action_denorm_to_zmq_arrays
from phi0.deploy.gt_io import (  # noqa: E402
    build_eval_clip_context,
    build_gt_backend,
    build_lazy_deploy_gt_norm_lut,
    is_pick_tissue_unified_cfg,
)
from phi0.deploy.pick_tissue_gt import clip_dataset_index_for_episode  # noqa: E402
from phi0.inference.session import (  # noqa: E402
    ClipInputsCache,
    PromptEmbedCache,
    resolve_deploy_action_chunk_size,
)
from phi0.runtime import (  # noqa: E402
    activate_cuda_device,
    apply_processor_stats_from_checkpoint,
    build_base_dataset,
    build_processor,
    create_phi0,
    resolve_inference_device,
    sync_model_action_norm,
)

# ponytail: reuse multi-chunk deploy inference from HGPT publisher
from experiments.phi0_hgpt_zmq.phi0_zmq_publisher import (  # noqa: E402
    _history_window,
    _predict_motion_deploy,
    _resolve_eval_dataset,
)

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Phi-0 -> ZMQ v4 SONIC latent publisher")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--config-dir", type=str, default=str(ROOT / "configs"))
    p.add_argument("--config-name", type=str, default="train_pick_tissue_xperience_unified_ddp4_8k")
    p.add_argument("--episode-idx", type=int, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--min-free-gb", type=float, default=12.0)
    p.add_argument("--zmq-host", type=str, default="127.0.0.1")
    p.add_argument("--zmq-port", type=int, default=5556)
    p.add_argument("--control-fps", type=float, default=50.0)
    p.add_argument("--motion-seconds", type=float, default=8.0)
    p.add_argument("--max-frames", type=int, default=0, help="0 = all motion_seconds frames")
    p.add_argument("--start-delay-s", type=float, default=0.5)
    p.add_argument("--arm-flag", type=str, default="")
    p.add_argument("--ready-flag", type=str, default="")
    p.add_argument("--arm-timeout-s", type=float, default=600.0)
    p.add_argument("--ready-timeout-s", type=float, default=900.0)
    p.add_argument("--hand-ramp-frames", type=int, default=40)
    p.add_argument(
        "--precompute-out",
        type=str,
        default="",
        help="Run offline inference and save tokens/hands npz, then exit.",
    )
    p.add_argument(
        "--precompute-in",
        type=str,
        default="",
        help="Skip model; stream motion from a prior --precompute-out npz.",
    )
    return p.parse_args()


def _wait_flag(path: Path, *, label: str, timeout_s: float = 240.0) -> None:
    logger.info("waiting for %s %s", label, path)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(0.2)
    raise TimeoutError(f"{label} flag not found: {path}")


def _send_deploy_start_commands(pub: zmq.Socket) -> None:
    pub.send(build_command_message(start=True, stop=False, planner=True))
    time.sleep(0.2)
    pub.send(build_command_message(start=True, stop=False, planner=False))
    time.sleep(0.2)
    logger.info("sent ZMQ command start (planner -> streamed motion)")


def _action_denorm_to_arrays(action_denorm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return unified_action_denorm_to_zmq_arrays(action_denorm)


def _save_precompute(
    path: Path,
    *,
    tokens: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    episode_idx: int,
    control_fps: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        tokens=tokens.astype(np.float32),
        left=left.astype(np.float32),
        right=right.astype(np.float32),
        episode_idx=int(episode_idx),
        control_fps=float(control_fps),
        num_frames=int(tokens.shape[0]),
    )
    logger.info("saved precompute %s (%d frames)", path, int(tokens.shape[0]))


def _load_precompute(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    data = np.load(path)
    tokens = np.asarray(data["tokens"], dtype=np.float32)
    left = np.asarray(data["left"], dtype=np.float32)
    right = np.asarray(data["right"], dtype=np.float32)
    episode_idx = int(data["episode_idx"])
    logger.info(
        "loaded precompute %s episode=%d frames=%d",
        path,
        episode_idx,
        int(tokens.shape[0]),
    )
    return tokens, left, right, episode_idx


@torch.no_grad()
def _run_model_inference(
    *,
    cfg,
    ckpt_path: Path,
    device: str,
    episode_idx: int,
    num_frames: int,
    control_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    activate_cuda_device(device)
    cfg.device = device

    logger.info("Loading checkpoint %s", ckpt_path)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and payload.get("cfg"):
        cfg = merge_saved_cfg(cfg, payload["cfg"])

    logger.info("building Phi-0 (VLM + action_expert)...")
    model = create_phi0(cfg, smoke=bool(cfg.get("smoke_action_only", False)))
    if isinstance(payload, dict) and ("model" in payload or "action_expert" in payload):
        model.load_checkpoint(str(ckpt_path))
    processor = build_processor(cfg).eval()
    if isinstance(payload, dict):
        apply_processor_stats_from_checkpoint(processor, payload, cfg)
    sync_model_action_norm(model, processor)
    model.eval()

    proprio_w = int(model.past_action_window_size)
    timeline_fps = float(control_fps)
    base = build_base_dataset(cfg)
    eval_ds, collate_fn = _resolve_eval_dataset(cfg, base)

    clip_row = episode_idx
    if is_pick_tissue_unified_cfg(cfg.data):
        clip_row = clip_dataset_index_for_episode(
            eval_ds, episode_idx, data_cfg=cfg.data
        )
        logger.info(
            "episode_index=%d -> clip_dataset_row=%d",
            episode_idx,
            clip_row,
        )

    clip_item = {
        "idx": episode_idx,
        "dataset": "g1_sonic",
        "control_fps": timeline_fps,
    }
    native_fps = float(getattr(eval_ds, "_fps", timeline_fps))
    gt_ctx = build_eval_clip_context(
        cfg.data,
        clip_item,
        hdf5_path="",
        native_fps=native_fps,
        control_fps=timeline_fps,
    )
    gt_backend = build_gt_backend(gt_ctx)
    chunk_h = resolve_deploy_action_chunk_size(model)
    history_w = _history_window(model)
    max_control = proprio_w + num_frames
    gt_norm_lut = build_lazy_deploy_gt_norm_lut(
        gt_backend,
        processor,
        num_frames=num_frames,
        proprio_w=proprio_w,
        chunk_h=chunk_h,
        history_w=history_w,
    )
    logger.info(
        "lazy GT proprio LUT: %d indices (full scan would load %d)",
        len(gt_norm_lut),
        max_control + 1,
    )

    logger.info("building clip inputs clip_row=%d", clip_row)
    inputs = ClipInputsCache().get_or_build(
        model,
        processor,
        eval_ds,
        clip_row,
        prompt_cache=PromptEmbedCache(),
        cache_action_context=True,
        collate_fn=collate_fn,
    )
    action_denorm = _predict_motion_deploy(
        model,
        processor,
        inputs,
        num_frames=num_frames,
        proprio_w=proprio_w,
        gt_norm_lut=gt_norm_lut,
    )
    logger.info(
        "inference done: %d frames x %d-d; GT LUT loaded %d/%d",
        num_frames,
        int(action_denorm.shape[-1]),
        gt_norm_lut.loaded_count(),
        len(gt_norm_lut),
    )
    return _action_denorm_to_arrays(action_denorm)


def _stream_over_zmq(
    *,
    tokens: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    episode_idx: int,
    zmq_host: str,
    zmq_port: int,
    control_fps: float,
    arm_flag: str,
    ready_flag: str,
    start_delay_s: float,
    hand_ramp_frames: int,
    arm_timeout_s: float = 600.0,
    ready_timeout_s: float = 900.0,
) -> None:
    num_frames = int(tokens.shape[0])
    messages = prebuild_latent_action_messages(
        tokens, left, right, hand_ramp_frames=int(hand_ramp_frames)
    )
    logger.info(
        "episode=%d frames=%d token[0]=%+.3f R_hand[0]=%+.3f",
        episode_idx,
        num_frames,
        float(tokens[0, 0]),
        float(right[0, 0]),
    )

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://{zmq_host}:{zmq_port}")
    time.sleep(0.5)
    logger.info("bound tcp://%s:%d", zmq_host, zmq_port)

    if arm_flag:
        _wait_flag(Path(arm_flag), label="arm", timeout_s=arm_timeout_s)
        time.sleep(start_delay_s)
        _send_deploy_start_commands(pub)

    if ready_flag:
        _wait_flag(Path(ready_flag), label="ready", timeout_s=ready_timeout_s)
        time.sleep(start_delay_s)
        _send_deploy_start_commands(pub)

    period = 1.0 / float(control_fps)
    for i, msg in enumerate(messages):
        t0 = time.monotonic()
        pub.send(msg)
        if i == 0 or (i + 1) % 100 == 0 or i + 1 == num_frames:
            logger.info(
                "frame %d/%d token[0]=%+.3f R_hand[0]=%+.3f",
                i + 1,
                num_frames,
                float(tokens[i, 0]),
                float(right[i, 0]),
            )
        rem = period - (time.monotonic() - t0)
        if rem > 0:
            time.sleep(rem)

    logger.info("done")
    pub.close()
    ctx.term()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # ponytail: VLM cfg uses ./checkpoints/... — must run from Phi_0 root (same as train.py).
    os.chdir(ROOT)
    logger.info("cwd=%s", Path.cwd())

    timeline_fps = float(args.control_fps)
    num_frames = max(1, int(round(float(args.motion_seconds) * timeline_fps)))
    if args.max_frames > 0:
        num_frames = min(num_frames, int(args.max_frames))
    episode_idx = int(args.episode_idx)

    precompute_in = Path(args.precompute_in).resolve() if args.precompute_in else None
    precompute_out = Path(args.precompute_out).resolve() if args.precompute_out else None

    if precompute_in is not None:
        tokens, left, right, episode_idx = _load_precompute(precompute_in)
    else:
        if not args.checkpoint:
            raise SystemExit("need --checkpoint or --precompute-in")
        ckpt_path = Path(args.checkpoint).resolve()
        device = resolve_inference_device(args.device, min_free_gb=float(args.min_free_gb))
        with initialize_config_dir(version_base="1.3", config_dir=args.config_dir):
            cfg = compose(config_name=args.config_name)
        tokens, left, right = _run_model_inference(
            cfg=cfg,
            ckpt_path=ckpt_path,
            device=device,
            episode_idx=episode_idx,
            num_frames=num_frames,
            control_fps=timeline_fps,
        )
        if precompute_out is not None:
            _save_precompute(
                precompute_out,
                tokens=tokens,
                left=left,
                right=right,
                episode_idx=episode_idx,
                control_fps=timeline_fps,
            )
            return

    _stream_over_zmq(
        tokens=tokens,
        left=left,
        right=right,
        episode_idx=episode_idx,
        zmq_host=args.zmq_host,
        zmq_port=args.zmq_port,
        control_fps=timeline_fps,
        arm_flag=args.arm_flag,
        ready_flag=args.ready_flag,
        start_delay_s=float(args.start_delay_s),
        hand_ramp_frames=int(args.hand_ramp_frames),
        arm_timeout_s=float(args.arm_timeout_s),
        ready_timeout_s=float(args.ready_timeout_s),
    )


if __name__ == "__main__":
    main()
