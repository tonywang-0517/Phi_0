#!/usr/bin/env python3
"""Offline closed-loop rollout from a saved observations.npz (live run replay)."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import tyro

ROOT = Path(__file__).resolve().parents[1]
_GR00T = Path(
    os.environ.get(
        "GR00T_ROOT",
        str(Path.home() / "YZY" / "GR00T-WholeBodyControl"),
    )
).expanduser().resolve()
for p in (ROOT / "src", ROOT, _GR00T, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gear_sonic.utils.inference.vla_utils import calculate_latency_compensated_index  # noqa: E402
from gear_sonic.utils.teleop.zmq.v4_latent_replay import hand_ramp_weights  # noqa: E402

from phi0.deploy.sonic_zmq_io import unified_action_denorm_to_zmq_arrays  # noqa: E402
from phi0.inference.session import ActionInferenceSession  # noqa: E402
from phi0_sonic_closed_loop_zmq import (  # noqa: E402
    ActionChunk,
    ClosedLoopRecorder,
    _frame_to_bcthw,
    _load_model_bundle,
    _rgb_to_chw,
)

logger = logging.getLogger(__name__)


@dataclass
class RecordedObs:
    ego: np.ndarray
    wrist: np.ndarray | None
    control_idx: np.ndarray
    timestamp: np.ndarray


def load_recorded_observations(path: Path) -> RecordedObs:
    data = np.load(path)
    for key in ("ego", "control_idx", "timestamp"):
        if key not in data.files:
            raise KeyError(f"{path} missing {key!r}; have {list(data.files)}")
    ego = np.asarray(data["ego"], dtype=np.uint8)
    control_idx = np.asarray(data["control_idx"], dtype=np.int32).reshape(-1)
    timestamp = np.asarray(data["timestamp"], dtype=np.float64).reshape(-1)
    wrist = np.asarray(data["wrist"], dtype=np.uint8) if "wrist" in data.files else None
    n = int(ego.shape[0])
    if control_idx.shape[0] != n or timestamp.shape[0] != n:
        raise ValueError(
            f"length mismatch ego={n} control_idx={control_idx.shape[0]} "
            f"timestamp={timestamp.shape[0]}"
        )
    if wrist is not None and wrist.shape[0] != n:
        raise ValueError(f"wrist rows {wrist.shape[0]} != ego rows {n}")
    return RecordedObs(ego=ego, wrist=wrist, control_idx=control_idx, timestamp=timestamp)


def build_inference_schedule(control_idx: np.ndarray) -> list[tuple[int, int]]:
    """(control_frame, observation_row) pairs in timeline order."""
    order = np.argsort(control_idx, kind="stable")
    return [(int(control_idx[i]), int(i)) for i in order]


def resolve_max_frames(
    *,
    control_idx: np.ndarray,
    chunk_h: int,
    max_frames: int,
    reference_output: Path | None,
) -> int:
    if max_frames > 0:
        return int(max_frames)
    if reference_output is not None and reference_output.is_file():
        ref = np.load(reference_output)
        if "num_frames" in ref.files:
            return int(ref["num_frames"])
        if "tokens" in ref.files:
            return int(ref["tokens"].shape[0])
    return int(control_idx[-1]) + int(chunk_h)


def _predict_from_hwc(
    *,
    session: ActionInferenceSession,
    model,
    processor,
    ego_hwc: np.ndarray,
    wrist_hwc: np.ndarray | None,
    prompt: str,
    chunk_h: int,
) -> ActionChunk:
    ego_chw = _rgb_to_chw(ego_hwc)
    wrist_chw = _rgb_to_chw(wrist_hwc) if wrist_hwc is not None else None
    ego_clip = _frame_to_bcthw(ego_chw, device=model.device, dtype=model.torch_dtype)
    wrist_clip = None
    if processor.use_wrist_view:
        if wrist_chw is None:
            wrist_chw = ego_chw
            logger.warning("wrist missing; reusing ego frame")
        wrist_clip = _frame_to_bcthw(wrist_chw, device=model.device, dtype=model.torch_dtype)

    if session.action_ctx is None:
        session.prefill_from_video_clip(ego_clip, prompt, wrist_video=wrist_clip)
    else:
        session.refresh_video_context_from_clip(
            ego_clip,
            prompt=prompt,
            wrist_video=wrist_clip,
        )

    use_amp = model.device.type == "cuda"
    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=use_amp):
        pred = session.predict(int(chunk_h), denormalize=True)
    action_denorm = pred.float().detach().cpu().numpy()
    tokens, left, right = unified_action_denorm_to_zmq_arrays(action_denorm)
    return ActionChunk(
        tokens=tokens,
        left=left,
        right=right,
        horizon=int(tokens.shape[0]),
    )


def rollout_offline(
    *,
    recorded: RecordedObs,
    session: ActionInferenceSession,
    model,
    processor,
    prompt: str,
    chunk_h: int,
    control_fps: float,
    hand_ramp_frames: int,
    inference_latency_s: float,
    max_frames: int,
    recorder: ClosedLoopRecorder,
) -> int:
    schedule = build_inference_schedule(recorded.control_idx)
    if not schedule:
        raise ValueError("no observations to rollout")

    cached_chunk: ActionChunk | None = None
    action_chunk_index = 0
    schedule_ptr = 0
    total_frames = 0
    t0 = time.monotonic()

    for ctrl in range(int(max_frames)):
        while schedule_ptr < len(schedule) and schedule[schedule_ptr][0] == ctrl:
            _, obs_i = schedule[schedule_ptr]
            schedule_ptr += 1
            infer_t0 = time.monotonic()
            cached_chunk = _predict_from_hwc(
                session=session,
                model=model,
                processor=processor,
                ego_hwc=recorded.ego[obs_i],
                wrist_hwc=None if recorded.wrist is None else recorded.wrist[obs_i],
                prompt=prompt,
                chunk_h=chunk_h,
            )
            delay = float(inference_latency_s)
            if inference_latency_s < 0:
                delay = max(0.0, time.monotonic() - infer_t0)
            action_chunk_index = calculate_latency_compensated_index(
                delay,
                float(control_fps),
                int(chunk_h),
            )
            logger.info(
                "infer obs=%d ctrl=%d chunk_i=%d token0=%+.3f",
                obs_i,
                ctrl,
                action_chunk_index,
                float(cached_chunk.tokens[action_chunk_index, 0]),
            )

        if cached_chunk is None:
            continue

        idx = min(action_chunk_index, cached_chunk.horizon - 1)
        ramp = hand_ramp_weights(total_frames + 1, int(hand_ramp_frames))[-1]
        tok = cached_chunk.tokens[idx]
        lh = cached_chunk.left[idx] * ramp
        rh = cached_chunk.right[idx] * ramp
        recorder.record_output(
            token=tok,
            left=lh,
            right=rh,
            control_idx=ctrl,
            chunk_idx=idx,
            frame_index=total_frames,
            hand_ramp=float(ramp),
            timestamp=t0 + ctrl / float(control_fps),
        )
        total_frames += 1
        action_chunk_index = min(action_chunk_index + 1, cached_chunk.horizon - 1)

    return total_frames


@dataclass
class OfflineRolloutConfig:
    observations: Path
    output: Path
    record_dir: Path | None = None
    checkpoint: str = ""
    config_dir: Path = Path("configs")
    config_name: str = "train_pick_tissue_xperience_unified_ddp4_3k"
    prompt: str = ""
    device: str = "cuda"
    min_free_gb: float = 12.0
    control_fps: float = 0.0
    hand_ramp_frames: int = 40
    max_frames: int = 0
    reference_output: Path | None = None
    inference_latency_s: float = 0.0
    """Simulated infer->stream delay (s). <0 = measure per inference."""


def _load_meta(record_dir: Path) -> dict:
    meta_path = record_dir / "record_meta.json"
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _build_args(cfg: OfflineRolloutConfig, meta: dict) -> SimpleNamespace:
    checkpoint = cfg.checkpoint.strip() or str(meta.get("checkpoint", "")).strip()
    if not checkpoint:
        raise ValueError("checkpoint required (--checkpoint or record_meta.json)")
    prompt = cfg.prompt.strip() or str(meta.get("prompt", "pick tissue"))
    return SimpleNamespace(
        checkpoint=checkpoint,
        config_dir=str(cfg.config_dir.expanduser().resolve()),
        config_name=cfg.config_name,
        device=cfg.device,
        min_free_gb=float(cfg.min_free_gb),
        prompt=prompt,
        proprio_source="roll-forward",
        seed_proprio=True,
    )


def main(config: OfflineRolloutConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.chdir(ROOT)

    obs_path = config.observations.expanduser().resolve()
    out_path = config.output.expanduser().resolve()
    record_dir = (
        config.record_dir.expanduser().resolve()
        if config.record_dir is not None
        else obs_path.parent
    )
    meta = _load_meta(record_dir)
    recorded = load_recorded_observations(obs_path)
    args = _build_args(config, meta)

    model, processor, session, prompt, chunk_h, _cfg, _proprio_source = _load_model_bundle(args)
    control_fps = float(config.control_fps or meta.get("control_fps", 50.0))

    recorder = ClosedLoopRecorder(
        prompt=prompt,
        camera_source="recorded",
        control_fps=control_fps,
        checkpoint=str(args.checkpoint),
    )
    ref_out = config.reference_output
    if ref_out is None:
        ref_out = record_dir / "outputs.npz"
    max_frames = resolve_max_frames(
        control_idx=recorded.control_idx,
        chunk_h=chunk_h,
        max_frames=int(config.max_frames),
        reference_output=ref_out if ref_out.is_file() else None,
    )
    logger.info(
        "offline rollout obs=%s inferences=%d max_frames=%d fps=%.1f chunk_h=%d",
        obs_path.name,
        len(recorded.control_idx),
        max_frames,
        control_fps,
        chunk_h,
    )

    n = rollout_offline(
        recorded=recorded,
        session=session,
        model=model,
        processor=processor,
        prompt=prompt,
        chunk_h=chunk_h,
        control_fps=control_fps,
        hand_ramp_frames=int(config.hand_ramp_frames),
        inference_latency_s=float(config.inference_latency_s),
        max_frames=max_frames,
        recorder=recorder,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # ponytail: skip rewriting input observations.npz (recorder has no obs rows)
    recorder.save(obs_path, out_path)
    logger.info("saved rollout outputs %s (%d frames)", out_path, n)


if __name__ == "__main__":
  main(tyro.cli(OfflineRolloutConfig))
