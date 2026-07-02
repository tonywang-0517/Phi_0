#!/usr/bin/env python3
"""Closed-loop rollout on a pick-tissue dataset episode -> outputs.npz (no ZMQ/camera)."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import tyro

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from phi0_sonic_closed_loop_zmq import run_closed_loop  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class EpisodeRolloutConfig:
    episode_idx: int
    """Unified dataset episode_index (e.g. 447)."""
    output: Path
    """outputs.npz path (observations.npz written alongside if record_dir set)."""
    checkpoint: str
    config_dir: Path = Path("configs")
    config_name: str = "train_pick_tissue_finetune_rtc_ddp4"
    prompt: str = "pick tissue"
    device: str = "cuda"
    min_free_gb: float = 12.0
    control_fps: float = 50.0
    inference_rate: float = 2.5
    hand_ramp_frames: int = 40
    start_control_idx: int = 0
    motion_seconds: float = 0.0
    """0 = full episode length."""


def _record_dir(output: Path) -> Path:
    return output.expanduser().resolve().parent


def main(config: EpisodeRolloutConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.chdir(ROOT)

    out_path = config.output.expanduser().resolve()
    record_dir = _record_dir(out_path)
    record_dir.mkdir(parents=True, exist_ok=True)

    proprio = "roll-forward"

    args = SimpleNamespace(
        checkpoint=str(config.checkpoint),
        config_dir=str(config.config_dir.expanduser().resolve()),
        config_name=config.config_name,
        device=config.device,
        min_free_gb=float(config.min_free_gb),
        prompt=config.prompt,
        camera_source="gt",
        camera_host="127.0.0.1",
        camera_port=5555,
        camera_wait_s=30.0,
        gt_camera_episode=int(config.episode_idx),
        gt_camera_start_idx=int(config.start_control_idx),
        zmq_host="127.0.0.1",
        zmq_port=5556,
        state_zmq_host="127.0.0.1",
        state_zmq_port=5557,
        control_fps=float(config.control_fps),
        inference_rate=float(config.inference_rate),
        motion_seconds=float(config.motion_seconds),
        hand_ramp_frames=int(config.hand_ramp_frames),
        start_delay_s=0.0,
        proprio_source="roll-forward",
        seed_proprio=True,
        wait_robot_proprio=False,
        wait_deploy_state=False,
        stream_now=False,
        no_zmq=True,
        deploy_keyboard=False,
        record_dir=str(record_dir),
        record_obs=str(record_dir / "observations.npz"),
        record_output=str(out_path),
    )

    logger.info(
        "episode rollout ep=%d -> %s (proprio=%s, infer=%.2fHz)",
        config.episode_idx,
        out_path,
        proprio,
        config.inference_rate,
    )
    run_closed_loop(args)


if __name__ == "__main__":
    main(tyro.cli(EpisodeRolloutConfig))
