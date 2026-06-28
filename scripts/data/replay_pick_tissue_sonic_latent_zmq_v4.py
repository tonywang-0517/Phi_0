#!/usr/bin/env python3
"""Replay pick-tissue GT SONIC motion_token (+ hands) over ZMQ pose v4 only."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import tyro
import zmq

_PHI0 = Path(__file__).resolve().parents[2]
_GR00T = _PHI0.parent / "GR00T-WholeBodyControl"
for root in (_GR00T, _PHI0 / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message  # noqa: E402

from phi0.deploy.sonic_latent_gt_replay import (  # noqa: E402
    build_replay_messages,
    load_sonic_latent_replay_arrays,
)


@dataclass
class ReplayConfig:
    parquet: Path
    token_source: str = "auto"
    """auto | valid_column | unified_slice"""
    valid_parquet_for_hands: Path | None = None
    """When parquet is unified-only, load teleop hands from valid GR00T parquet."""
    zmq_host: str = "127.0.0.1"
    zmq_port: int = 5556
    fps: float = 50.0
    max_frames: int | None = None
    start_delay_s: float = 0.5
    ready_flag: str = ""
    """Wait for this flag before streaming pose frames."""
    arm_flag: str = ""
    """Wait for this flag, then send ZMQ command start (zmq_manager ignores keyboard ']')."""
    hand_ramp_frames: int = 40


def _wait_flag(path: Path, *, label: str, timeout_s: float = 240.0) -> None:
    print(f"[pick_tissue_sonic_latent] waiting for {label} {path}")
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
    print("[pick_tissue_sonic_latent] sent ZMQ command start (planner -> streamed motion)")


def main(config: ReplayConfig) -> None:
    tokens, left, right, token_source = load_sonic_latent_replay_arrays(
        config.parquet,
        token_source=config.token_source,
        valid_parquet_for_hands=config.valid_parquet_for_hands,
        max_frames=config.max_frames,
    )
    n = len(tokens)
    messages = build_replay_messages(
        tokens, left, right, hand_ramp_frames=config.hand_ramp_frames
    )

    print(
        f"[pick_tissue_sonic_latent] parquet={config.parquet.name} frames={n} "
        f"token_source={token_source}"
    )

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://{config.zmq_host}:{config.zmq_port}")
    time.sleep(0.5)
    print(f"[pick_tissue_sonic_latent] bound tcp://{config.zmq_host}:{config.zmq_port}")

    if config.arm_flag:
        _wait_flag(Path(config.arm_flag), label="arm")
        time.sleep(config.start_delay_s)
        _send_deploy_start_commands(pub)

    if config.ready_flag:
        _wait_flag(Path(config.ready_flag), label="ready")
        time.sleep(config.start_delay_s)
        _send_deploy_start_commands(pub)

    period = 1.0 / config.fps
    for i, msg in enumerate(messages):
        t0 = time.monotonic()
        pub.send(msg)
        if i == 0 or (i + 1) % 100 == 0 or i + 1 == n:
            print(
                f"[pick_tissue_sonic_latent] frame {i + 1}/{n} "
                f"token[0]={tokens[i][0]:+.3f} R_hand[0]={right[i][0]:+.3f}"
            )
        elapsed = time.monotonic() - t0
        rem = period - elapsed
        if rem > 0:
            time.sleep(rem)

    print("[pick_tissue_sonic_latent] done")


if __name__ == "__main__":
    main(tyro.cli(ReplayConfig))
