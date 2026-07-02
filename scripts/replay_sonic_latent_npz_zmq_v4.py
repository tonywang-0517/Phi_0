#!/usr/bin/env python3
"""Replay saved SONIC latent npz (tokens + hands) over ZMQ pose v4 in sim/deploy."""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import zmq

_PHI0 = Path(__file__).resolve().parents[1]
_GR00T = Path(
    __import__("os").environ.get(
        "GR00T_ROOT",
        str(Path.home() / "YZY" / "GR00T-WholeBodyControl"),
    )
).expanduser().resolve()
for root in (_GR00T, _PHI0 / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from gear_sonic.utils.teleop.zmq.v4_latent_replay import prebuild_latent_action_messages  # noqa: E402
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message  # noqa: E402

logger = logging.getLogger(__name__)


def load_motion_npz(path: Path, *, max_frames: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    for key in ("tokens", "left", "right"):
        if key not in data.files:
            raise KeyError(f"{path} missing {key!r}; have {list(data.files)}")
    tokens = np.asarray(data["tokens"], dtype=np.float32)
    left = np.asarray(data["left"], dtype=np.float32)
    right = np.asarray(data["right"], dtype=np.float32)
    if tokens.ndim != 2 or tokens.shape[1] != 64:
        raise ValueError(f"tokens must be (T, 64), got {tokens.shape}")
    n = int(tokens.shape[0])
    if left.shape[0] != n or right.shape[0] != n:
        raise ValueError(
            f"length mismatch tokens={n} left={left.shape} right={right.shape}"
        )
    if max_frames > 0:
        n = min(n, int(max_frames))
        tokens, left, right = tokens[:n], left[:n], right[:n]
    return tokens, left, right


@dataclass
class NpzReplayConfig:
    npz: Path
    zmq_host: str = "127.0.0.1"
    zmq_port: int = 5556
    fps: float = 50.0
    max_frames: int = 0
    start_delay_s: float = 0.5
    arm_flag: str = ""
    ready_flag: str = ""
    hand_ramp_frames: int = 0
    """Use 0 for closed-loop outputs.npz (hands already ramped). Use 40 for raw precompute."""


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


def main(config: NpzReplayConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = config.npz.expanduser().resolve()
    tokens, left, right = load_motion_npz(path, max_frames=int(config.max_frames))
    n = int(tokens.shape[0])
    meta_path = path.parent / "record_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        logger.info("record_meta: %s", meta)

    messages = prebuild_latent_action_messages(
        tokens,
        left,
        right,
        hand_ramp_frames=int(config.hand_ramp_frames),
    )
    logger.info(
        "npz=%s frames=%d fps=%.1f hand_ramp=%d token[0]=%+.3f",
        path.name,
        n,
        float(config.fps),
        int(config.hand_ramp_frames),
        float(tokens[0, 0]),
    )

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://{config.zmq_host}:{config.zmq_port}")
    time.sleep(0.5)
    logger.info("bound tcp://%s:%d", config.zmq_host, config.zmq_port)

    if config.arm_flag:
        _wait_flag(Path(config.arm_flag), label="arm")
        time.sleep(config.start_delay_s)
        _send_deploy_start_commands(pub)

    if config.ready_flag:
        _wait_flag(Path(config.ready_flag), label="ready")
        time.sleep(config.start_delay_s)
        _send_deploy_start_commands(pub)

    period = 1.0 / float(config.fps)
    for i, msg in enumerate(messages):
        t0 = time.monotonic()
        pub.send(msg)
        if i == 0 or (i + 1) % 100 == 0 or i + 1 == n:
            logger.info(
                "frame %d/%d token[0]=%+.3f R_hand[0]=%+.3f",
                i + 1,
                n,
                float(tokens[i, 0]),
                float(right[i, 0]),
            )
        rem = period - (time.monotonic() - t0)
        if rem > 0:
            time.sleep(rem)

    logger.info("done (%d frames)", n)
    pub.close()
    ctx.term()


if __name__ == "__main__":
    main(tyro.cli(NpzReplayConfig))
