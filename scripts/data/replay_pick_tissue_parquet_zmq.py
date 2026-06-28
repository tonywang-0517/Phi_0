#!/usr/bin/env python3
"""Replay pick-tissue LeRobot parquet (SMPL + motion_token) over ZMQ pose v4."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
import zmq

_PHI0 = Path(__file__).resolve().parents[2]
_GR00T = _PHI0.parent / "GR00T-WholeBodyControl"
if str(_GR00T) not in sys.path:
    sys.path.insert(0, str(_GR00T))

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (  # noqa: E402
    build_command_message,
    pack_pose_message,
)


def _row_smpl_joints(row) -> np.ndarray:
    j = np.asarray(row["teleop.smpl_joints"], dtype=np.float32).reshape(24, 3)
    return j.reshape(1, 24, 3)


def _row_smpl_pose(row) -> np.ndarray:
    p = np.asarray(row["teleop.smpl_pose"], dtype=np.float32).reshape(-1)
    if p.size != 63:
        raise ValueError(f"expected smpl_pose 63, got {p.size}")
    return p.reshape(1, 21, 3)


def main(
    parquet: Path,
    *,
    zmq_host: str = "127.0.0.1",
    zmq_port: int = 5556,
    fps: float = 50.0,
    start_delay_s: float = 0.5,
    ready_flag: Path | None = None,
    max_frames: int = 0,
) -> None:
    df = pd.read_parquet(parquet)
    n = len(df) if max_frames <= 0 else min(len(df), max_frames)
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{zmq_port}")
    print(f"[pick_tissue_replay] bound tcp://*:{zmq_port} parquet={parquet.name} frames={n}")
    time.sleep(0.3)

    if ready_flag is not None:
        while not ready_flag.is_file():
            time.sleep(0.05)

    time.sleep(start_delay_s)
    sock.send(build_command_message(start=True, stop=False, planner=True))
    time.sleep(0.2)
    sock.send(build_command_message(start=True, stop=False, planner=False))
    time.sleep(0.2)

    dt = 1.0 / float(fps)
    t_next = time.perf_counter()
    for i in range(n):
        row = df.iloc[i]
        pose = {
            "smpl_joints": _row_smpl_joints(row),
            "smpl_pose": _row_smpl_pose(row),
            "body_quat_w": np.asarray(row["teleop.body_quat_w"], dtype=np.float32).reshape(1, 4),
            "left_hand_joints": np.asarray(row["teleop.left_hand_joints"], dtype=np.float32).reshape(1, 7),
            "right_hand_joints": np.asarray(row["teleop.right_hand_joints"], dtype=np.float32).reshape(1, 7),
            "frame_index": np.asarray(row["teleop.smpl_frame_index"], dtype=np.int64).reshape(1),
            "token_state": np.asarray(row["action.motion_token"], dtype=np.float64).reshape(1, -1),
        }
        sock.send(pack_pose_message(pose, topic="pose", version=4))
        t_next += dt
        sleep_s = t_next - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)

    sock.close(linger=0)
    ctx.term()
    print(f"[pick_tissue_replay] done {n} frames")


if __name__ == "__main__":
    tyro.cli(main)
