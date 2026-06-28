#!/usr/bin/env python3
"""Subscribe g1_debug and record base_trans_measured while replay runs."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import msgpack
import msgpack_numpy as mnp
import numpy as np
import tyro
import zmq

_PHI0 = Path(__file__).resolve().parents[2]
_GR00T = _PHI0.parent / "GR00T-WholeBodyControl"
if str(_GR00T) not in sys.path:
    sys.path.insert(0, str(_GR00T))

from gear_sonic.utils.data_collection.zmq_state_subscriber import STATE_ZMQ_TOPIC  # noqa: E402


def _base_xyz(msg: dict) -> np.ndarray:
    for key in ("base_trans_measured", "base_trans", "base_trans_target"):
        if key in msg:
            return np.asarray(msg[key], dtype=np.float64).reshape(3)
    raise KeyError(f"no base trans in g1_debug keys={list(msg.keys())[:20]}")


def main(
    *,
    num_frames: int,
    zmq_host: str = "127.0.0.1",
    zmq_port: int = 5557,
    out_npy: Path,
    warmup_s: float = 1.0,
    timeout_s: float = 120.0,
) -> None:
    mnp.patch()
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{zmq_host}:{zmq_port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, STATE_ZMQ_TOPIC)
    sock.setsockopt(zmq.RCVTIMEO, 500)
    time.sleep(warmup_s)

    out: list[np.ndarray] = []
    deadline = time.monotonic() + timeout_s
    while len(out) < num_frames and time.monotonic() < deadline:
        try:
            raw = sock.recv()
        except zmq.Again:
            continue
        payload = raw[len(STATE_ZMQ_TOPIC) :]
        msg = msgpack.unpackb(payload, raw=False)
        for k, v in list(msg.items()):
            if isinstance(v, list):
                msg[k] = np.asarray(v)
        out.append(_base_xyz(msg).astype(np.float32))

    sock.close(linger=0)
    ctx.term()
    if len(out) < num_frames:
        raise RuntimeError(f"captured {len(out)}/{num_frames} base_trans frames (timeout={timeout_s}s)")
    arr = np.stack(out[:num_frames], axis=0)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, arr)
    span = arr.max(0) - arr.min(0)
    print(f"[capture] saved {out_npy} shape={arr.shape} xyz_span={span}")


if __name__ == "__main__":
    tyro.cli(main)
