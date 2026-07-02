#!/usr/bin/env python3
"""Live tap on Phi-0 SONIC sim ZMQ ports (camera / pose tokens / deploy debug)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import msgpack
import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[1]
_GR00T = Path(
    os.environ.get("GR00T_ROOT", str(Path.home() / "YZY" / "GR00T-WholeBodyControl"))
).expanduser().resolve()
for p in (ROOT / "src", _GR00T):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gear_sonic.camera.sensor_server import ImageMessageSchema  # noqa: E402
from gear_sonic.utils.data_collection.zmq_state_subscriber import (  # noqa: E402
    CONFIG_ZMQ_TOPIC,
    STATE_ZMQ_TOPIC,
)

DEBUG_TOPICS = (STATE_ZMQ_TOPIC, CONFIG_ZMQ_TOPIC)
from gear_sonic.utils.zmq_pose_unpack import unpack_pose_message  # noqa: E402


def _arr_summary(x: np.ndarray, *, max_vals: int = 8) -> dict:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    out = {"shape": list(x.shape), "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean())}
    if x.size <= max_vals:
        out["values"] = x.tolist()
    else:
        out["head"] = x[:max_vals].tolist()
        out["tail"] = x[-4:].tolist()
    return out


def _summarize_camera(raw: bytes) -> dict:
    data = msgpack.unpackb(raw, raw=False)
    schema = ImageMessageSchema.deserialize(data)
    d = schema.asdict()
    images = {}
    for name, img in d.get("images", {}).items():
        arr = np.asarray(img)
        images[name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "mean": float(arr.mean()),
        }
    return {"timestamps": d.get("timestamps", {}), "images": images}


def _summarize_pose(raw: bytes) -> dict:
    if raw.startswith(b"command"):
        return {"topic": "command", "note": "deploy start/stop/planner control (not pose)"}
    if raw.startswith(b"planner"):
        return {"topic": "planner", "note": "planner mode message"}
    if not raw.startswith(b"pose"):
        return {"topic": raw[:16].decode("utf-8", errors="replace"), "note": "unknown prefix"}
    msg = unpack_pose_message(raw, topic="pose")
    out: dict = {"topic": "pose"}
    if "frame_index" in msg:
        out["frame_index"] = int(np.asarray(msg["frame_index"]).reshape(-1)[0])
    if "token_state" in msg:
        out["token_state"] = _arr_summary(msg["token_state"], max_vals=64)
    if "left_hand_joints" in msg:
        out["left_hand_joints"] = _arr_summary(msg["left_hand_joints"], max_vals=7)
    if "right_hand_joints" in msg:
        out["right_hand_joints"] = _arr_summary(msg["right_hand_joints"], max_vals=7)
    return out


def _parse_debug_message(raw: bytes) -> tuple[str, bytes]:
    for topic in DEBUG_TOPICS:
        prefix = topic.encode("utf-8")
        if raw.startswith(prefix):
            return topic, raw[len(prefix) :]
    head = raw[:24].decode("utf-8", errors="replace")
    return "unknown", raw


def _summarize_debug_payload(topic: str, payload: bytes) -> dict:
    if topic == CONFIG_ZMQ_TOPIC:
        data = msgpack.unpackb(payload, raw=False)
        if not isinstance(data, dict):
            return {"topic": topic, "error": "not a dict"}
        return {"topic": topic, "n_fields": len(data), "keys": sorted(data.keys())[:16]}

    data = msgpack.unpackb(payload, raw=False)
    if not isinstance(data, dict):
        return {"topic": topic, "error": "not a dict"}
    out: dict = {"topic": topic}
    if "index" in data:
        out["index"] = int(np.asarray(data["index"]).reshape(-1)[0])
    for key in (
        "token_state",
        "base_trans_measured",
        "base_trans_target",
        "base_quat",
        "base_quat_target",
        "body_q",
        "last_action",
        "last_left_hand_action",
        "last_right_hand_action",
        "left_hand_q",
        "right_hand_q",
        "vr_3pt_position",
        "vr_3pt_orientation",
    ):
        if key not in data:
            continue
        val = data[key]
        if isinstance(val, (list, tuple, np.ndarray)):
            out[key] = _arr_summary(np.asarray(val))
        else:
            out[key] = val
    out["keys"] = sorted(data.keys())
    return out


def _summarize_debug(raw: bytes) -> dict:
    topic, payload = _parse_debug_message(raw)
    summary = _summarize_debug_payload(topic, payload)
    if topic == "unknown":
        summary["note"] = "unrecognized topic prefix on debug port"
    return summary


def _safe_dump_name(label: str) -> str:
    return label.replace("/", "_").replace(" ", "_")


@dataclass
class PortRate:
    window_s: float = 2.0
    count: int = 0
    first_ts: float | None = None
    last_ts: float | None = None
    _times: deque[float] = field(default_factory=deque)

    def tick(self, t: float) -> None:
        self.count += 1
        if self.first_ts is None:
            self.first_ts = t
        self.last_ts = t
        self._times.append(t)
        cutoff = t - self.window_s
        while self._times and self._times[0] < cutoff:
            self._times.popleft()

    def window_hz(self) -> float:
        if len(self._times) < 2:
            return 0.0
        dt = self._times[-1] - self._times[0]
        if dt <= 0:
            return 0.0
        return (len(self._times) - 1) / dt

    def avg_hz(self) -> float:
        if self.first_ts is None or self.last_ts is None or self.count < 2:
            return 0.0
        dt = self.last_ts - self.first_ts
        return (self.count - 1) / dt if dt > 0 else 0.0


def _print_block(
    port: int,
    label: str,
    summary: dict,
    *,
    full: bool,
    rate: PortRate | None = None,
) -> None:
    ts = time.strftime("%H:%M:%S")
    hz = f"  {rate.window_hz():.1f} Hz (avg {rate.avg_hz():.1f})" if rate else ""
    print(f"\n[{ts}] port {port} ({label}){hz}")
    if full:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(json.dumps(summary, default=str))


def _print_stats_line(rates: dict[int, PortRate], labels: dict[int, str]) -> None:
    parts = []
    for port in sorted(rates):
        r = rates[port]
        label = labels.get(port, str(port))
        parts.append(f"{port}/{label}: {r.window_hz():.1f} Hz (n={r.count}, avg {r.avg_hz():.1f})")
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] freq  " + " | ".join(parts), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--camera-port", type=int, default=5555)
    p.add_argument("--pose-port", type=int, default=5556)
    p.add_argument("--debug-port", type=int, default=5557)
    p.add_argument(
        "--debug-topic",
        default="",
        help="SUB filter on debug port (default '' = all topics: g1_debug, robot_config)",
    )
    p.add_argument("--duration-s", type=float, default=0.0, help="0 = run until Ctrl-C")
    p.add_argument("--print-every", type=int, default=50, help="log every N msgs per port (ignored with --freq-only)")
    p.add_argument("--full", action="store_true", help="print full JSON (incl. all 64 token dims)")
    p.add_argument("--dump-dir", type=str, default="", help="save first sample per port as JSON")
    p.add_argument(
        "--stats-interval",
        type=float,
        default=2.0,
        help="print per-port Hz every N seconds (0=off)",
    )
    p.add_argument(
        "--hz-window",
        type=float,
        default=2.0,
        help="rolling window (seconds) for instantaneous Hz",
    )
    p.add_argument(
        "--freq-only",
        action="store_true",
        help="only print frequency lines (no payload dumps)",
    )
    p.add_argument(
        "--no-debug",
        action="store_true",
        help="do not monitor deploy debug port (default 5557: g1_debug + robot_config)",
    )
    args = p.parse_args()

    ctx = zmq.Context()
    poller = zmq.Poller()
    # poller.poll() keys are Socket objects, not id(socket).
    sockets: dict[zmq.Socket, tuple[int, str]] = {}
    all_socks: list[zmq.Socket] = []

    def _add_sub(port: int, label: str, *, topic: str = "") -> zmq.Socket:
        sock = ctx.socket(zmq.SUB)
        sock.connect(f"tcp://{args.host}:{port}")
        sock.setsockopt_string(zmq.SUBSCRIBE, topic)
        poller.register(sock, zmq.POLLIN)
        sockets[sock] = (port, label)
        all_socks.append(sock)
        return sock

    _add_sub(args.camera_port, "camera")
    _add_sub(args.pose_port, "pose/cmd")
    if not args.no_debug:
        _add_sub(args.debug_port, "deploy_debug", topic=args.debug_topic)

    counts = {args.camera_port: 0, args.pose_port: 0}
    labels = {args.camera_port: "camera", args.pose_port: "pose"}
    if not args.no_debug:
        counts[args.debug_port] = 0
        labels[args.debug_port] = "deploy"
    rates = {p: PortRate(window_s=args.hz_window) for p in counts}
    dumped: set[int] = set()
    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"monitoring tcp://{args.host}:"
        f"{args.camera_port} (camera), "
        f"{args.pose_port} (pose v4 / command)"
        + (
            ""
            if args.no_debug
            else f", {args.debug_port} (deploy: {STATE_ZMQ_TOPIC}, {CONFIG_ZMQ_TOPIC})"
        )
    )
    if args.stats_interval > 0:
        print(f"frequency stats every {args.stats_interval:.1f}s (window={args.hz_window:.1f}s)")
    print("start the sim / publisher / deploy in another terminal; Ctrl-C to stop")

    t_end = time.monotonic() + args.duration_s if args.duration_s > 0 else None
    next_stats = time.monotonic() + args.stats_interval if args.stats_interval > 0 else None
    try:
        while t_end is None or time.monotonic() < t_end:
            if next_stats is not None and time.monotonic() >= next_stats:
                _print_stats_line(rates, labels)
                next_stats = time.monotonic() + args.stats_interval

            events = dict(poller.poll(timeout=200))
            for sock, _ in events.items():
                if sock not in sockets:
                    continue
                port, label = sockets[sock]
                while True:
                    try:
                        raw = sock.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break

                    now = time.monotonic()
                    rates[port].tick(now)
                    counts[port] += 1
                    n = counts[port]

                    if port == args.camera_port:
                        summary = _summarize_camera(raw)
                    elif port == args.pose_port:
                        summary = _summarize_pose(raw)
                    else:
                        summary = _summarize_debug(raw)

                    if dump_dir is not None and port not in dumped:
                        dumped.add(port)
                        path = dump_dir / f"port_{port}_{_safe_dump_name(label)}.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(summary, indent=2, default=str))
                        print(f"[dump] {path}")

                    if not args.freq_only and (n == 1 or n % max(args.print_every, 1) == 0):
                        _print_block(port, label, summary, full=args.full, rate=rates[port])
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[done] message counts:", counts)
        _print_stats_line(rates, labels)
        if sum(counts.values()) == 0:
            print(
                "[hint] no messages — start sim + deploy + publisher first, "
                "then re-run this monitor (or run monitor before the demo)."
            )
        elif counts.get(args.debug_port, 0) == 0 and not args.no_debug:
            print(
                f"[hint] port {args.debug_port} silent — deploy g1_debug (~50 Hz) only after "
                "control loop starts (sim deploy or g1_deploy_onnx_ref --input-type zmq_manager)."
            )
        for sock in all_socks:
            try:
                poller.unregister(sock)
            except Exception:
                pass
            sock.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    main()
