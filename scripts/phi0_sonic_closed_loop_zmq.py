#!/usr/bin/env python3
"""Closed-loop Phi-0 -> SONIC ZMQ v4: live or GT camera + robot or roll-forward proprio."""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from typing import Callable, Protocol

import numpy as np
import torch
import zmq
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
_GR00T = Path(
    os.environ.get(
        "GR00T_ROOT",
        str(Path.home() / "YZY" / "GR00T-WholeBodyControl"),
    )
).expanduser().resolve()
for p in (ROOT / "src", ROOT, _GR00T):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gear_sonic.utils.inference.vla_utils import (  # noqa: E402
    calculate_latency_compensated_index,
    should_trigger_new_inference,
)
from gear_sonic.utils.teleop.zmq.v4_latent_replay import (  # noqa: E402
    hand_ramp_weights,
    pack_latent_action_message,
)

from phi0.checkpoint_utils import merge_saved_cfg  # noqa: E402
from phi0.deploy.pick_tissue_gt import (  # noqa: E402
    PickTissueEpisodeSpan,
    control_index_to_global_frame,
    reader_from_data_cfg,
)
from phi0.deploy.gt_io import (  # noqa: E402
    GtEpisodeProprioSource,
    PickTissueGtBackend,
    is_pick_tissue_unified_cfg,
)
from phi0.deploy.deploy_keyboard import DeployKeyboardListener, send_start_streamed  # noqa: E402
from phi0.deploy.robot_proprio import RobotProprioSource  # noqa: E402
from phi0.deploy.sonic_zmq_io import unified_action_denorm_to_zmq_arrays  # noqa: E402
from phi0.schema.unified_action_schema import D_UNIFIED  # noqa: E402
from phi0.inference.session import ActionInferenceSession, resolve_deploy_action_chunk_size
from phi0.models.vlm.preprocess import normalize_vlm_instruction
from phi0.runtime import (  # noqa: E402
    activate_cuda_device,
    apply_processor_stats_from_checkpoint,
    build_processor,
    create_phi0,
    resolve_inference_device,
    sync_model_action_norm,
)

logger = logging.getLogger(__name__)

_EGO_KEYS = ("ego_view", "head", "observation.images.ego_view")
_WRIST_KEYS = (
    "left_wrist",
    "observation.images.left_wrist",
    "wrist_view",
    "right_wrist",
    "observation.images.right_wrist",
)


@dataclass
class ActionChunk:
    tokens: np.ndarray
    left: np.ndarray
    right: np.ndarray
    horizon: int


@dataclass
class ObsSnapshot:
    control_idx: int
    ego_hwc: np.ndarray
    wrist_hwc: np.ndarray | None
    timestamp: float


@dataclass
class PredictResult:
    chunk: ActionChunk
    obs: ObsSnapshot


class ClosedLoopRecorder:
    """Append-only trace; flushed to npz on save()."""

    def __init__(
        self,
        *,
        prompt: str,
        camera_source: str,
        control_fps: float,
        checkpoint: str,
    ):
        self._meta = {
            "prompt": prompt,
            "camera_source": camera_source,
            "control_fps": float(control_fps),
            "checkpoint": str(checkpoint),
        }
        self._obs_ego: list[np.ndarray] = []
        self._obs_wrist: list[np.ndarray] = []
        self._obs_control_idx: list[int] = []
        self._obs_timestamp: list[float] = []
        self._obs_inference_elapsed_s: list[float] = []
        self._has_wrist = False
        self._out_tokens: list[np.ndarray] = []
        self._out_left: list[np.ndarray] = []
        self._out_right: list[np.ndarray] = []
        self._out_control_idx: list[int] = []
        self._out_chunk_idx: list[int] = []
        self._out_frame_index: list[int] = []
        self._out_hand_ramp: list[float] = []
        self._out_timestamp: list[float] = []

    def record_observation(
        self,
        obs: ObsSnapshot,
        *,
        inference_elapsed_s: float | None = None,
    ) -> None:
        self._obs_ego.append(np.asarray(obs.ego_hwc, dtype=np.uint8))
        self._obs_control_idx.append(int(obs.control_idx))
        self._obs_timestamp.append(float(obs.timestamp))
        if inference_elapsed_s is not None:
            self._obs_inference_elapsed_s.append(float(inference_elapsed_s))
        if obs.wrist_hwc is not None:
            self._has_wrist = True
            self._obs_wrist.append(np.asarray(obs.wrist_hwc, dtype=np.uint8))
        elif self._has_wrist:
            self._obs_wrist.append(
                np.zeros_like(self._obs_ego[-1], dtype=np.uint8)
            )

    def record_output(
        self,
        *,
        token: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        control_idx: int,
        chunk_idx: int,
        frame_index: int,
        hand_ramp: float,
        timestamp: float,
    ) -> None:
        self._out_tokens.append(np.asarray(token, dtype=np.float32).reshape(-1))
        self._out_left.append(np.asarray(left, dtype=np.float32).reshape(-1))
        self._out_right.append(np.asarray(right, dtype=np.float32).reshape(-1))
        self._out_control_idx.append(int(control_idx))
        self._out_chunk_idx.append(int(chunk_idx))
        self._out_frame_index.append(int(frame_index))
        self._out_hand_ramp.append(float(hand_ramp))
        self._out_timestamp.append(float(timestamp))

    def save(self, obs_path: Path, output_path: Path) -> None:
        obs_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path = obs_path.parent / "record_meta.json"
        meta_path.write_text(json.dumps(self._meta, indent=2), encoding="utf-8")
        logger.info("saved metadata %s", meta_path)
        if self._obs_ego:
            obs_payload: dict = {
                "control_idx": np.asarray(self._obs_control_idx, dtype=np.int32),
                "timestamp": np.asarray(self._obs_timestamp, dtype=np.float64),
                "ego": np.stack(self._obs_ego, axis=0),
                "num_inferences": np.int32(len(self._obs_ego)),
            }
            if self._has_wrist and len(self._obs_wrist) == len(self._obs_ego):
                obs_payload["wrist"] = np.stack(self._obs_wrist, axis=0)
            if len(self._obs_inference_elapsed_s) == len(self._obs_ego):
                elapsed = np.asarray(self._obs_inference_elapsed_s, dtype=np.float64)
                obs_payload["inference_elapsed_s"] = elapsed
            np.savez(obs_path, **obs_payload)
            if "inference_elapsed_s" in obs_payload:
                e = obs_payload["inference_elapsed_s"]
                logger.info(
                    "saved observations %s (%d inferences, ego %s, infer %.3f/%.3f/%.3fs mean/p95/max)",
                    obs_path,
                    len(self._obs_ego),
                    obs_payload["ego"].shape,
                    float(np.mean(e)),
                    float(np.percentile(e, 95)),
                    float(np.max(e)),
                )
            else:
                logger.info(
                    "saved observations %s (%d inferences, ego %s)",
                    obs_path,
                    len(self._obs_ego),
                    obs_payload["ego"].shape,
                )
        else:
            logger.warning("no observations recorded; skip %s", obs_path)

        if self._out_tokens:
            np.savez(
                output_path,
                tokens=np.stack(self._out_tokens, axis=0).astype(np.float32),
                left=np.stack(self._out_left, axis=0).astype(np.float32),
                right=np.stack(self._out_right, axis=0).astype(np.float32),
                control_idx=np.asarray(self._out_control_idx, dtype=np.int32),
                chunk_idx=np.asarray(self._out_chunk_idx, dtype=np.int32),
                frame_index=np.asarray(self._out_frame_index, dtype=np.int32),
                hand_ramp=np.asarray(self._out_hand_ramp, dtype=np.float32),
                timestamp=np.asarray(self._out_timestamp, dtype=np.float64),
                num_frames=np.int32(len(self._out_tokens)),
            )
            logger.info(
                "saved outputs %s (%d frames x 64-d tokens)",
                output_path,
                len(self._out_tokens),
            )
        else:
            logger.warning("no outputs recorded; skip %s", output_path)


class CameraSource(Protocol):
    def read_ego_wrist_chw(self, control_idx: int) -> tuple[torch.Tensor, torch.Tensor | None]: ...

    def close(self) -> None: ...


@dataclass
class LiveCameraSource:
    client: object

    def read_ego_wrist_chw(self, control_idx: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        del control_idx
        return _read_live_camera_frames(self.client)

    def close(self) -> None:
        self.client.close()


@dataclass
class GtCameraSource:
    reader: object
    span: PickTissueEpisodeSpan
    native_fps: float
    control_fps: float
    start_control_idx: int
    max_control_frames: int

    def read_ego_wrist_chw(self, control_idx: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        rel = int(control_idx) - int(self.start_control_idx)
        rel = max(0, min(rel, self.max_control_frames - 1))
        global_frame = control_index_to_global_frame(
            self.span.frame_start,
            rel,
            native_fps=self.native_fps,
            control_fps=self.control_fps,
        )
        ego_rgb, wrist_rgb = self.reader.read_ego_wrist_pair(global_frame, self.span)
        wrist = _rgb_to_chw(wrist_rgb)
        return _rgb_to_chw(ego_rgb), wrist

    def close(self) -> None:
        close_fn = getattr(self.reader, "close", None)
        if callable(close_fn):
            close_fn()


def _summarize_camera_msg(msg: dict) -> str:
    images = msg.get("images") or {}
    parts = []
    for name, img in sorted(images.items()):
        arr = np.asarray(img)
        parts.append(f"{name}={list(arr.shape)}")
    ts = msg.get("timestamps") or {}
    ts_part = ", ".join(f"{k}={v:.3f}" for k, v in sorted(ts.items())[:4])
    return f"keys=[{', '.join(parts)}] ts=[{ts_part}]"


def _probe_live_camera(client, *, host: str, port: int, wait_s: float) -> dict:
    """Block until composed_camera on tcp://host:port returns at least one frame."""
    deadline = time.monotonic() + wait_s
    last_keys: list[str] = []
    while time.monotonic() < deadline:
        msg = client.read(blocking=False)
        if msg and msg.get("images"):
            summary = _summarize_camera_msg(msg)
            logger.info(
                "SONIC camera_server tcp://%s:%d ready %s",
                host,
                port,
                summary,
            )
            return msg
        if msg:
            last_keys = sorted((msg.get("images") or {}).keys())
        time.sleep(0.05)
    raise TimeoutError(
        f"no frames from SONIC composed_camera tcp://{host}:{port} within {wait_s:.0f}s"
        + (f" (keys seen: {last_keys})" if last_keys else "")
    )


def _build_camera_source(args, *, data_cfg) -> CameraSource:
    source = str(args.camera_source).strip().lower()
    if source == "gt":
        ep = int(args.gt_camera_episode)
        reader = reader_from_data_cfg(data_cfg)
        span = reader.episode_span(ep)
        native_fps = float(reader.native_fps)
        control_fps = float(args.control_fps)
        max_frames = int(span.frame_count)
        if float(args.motion_seconds) > 0:
            max_frames = min(
                max_frames,
                int(round(float(args.motion_seconds) * control_fps)),
            )
        src = GtCameraSource(
            reader=reader,
            span=span,
            native_fps=native_fps,
            control_fps=control_fps,
            start_control_idx=int(args.gt_camera_start_idx),
            max_control_frames=max(1, max_frames),
        )
        ego, wrist = src.read_ego_wrist_chw(int(args.gt_camera_start_idx))
        logger.info(
            "GT camera episode=%d frames=%d start_ctrl=%d ego=%s wrist=%s",
            ep,
            src.max_control_frames,
            int(args.gt_camera_start_idx),
            tuple(ego.shape),
            None if wrist is None else tuple(wrist.shape),
        )
        return src

    host = args.camera_host.strip()
    port = int(args.camera_port)
    from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

    logger.info("connecting SONIC composed_camera tcp://%s:%d", host, port)
    client = ComposedCameraClientSensor(server_ip=host, port=port)
    _probe_live_camera(client, host=host, port=port, wait_s=float(args.camera_wait_s))
    return LiveCameraSource(client=client)


def parse_args():
    p = argparse.ArgumentParser(
        description="Phi-0 closed-loop SONIC latent publisher (camera + roll-forward proprio)",
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config-dir", type=str, default=str(ROOT / "configs"))
    p.add_argument(
        "--config-name",
        type=str,
        default="train_pick_tissue_xperience_unified_ddp4_3k",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--min-free-gb", type=float, default=12.0)
    p.add_argument("--prompt", type=str, default="pick tissue")
    p.add_argument(
        "--camera-source",
        type=str,
        choices=("live", "gt"),
        default="live",
        help="live=SONIC composed_camera ZMQ :5555; gt=pick-tissue dataset episode.",
    )
    p.add_argument("--camera-host", type=str, default="127.0.0.1")
    p.add_argument("--camera-port", type=int, default=5555)
    p.add_argument("--camera-wait-s", type=float, default=30.0)
    p.add_argument(
        "--gt-camera-episode",
        type=int,
        default=447,
        help="Dataset episode_index when --camera-source=gt.",
    )
    p.add_argument(
        "--gt-camera-start-idx",
        type=int,
        default=0,
        help="Control-frame offset into the GT episode (50 Hz timeline).",
    )
    p.add_argument("--zmq-host", type=str, default="127.0.0.1")
    p.add_argument("--zmq-port", type=int, default=5556)
    p.add_argument("--state-zmq-host", type=str, default="127.0.0.1")
    p.add_argument("--state-zmq-port", type=int, default=5557)
    p.add_argument("--control-fps", type=float, default=50.0)
    p.add_argument(
        "--inference-rate",
        type=float,
        default=2.5,
        help="Max policy re-inference rate (Hz); async worker runs at most this often.",
    )
    p.add_argument("--motion-seconds", type=float, default=0.0, help="0 = run until Ctrl-C")
    p.add_argument("--hand-ramp-frames", type=int, default=40)
    p.add_argument("--start-delay-s", type=float, default=0.5)
    p.add_argument(
        "--gt-proprio-episode",
        type=int,
        default=0,
        help=(
            "Fallback GT proprio episode_index when g1_debug is absent (live camera). "
            "Default 0: with --camera-source=gt, uses --gt-camera-episode automatically."
        ),
    )
    p.add_argument(
        "--proprio-source",
        type=str,
        choices=("robot", "hybrid", "roll-forward"),
        default="robot",
        help=(
            "robot=g1_debug 5557 unified proprio; "
            "hybrid=SMPL semantic from last pred + robot tail; "
            "roll-forward=predicted proprio only. "
            "When robot proprio is missing, dataset GT from the video episode is used if available."
        ),
    )
    p.add_argument(
        "--seed-proprio",
        action="store_true",
        help="Use dataset-mean proprio before g1_debug arrives (robot/hybrid only). Default: off.",
    )
    p.add_argument(
        "--wait-robot-proprio",
        action="store_true",
        help=(
            "Block until first g1_debug on state port before starting. "
            "Default: subscribe immediately and apply robot proprio once deploy control loop publishes (~50 Hz)."
        ),
    )
    p.add_argument(
        "--wait-deploy-state",
        action="store_true",
        help="Wait for first g1_debug frame on state port (roll-forward mode only).",
    )
    p.add_argument(
        "--stream-now",
        action="store_true",
        help="Send deploy start commands after the first action chunk is ready.",
    )
    p.add_argument(
        "--no-zmq",
        action="store_true",
        help="Do not bind/send ZMQ tokens or deploy commands; only save to npz.",
    )
    p.add_argument(
        "--deploy-keyboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read stdin keys to send deploy ZMQ commands (], O, Enter, p). Ignored with --no-zmq.",
    )
    p.add_argument(
        "--record-dir",
        type=str,
        default="",
        help="If set, save observations.npz (per inference) and outputs.npz (per tx frame).",
    )
    p.add_argument(
        "--record-obs",
        type=str,
        default="",
        help="Override observations npz path (default: RECORD_DIR/observations.npz).",
    )
    p.add_argument(
        "--record-output",
        type=str,
        default="",
        help="Override outputs npz path (default: RECORD_DIR/outputs.npz).",
    )
    return p.parse_args()


def _resolve_record_paths(args) -> tuple[Path | None, Path | None]:
    record_dir = args.record_dir.strip()
    obs_explicit = args.record_obs.strip()
    out_explicit = args.record_output.strip()
    if not record_dir and not obs_explicit and not out_explicit:
        return None, None
    if record_dir:
        base = Path(record_dir).expanduser().resolve()
        obs_path = (
            Path(obs_explicit).expanduser().resolve()
            if obs_explicit
            else base / "observations.npz"
        )
        out_path = (
            Path(out_explicit).expanduser().resolve()
            if out_explicit
            else base / "outputs.npz"
        )
    else:
        if not obs_explicit or not out_explicit:
            raise SystemExit(
                "--record-obs and --record-output required when --record-dir is omitted"
            )
        obs_path = Path(obs_explicit).expanduser().resolve()
        out_path = Path(out_explicit).expanduser().resolve()
    return obs_path, out_path


def _chw_to_hwc_uint8(frame_chw: torch.Tensor) -> np.ndarray:
    x = frame_chw.detach().float().cpu().numpy()
    if x.ndim != 3:
        raise ValueError(f"expected CHW, got {x.shape}")
    if x.max() <= 1.0 + 1e-6:
        x = np.clip(x * 255.0, 0.0, 255.0)
    return np.ascontiguousarray(x.transpose(1, 2, 0).astype(np.uint8))


def _pick_image(images: dict, keys: tuple[str, ...], *, label: str) -> np.ndarray:
    for key in keys:
        if key in images:
            return np.asarray(images[key], dtype=np.uint8)
    available = sorted(images.keys())
    if label == "ego" and available:
        logger.warning("%s missing; using %s", label, available[0])
        return np.asarray(images[available[0]], dtype=np.uint8)
    raise KeyError(f"{label} camera missing (have {available})")


def _rgb_to_chw(rgb: np.ndarray) -> torch.Tensor:
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HWC RGB, got {arr.shape}")
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0


def _frame_to_bcthw(frame_chw: torch.Tensor, *, device, dtype) -> torch.Tensor:
    clip = frame_chw.unsqueeze(0).unsqueeze(2)
    return (clip.to(device=device, dtype=dtype) * 2.0 - 1.0)


def _read_live_camera_frames(client) -> tuple[torch.Tensor, torch.Tensor | None]:
    deadline = time.monotonic() + 2.0
    last_keys: list[str] = []
    while time.monotonic() < deadline:
        msg = client.read(blocking=False)
        if msg and msg.get("images"):
            images = msg["images"]
            last_keys = sorted(images.keys())
            ego = _pick_image(images, _EGO_KEYS, label="ego")
            wrist_chw = None
            for key in _WRIST_KEYS:
                if key in images:
                    wrist_chw = _rgb_to_chw(images[key])
                    break
            return _rgb_to_chw(ego), wrist_chw
        time.sleep(0.01)
    hint = f" keys seen: {last_keys}" if last_keys else ""
    raise TimeoutError(f"no camera frame within 2s{hint}")



def _wait_for_deploy_state(host: str, port: int, timeout_s: float) -> None:
    import msgpack
    import msgpack_numpy as mnp

    from gear_sonic.utils.data_collection.zmq_state_subscriber import STATE_ZMQ_TOPIC

    mnp.patch()
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{host}:{port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, STATE_ZMQ_TOPIC)
    sock.setsockopt(zmq.RCVTIMEO, 500)
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            try:
                raw = sock.recv()
            except zmq.Again:
                continue
            msg = msgpack.unpackb(raw[len(STATE_ZMQ_TOPIC) :], raw=False)
            logger.info("deploy state ready keys=%s", sorted(msg.keys())[:12])
            return
    finally:
        sock.close(linger=0)
    raise TimeoutError(f"no g1_debug on tcp://{host}:{port} within {timeout_s:.0f}s")


def _send_deploy_start_commands(pub: zmq.Socket, send_lock: threading.Lock | None = None) -> None:
    send_start_streamed(pub, send_lock=send_lock)
    time.sleep(0.2)
    logger.info("sent ZMQ command start (planner -> streamed motion)")


def _load_model_bundle(args):
    ckpt_path = Path(args.checkpoint).resolve()
    device = resolve_inference_device(args.device, min_free_gb=float(args.min_free_gb))
    activate_cuda_device(device)
    with initialize_config_dir(version_base="1.3", config_dir=args.config_dir):
        cfg = compose(config_name=args.config_name)
    cfg.device = device

    logger.info("loading checkpoint %s", ckpt_path)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and payload.get("cfg"):
        cfg = merge_saved_cfg(cfg, payload["cfg"])

    model = create_phi0(cfg, smoke=bool(cfg.get("smoke_action_only", False)))
    if isinstance(payload, dict) and ("model" in payload or "action_expert" in payload):
        model.load_checkpoint(str(ckpt_path))
    processor = build_processor(cfg).eval()
    if isinstance(payload, dict):
        apply_processor_stats_from_checkpoint(processor, payload, cfg)
    sync_model_action_norm(model, processor)
    model.eval()

    chunk_h = resolve_deploy_action_chunk_size(
        model, seq_len=int(cfg.data.get("seq_len", 33))
    )
    prompt = normalize_vlm_instruction(args.prompt)
    proprio_source = str(args.proprio_source).strip().lower()
    use_robot_proprio = proprio_source in {"robot", "hybrid"}
    session = ActionInferenceSession(
        model,
        processor=processor,
        deploy_seq_len=int(cfg.data.get("seq_len", 33)),
        use_gt_proprio=use_robot_proprio,
        use_wrist_view=bool(processor.use_wrist_view),
    )
    if proprio_source == "roll-forward" or bool(getattr(args, "seed_proprio", False)):
        mean = processor.mean.to(device=model.device, dtype=model.torch_dtype)
        session.seed_proprio_from_normalized(mean.reshape(-1))
        logger.info("proprio seeded from dataset mean")
    elif use_robot_proprio:
        logger.info("proprio: no mean seed — inference waits for g1_debug")
    logger.info(
        "model ready chunk_h=%d past_w=%d use_wrist=%s prompt=%r proprio=%s",
        chunk_h,
        int(getattr(model, "past_action_window_size", 1)),
        processor.use_wrist_view,
        prompt,
        proprio_source,
    )
    return model, processor, session, prompt, chunk_h, cfg, proprio_source


def _build_gt_proprio_source(
    args,
    data_cfg,
    camera: CameraSource,
) -> GtEpisodeProprioSource | None:
    if not is_pick_tissue_unified_cfg(data_cfg):
        return None
    control_fps = float(args.control_fps)
    if isinstance(camera, GtCameraSource):
        return GtEpisodeProprioSource(
            PickTissueGtBackend(
                reader=camera.reader,
                span=camera.span,
                native_fps=float(camera.native_fps),
                control_fps=float(camera.control_fps),
            )
        )
    ep = int(getattr(args, "gt_proprio_episode", 0) or 0)
    if ep <= 0:
        return None
    reader = reader_from_data_cfg(data_cfg)
    span = reader.episode_span(ep)
    return GtEpisodeProprioSource(
        PickTissueGtBackend(
            reader=reader,
            span=span,
            native_fps=float(reader.native_fps),
            control_fps=control_fps,
        )
    )


def _apply_robot_proprio(
    *,
    session: ActionInferenceSession,
    processor,
    robot_proprio: RobotProprioSource | None,
    proprio_source: str,
) -> bool:
    del proprio_source
    if robot_proprio is None:
        return False
    robot_proprio.poll()
    if not robot_proprio.ready:
        return False
    norm = robot_proprio.build_normalized(processor)
    w = int(getattr(session.model, "past_action_window_size", 1) or 1)
    steps = norm.reshape(1, -1).expand(w, -1)
    session.set_proprio_gt(steps)
    return True


def _apply_proprio(
    *,
    session: ActionInferenceSession,
    processor,
    control_idx: int,
    robot_proprio: RobotProprioSource | None,
    gt_proprio: GtEpisodeProprioSource | None,
    proprio_source: str,
    gt_fallback_logged: list[bool] | None = None,
) -> bool:
    if proprio_source in {"robot", "hybrid"} and robot_proprio is not None:
        robot_proprio.poll()
        if robot_proprio.ready:
            norm = robot_proprio.build_normalized(processor)
            w = int(getattr(session.model, "past_action_window_size", 1) or 1)
            steps = norm.reshape(1, -1).expand(w, -1)
            session.set_proprio_gt(steps)
            return True
    if gt_proprio is not None:
        if (
            proprio_source in {"robot", "hybrid"}
            and robot_proprio is not None
            and gt_fallback_logged is not None
            and not gt_fallback_logged[0]
        ):
            gt_fallback_logged[0] = True
            logger.info(
                "robot proprio unavailable; using dataset GT proprio at control_idx=%d",
                int(control_idx),
            )
        gt_proprio.apply_to_session(session, processor, int(control_idx))
        return True
    return False


def _predict_chunk(
    *,
    session: ActionInferenceSession,
    model,
    processor,
    camera: CameraSource,
    get_control_idx: Callable[[], int],
    prompt: str,
    chunk_h: int,
    robot_proprio: RobotProprioSource | None = None,
    gt_proprio: GtEpisodeProprioSource | None = None,
    proprio_source: str = "roll-forward",
    gt_fallback_logged: list[bool] | None = None,
) -> PredictResult:
    control_idx = int(get_control_idx())
    ego_chw, wrist_chw = camera.read_ego_wrist_chw(control_idx)
    ego_clip = _frame_to_bcthw(ego_chw, device=model.device, dtype=model.torch_dtype)
    wrist_clip = None
    wrist_hwc = _chw_to_hwc_uint8(wrist_chw) if wrist_chw is not None else None
    if processor.use_wrist_view:
        if wrist_chw is None:
            wrist_chw = ego_chw
            wrist_hwc = _chw_to_hwc_uint8(ego_chw)
            logger.warning("wrist camera missing; reusing ego frame")
        wrist_clip = _frame_to_bcthw(wrist_chw, device=model.device, dtype=model.torch_dtype)

    if session.action_ctx is None:
        session.prefill_from_video_clip(
            ego_clip,
            prompt,
            wrist_video=wrist_clip,
        )
    else:
        session.refresh_video_context_from_clip(
            ego_clip,
            prompt=prompt,
            wrist_video=wrist_clip,
        )

    _apply_proprio(
        session=session,
        processor=processor,
        control_idx=control_idx,
        robot_proprio=robot_proprio,
        gt_proprio=gt_proprio,
        proprio_source=proprio_source,
        gt_fallback_logged=gt_fallback_logged,
    )

    use_amp = model.device.type == "cuda"
    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=use_amp):
        pred = session.predict(int(chunk_h), denormalize=True)
    action_denorm = pred.float().detach().cpu().numpy()
    if robot_proprio is not None and proprio_source == "hybrid":
        if action_denorm.ndim == 2:
            robot_proprio.set_semantic_base(action_denorm[-1])
        else:
            robot_proprio.set_semantic_base(action_denorm.reshape(-1)[:D_UNIFIED])
    tokens, left, right = unified_action_denorm_to_zmq_arrays(action_denorm)
    chunk = ActionChunk(
        tokens=tokens,
        left=left,
        right=right,
        horizon=int(tokens.shape[0]),
    )
    obs = ObsSnapshot(
        control_idx=control_idx,
        ego_hwc=_chw_to_hwc_uint8(ego_chw),
        wrist_hwc=wrist_hwc,
        timestamp=time.monotonic(),
    )
    return PredictResult(chunk=chunk, obs=obs)


def _inference_worker(
    *,
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
    session: ActionInferenceSession,
    model,
    processor,
    camera: CameraSource,
    get_control_idx: Callable[[], int],
    prompt: str,
    chunk_h: int,
    recorder: ClosedLoopRecorder | None = None,
    robot_proprio: RobotProprioSource | None = None,
    gt_proprio: GtEpisodeProprioSource | None = None,
    proprio_source: str = "roll-forward",
    robot_proprio_ready_logged: list[bool] | None = None,
    gt_fallback_logged: list[bool] | None = None,
    defer_inference_without_robot: bool = False,
    defer_inference_logged: list[bool] | None = None,
) -> None:
    while not stop_event.is_set():
        try:
            inference_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        busy_event.set()
        try:
            if defer_inference_without_robot and proprio_source in {"robot", "hybrid"}:
                has_proprio = False
                if robot_proprio is not None:
                    robot_proprio.poll()
                    has_proprio = robot_proprio.ready
                if not has_proprio and gt_proprio is None:
                    if defer_inference_logged is not None and not defer_inference_logged[0]:
                        defer_inference_logged[0] = True
                        logger.info(
                            "inference deferred until first g1_debug (seeded proprio disabled)"
                        )
                    continue
            t0 = time.monotonic()
            result = _predict_chunk(
                session=session,
                model=model,
                processor=processor,
                camera=camera,
                get_control_idx=get_control_idx,
                prompt=prompt,
                chunk_h=chunk_h,
                robot_proprio=robot_proprio,
                gt_proprio=gt_proprio,
                proprio_source=proprio_source,
                gt_fallback_logged=gt_fallback_logged,
            )
            if (
                robot_proprio is not None
                and robot_proprio.ready
                and robot_proprio_ready_logged is not None
                and not robot_proprio_ready_logged[0]
            ):
                robot_proprio_ready_logged[0] = True
                logger.info(
                    "robot proprio active (first g1_debug applied; deploy control loop running)"
                )
            chunk = result.chunk
            delay = time.monotonic() - t0
            if recorder is not None:
                recorder.record_observation(result.obs, inference_elapsed_s=delay)
            logger.info(
                "inference done horizon=%d token0=%+.3f elapsed=%.3fs ctrl=%d",
                chunk.horizon,
                float(chunk.tokens[0, 0]),
                delay,
                result.obs.control_idx,
            )
            try:
                result_queue.put_nowait((chunk, t0))
            except queue.Full:
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass
                result_queue.put_nowait((chunk, t0))
        except Exception:
            logger.exception("inference worker failed")
        finally:
            busy_event.clear()


def _sleep_remaining(t_start: float, period: float) -> None:
    rem = period - (time.monotonic() - t_start)
    if rem > 0:
        time.sleep(rem)


def run_closed_loop(args) -> None:
    os.chdir(ROOT)
    model, processor, session, prompt, chunk_h, cfg, proprio_source = _load_model_bundle(args)
    obs_path, output_path = _resolve_record_paths(args)
    recorder: ClosedLoopRecorder | None = None
    if obs_path is not None and output_path is not None:
        recorder = ClosedLoopRecorder(
            prompt=prompt,
            camera_source=str(args.camera_source),
            control_fps=float(args.control_fps),
            checkpoint=str(args.checkpoint),
        )
        logger.info("recording obs=%s output=%s", obs_path, output_path)

    camera = _build_camera_source(args, data_cfg=cfg.data)
    gt_proprio = _build_gt_proprio_source(args, cfg.data, camera)
    if gt_proprio is not None:
        session.use_gt_proprio = True
        logger.info(
            "dataset GT proprio enabled (episode fallback when g1_debug absent)"
        )
    total_frames_sent = 0
    robot_proprio: RobotProprioSource | None = None

    def _control_idx() -> int:
        if str(args.camera_source).strip().lower() == "gt":
            return int(args.gt_camera_start_idx) + total_frames_sent
        return total_frames_sent

    try:
        if proprio_source in {"robot", "hybrid"}:
            robot_proprio = RobotProprioSource(
                args.state_zmq_host.strip(),
                int(args.state_zmq_port),
            )
            if args.wait_robot_proprio:
                first = robot_proprio.wait_first(timeout_s=float(args.camera_wait_s))
                norm0 = robot_proprio.build_normalized(processor, msg=first)
                session.set_proprio_gt(norm0.reshape(1, -1))
                logger.info(
                    "robot proprio ready tcp://%s:%d keys=%s",
                    args.state_zmq_host,
                    int(args.state_zmq_port),
                    sorted(first.keys())[:12],
                )
            else:
                logger.info(
                    "robot proprio SUB tcp://%s:%d — inference starts once deploy publishes g1_debug "
                    "(~50 Hz after SONIC control loop starts)",
                    args.state_zmq_host,
                    int(args.state_zmq_port),
                )
        elif args.wait_deploy_state:
            _wait_for_deploy_state(
                args.state_zmq_host.strip(),
                int(args.state_zmq_port),
                float(args.camera_wait_s),
            )

        zmq_enabled = not args.no_zmq
        pub: zmq.Socket | None = None
        ctx: zmq.Context | None = None
        send_lock = threading.Lock()
        if zmq_enabled:
            ctx = zmq.Context()
            pub = ctx.socket(zmq.PUB)
            pub.bind(f"tcp://{args.zmq_host}:{args.zmq_port}")
            time.sleep(0.5)
            logger.info("bound tcp://%s:%d", args.zmq_host, args.zmq_port)
        else:
            logger.info("ZMQ disabled — outputs will be saved to npz only")

        inference_queue: queue.Queue = queue.Queue(maxsize=1)
        result_queue: queue.Queue = queue.Queue(maxsize=1)
        stop_event = threading.Event()
        busy_event = threading.Event()

        deploy_keyboard: DeployKeyboardListener | None = None
        if zmq_enabled and args.deploy_keyboard:
            assert pub is not None
            deploy_keyboard = DeployKeyboardListener(
                pub=pub,
                stop_event=stop_event,
                send_lock=send_lock,
                on_quit=lambda: stop_event.set(),
            )
            deploy_keyboard.start()

        robot_proprio_ready_logged = [False]
        gt_fallback_logged = [False]
        defer_inference_without_robot = proprio_source in {
            "robot",
            "hybrid",
        } and not bool(args.seed_proprio) and gt_proprio is None
        defer_inference_logged = [False]

        worker = threading.Thread(
            target=_inference_worker,
            kwargs={
                "inference_queue": inference_queue,
                "result_queue": result_queue,
                "stop_event": stop_event,
                "busy_event": busy_event,
                "session": session,
                "model": model,
                "processor": processor,
                "camera": camera,
                "get_control_idx": _control_idx,
                "prompt": prompt,
                "chunk_h": chunk_h,
                "recorder": recorder,
                "robot_proprio": robot_proprio,
                "gt_proprio": gt_proprio,
                "proprio_source": proprio_source,
                "robot_proprio_ready_logged": robot_proprio_ready_logged,
                "gt_fallback_logged": gt_fallback_logged,
                "defer_inference_without_robot": defer_inference_without_robot,
                "defer_inference_logged": defer_inference_logged,
            },
            name="phi0_closed_loop_infer",
            daemon=True,
        )
        worker.start()

        control_fps = float(args.control_fps)
        loop_period = 1.0 / control_fps
        inference_interval = 1.0 / max(float(args.inference_rate), 0.1)
        action_horizon = int(chunk_h)

        cached_chunk: ActionChunk | None = None
        action_chunk_index = 0
        last_inference_time = 0.0
        zmq_frame_counter = 0
        deploy_started = False
        gt_max_frames = (
            getattr(camera, "max_control_frames", None)
            if str(args.camera_source).strip().lower() == "gt"
            else None
        )
        deadline = (
            time.monotonic() + float(args.motion_seconds)
            if float(args.motion_seconds) > 0
            else None
        )

        cam_mode = (
            f"GT ep{int(args.gt_camera_episode)}"
            if str(args.camera_source).strip().lower() == "gt"
            else f"SONIC live tcp://{args.camera_host}:{args.camera_port}"
        )
        logger.info(
            "closed loop: camera=%s control=%.1fHz infer<=%.2fHz chunk_h=%d proprio=%s zmq=%s",
            cam_mode,
            control_fps,
            float(args.inference_rate),
            chunk_h,
            proprio_source,
            "on" if zmq_enabled else "off (npz only)",
        )

        try:
            while deadline is None or time.monotonic() < deadline:
                if gt_max_frames is not None and total_frames_sent >= int(gt_max_frames):
                    logger.info("GT episode exhausted (%d frames)", total_frames_sent)
                    break
                t_start = time.monotonic()
                if robot_proprio is not None:
                    robot_proprio.poll()

                try:
                    chunk, infer_t0 = result_queue.get_nowait()
                    delay = time.monotonic() - infer_t0
                    action_chunk_index = calculate_latency_compensated_index(
                        delay, control_fps, action_horizon
                    )
                    cached_chunk = chunk
                    last_inference_time = time.monotonic()
                    logger.info(
                        "new chunk idx=%d latency=%.3fs token0=%+.3f",
                        action_chunk_index,
                        delay,
                        float(chunk.tokens[action_chunk_index, 0]),
                    )
                    if zmq_enabled and args.stream_now and not deploy_started:
                        assert pub is not None
                        time.sleep(float(args.start_delay_s))
                        _send_deploy_start_commands(pub, send_lock=send_lock)
                        deploy_started = True
                except queue.Empty:
                    pass

                if cached_chunk is None:
                    if should_trigger_new_inference(
                        cached_chunk_exists=False,
                        inference_thread_running=busy_event.is_set(),
                        time_since_last_inference=time.monotonic() - last_inference_time,
                        inference_interval=inference_interval,
                    ):
                        try:
                            inference_queue.put_nowait(None)
                        except queue.Full:
                            pass
                    _sleep_remaining(t_start, loop_period)
                    continue

                if should_trigger_new_inference(
                    cached_chunk_exists=True,
                    inference_thread_running=busy_event.is_set(),
                    time_since_last_inference=time.monotonic() - last_inference_time,
                    inference_interval=inference_interval,
                ):
                    try:
                        inference_queue.put_nowait(None)
                    except queue.Full:
                        pass

                idx = min(action_chunk_index, cached_chunk.horizon - 1)
                ramp = hand_ramp_weights(total_frames_sent + 1, int(args.hand_ramp_frames))[-1]
                tok = cached_chunk.tokens[idx]
                lh = cached_chunk.left[idx] * ramp
                rh = cached_chunk.right[idx] * ramp
                if zmq_enabled:
                    assert pub is not None
                    msg = pack_latent_action_message(
                        motion_token=tok,
                        frame_index=np.array([zmq_frame_counter], dtype=np.int64),
                        left_hand_joints=lh,
                        right_hand_joints=rh,
                    )
                    with send_lock:
                        pub.send(msg)
                if recorder is not None:
                    recorder.record_output(
                        token=tok,
                        left=lh,
                        right=rh,
                        control_idx=_control_idx(),
                        chunk_idx=idx,
                        frame_index=zmq_frame_counter,
                        hand_ramp=float(ramp),
                        timestamp=time.monotonic(),
                    )
                zmq_frame_counter += 1
                total_frames_sent += 1
                action_chunk_index = min(action_chunk_index + 1, action_horizon - 1)

                if total_frames_sent == 1 or total_frames_sent % 50 == 0:
                    verb = "tx" if zmq_enabled else "rec"
                    logger.info(
                        "%s frame=%d chunk_i=%d token0=%+.3f R_hand0=%+.3f",
                        verb,
                        total_frames_sent,
                        idx,
                        float(cached_chunk.tokens[idx, 0]),
                        float(cached_chunk.right[idx, 0]),
                    )

                _sleep_remaining(t_start, loop_period)
        except KeyboardInterrupt:
            logger.info("stopped by user")
        finally:
            stop_event.set()
            worker.join(timeout=2.0)
            if pub is not None:
                pub.close()
            if ctx is not None:
                ctx.term()
            logger.info(
                "%s %d frames",
                "sent" if zmq_enabled else "recorded",
                total_frames_sent,
            )
            if recorder is not None and obs_path is not None and output_path is not None:
                recorder.save(obs_path, output_path)
            if robot_proprio is not None:
                robot_proprio.close()
    finally:
        camera.close()


def main() -> None:
    args = parse_args()
    if args.no_zmq and not args.record_dir.strip() and not (
        args.record_obs.strip() and args.record_output.strip()
    ):
        raise SystemExit("--no-zmq requires --record-dir (or --record-obs and --record-output)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_closed_loop(args)


if __name__ == "__main__":
    main()
