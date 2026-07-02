#!/usr/bin/env python3
"""Publish Phi-0 predicted SONIC motion_token (+ gripper hands) over ZMQ pose v4."""

from __future__ import annotations

import argparse
import copy
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

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
    ActionInferenceSession,
    ClipInputsCache,
    PromptEmbedCache,
    resolve_deploy_action_chunk_size,
)
from phi0.inference.rtc import create_rtc_soft_mask, validate_rtc_params  # noqa: E402
from phi0.runtime import (  # noqa: E402
    activate_cuda_device,
    apply_processor_stats_from_checkpoint,
    build_base_dataset,
    build_processor,
    create_phi0,
    prepare_model_batch,
    resolve_inference_device,
    sync_model_action_norm,
)

from phi0.deploy.deploy_inference import (  # noqa: E402
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
    # RTC flags — override model cfg rtc.* defaults
    p.add_argument("--rtc", action="store_true", default=False, help="Enable Psi0-style RTC blending")
    p.add_argument("--rtc-inference-delay", type=int, default=0, help="d: frozen steps (0 = use model cfg)")
    p.add_argument("--rtc-execution-horizon", type=int, default=0, help="s: re-query cadence (0 = use model cfg)")
    p.add_argument("--rtc-schedule", type=str, default="", help="exponential|linear|hard|simple ('' = model cfg)")
    p.add_argument(
        "--camera-host",
        type=str,
        default="",
        help="Live composed_camera server host (empty = dataset ego video).",
    )
    p.add_argument("--camera-port", type=int, default=5555)
    p.add_argument(
        "--camera-wait-s",
        type=float,
        default=30.0,
        help="Wait for first ego_view frame from camera server.",
    )
    p.add_argument(
        "--stream-now",
        action="store_true",
        help="Skip --arm-flag/--ready-flag; send deploy start + stream immediately.",
    )
    p.add_argument(
        "--record-mp4",
        type=str,
        default="",
        help="Record live ego camera to mp4 while streaming tokens (needs --camera-host).",
    )
    p.add_argument(
        "--record-fps",
        type=float,
        default=30.0,
        help="Video fps for --record-mp4 (camera may be ~30 Hz).",
    )
    p.add_argument(
        "--record-motion-npz",
        type=str,
        default="",
        help="Save streamed tokens/hands npz (default: next to --record-mp4).",
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


def _remux_mp4_h264(path: Path) -> None:
    tmp = path.with_suffix(".h264.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(tmp),
            ],
            check=True,
        )
        tmp.replace(path)
        logger.info("remuxed video to H.264: %s", path)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logger.warning("ffmpeg remux skipped (%s); raw mp4 at %s", exc, path)
        if tmp.exists():
            tmp.unlink(missing_ok=True)


        if tmp.exists():
            tmp.unlink(missing_ok=True)


class _StreamOverlayState:
    """Thread-safe index into the motion chunk being streamed to deploy."""

    def __init__(
        self,
        tokens: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        *,
        control_fps: float,
    ):
        self._lock = threading.Lock()
        self._idx = 0
        self.tokens = tokens
        self.left = left
        self.right = right
        self.control_fps = float(control_fps)
        self.num_frames = int(tokens.shape[0])

    def set_frame(self, idx: int) -> None:
        with self._lock:
            self._idx = int(idx)

    def snapshot(self) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            i = min(max(self._idx, 0), self.num_frames - 1)
        return i, self.tokens[i], self.left[i], self.right[i]


def _save_motion_trace(
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
    logger.info("saved motion trace %s (%d frames)", path, int(tokens.shape[0]))


def _render_plan_panel(
    height: int,
    width: int,
    frame_idx: int,
    num_frames: int,
    token: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    import cv2

    panel = np.zeros((height, width, 3), dtype=np.uint8)
    y = 28
    lines = [
        "Phi-0 -> SONIC plan",
        f"frame {frame_idx + 1}/{num_frames}",
        f"token dim={token.size}",
        f"  [0]={token[0]:+.3f}  [1]={token[1]:+.3f}  [2]={token[2]:+.3f}",
        f"  [3]={token[3]:+.3f}  [4]={token[4]:+.3f}",
    ]
    if token.size > 8:
        lines.append(
            f"  ... [{token[5]:+.2f}, {token[6]:+.2f}, {token[7]:+.2f}]"
        )
    for text in lines:
        cv2.putText(
            panel,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        y += 22

    def _hand_bars(label: str, vals: np.ndarray, y0: int) -> int:
        cv2.putText(
            panel,
            label,
            (12, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (120, 200, 255),
            1,
            cv2.LINE_AA,
        )
        y0 += 18
        vals = np.asarray(vals, dtype=np.float64).reshape(-1)
        bar_w = max(width - 40, 80)
        for j, v in enumerate(vals[:7]):
            frac = float(np.clip(abs(v), 0.0, 1.0))
            x0, x1 = 20, 20 + int(bar_w * frac)
            cv2.rectangle(panel, (x0, y0), (x1, y0 + 10), (80, 180, 255), -1)
            cv2.putText(
                panel,
                f"{j}:{v:+.2f}",
                (20 + bar_w + 6, y0 + 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            y0 += 16
        return y0 + 6

    y = _hand_bars("L hand cmd", left, y)
    _hand_bars("R hand cmd", right, y)
    return panel


class _EgoVideoRecorder:
    """Record ego camera + optional motion-plan panel while streaming tokens."""

    def __init__(
        self,
        camera_host: str,
        camera_port: int,
        out_path: Path,
        fps: float,
        overlay: _StreamOverlayState | None = None,
        *,
        include_plan_panel: bool = True,
        plan_panel_width: int = 360,
    ):
        self._host = camera_host
        self._port = camera_port
        self._out_path = out_path
        self._period = 1.0 / max(float(fps), 1.0)
        self._overlay = overlay
        self._include_plan_panel = include_plan_panel and overlay is not None
        self._plan_panel_width = int(plan_panel_width)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frames = 0

    def start(self) -> None:
        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="ego_mp4_recorder", daemon=True)
        self._thread.start()
        logger.info("recording ego video -> %s @ %.1f fps", self._out_path, 1.0 / self._period)

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        logger.info("saved ego video %s (%d frames)", self._out_path, self._frames)
        if self._frames > 0 and self._out_path.stat().st_size > 1000:
            _remux_mp4_h264(self._out_path)
        return self._frames

    def _run(self) -> None:
        import cv2

        from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

        client = ComposedCameraClientSensor(server_ip=self._host, port=self._port)
        writer: cv2.VideoWriter | None = None
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                msg = client.read(blocking=False)
                if not msg or not msg.get("images"):
                    time.sleep(0.01)
                    continue
                rgb = msg["images"].get("ego_view")
                if rgb is None:
                    rgb = next(iter(msg["images"].values()))
                frame = np.asarray(rgb, dtype=np.uint8)
                if frame.ndim != 3 or frame.shape[2] != 3:
                    continue
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if self._overlay is not None:
                    idx, tok, lh, rh = self._overlay.snapshot()
                    cv2.putText(
                        bgr,
                        f"deploy f{idx + 1}/{self._overlay.num_frames} tok0={tok[0]:+.3f}",
                        (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (40, 255, 40),
                        2,
                        cv2.LINE_AA,
                    )
                    if self._include_plan_panel:
                        panel = _render_plan_panel(
                            bgr.shape[0],
                            self._plan_panel_width,
                            idx,
                            self._overlay.num_frames,
                            tok,
                            lh,
                            rh,
                        )
                        bgr = np.hstack([bgr, panel])
                if writer is None:
                    h, w = bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(self._out_path), fourcc, 1.0 / self._period, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"cannot open VideoWriter: {self._out_path}")
                writer.write(bgr)
                self._frames += 1
                rem = self._period - (time.monotonic() - t0)
                if rem > 0:
                    time.sleep(rem)
        finally:
            client.close()
            if writer is not None:
                writer.release()


def _fetch_live_ego_chw(camera_host: str, camera_port: int, wait_s: float) -> torch.Tensor:
    from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

    endpoint = f"tcp://{camera_host}:{camera_port}"
    logger.info("connecting live camera %s", endpoint)
    client = ComposedCameraClientSensor(server_ip=camera_host, port=camera_port)
    deadline = time.monotonic() + wait_s
    last_keys: list[str] = []
    try:
        while time.monotonic() < deadline:
            msg = client.read(blocking=False)
            if msg and msg.get("images"):
                last_keys = sorted(msg["images"].keys())
                rgb = msg["images"].get("ego_view")
                if rgb is None and last_keys:
                    rgb = msg["images"][last_keys[0]]
                    logger.warning("ego_view missing; using %s", last_keys[0])
                if rgb is not None:
                    rgb = np.asarray(rgb, dtype=np.uint8)
                    if rgb.ndim == 3 and rgb.shape[2] == 3:
                        logger.info(
                            "live ego_view %dx%d from %s",
                            rgb.shape[1],
                            rgb.shape[0],
                            endpoint,
                        )
                        return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
            time.sleep(0.05)
    finally:
        client.close()
    hint = f" keys seen: {last_keys}" if last_keys else ""
    raise TimeoutError(f"no ego_view from {endpoint} within {wait_s:.0f}s.{hint}")


def _build_clip_inputs(
    model,
    processor,
    eval_ds,
    clip_row: int,
    *,
    collate_fn,
    camera_host: str = "",
    camera_port: int = 5555,
    camera_wait_s: float = 30.0,
) -> dict:
    item = eval_ds[clip_row]
    if camera_host:
        live_ego = _fetch_live_ego_chw(camera_host, camera_port, camera_wait_s)
        item = copy.deepcopy(item)
        t_len = int(item["images"]["ego_view"].shape[0])
        item["images"]["ego_view"] = live_ego.unsqueeze(0).expand(t_len, -1, -1, -1).clone()

    batch = collate_fn([item])
    mb = prepare_model_batch(model, processor, batch)
    inputs = model.build_inputs(mb)
    if "action_ctx" not in inputs:
        inputs["action_ctx"], inputs["action_ctx_mask"] = model._resolve_action_context(
            inputs=inputs
        )
    return inputs


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


def _shift_chunk_rtc(chunk: torch.Tensor, s: int) -> torch.Tensor:
    """Roll chunk forward by s steps; pad tail with last frame (align prev chunk to next query time)."""
    # chunk: [H, D]
    if chunk.ndim == 3:
        c = chunk[0]
        shifted = torch.cat([c[s:], c[-1:].expand(s, -1)], dim=0)
        return shifted.unsqueeze(0)
    shifted = torch.cat([chunk[s:], chunk[-1:].expand(s, -1)], dim=0)
    return shifted


@torch.no_grad()
def _predict_motion_deploy_rtc(
    model,
    processor,
    inputs: dict,
    *,
    num_frames: int,
    proprio_w: int,
    gt_norm_lut,
    inference_delay: int,
    execution_horizon: int,
    schedule: str = "exponential",
) -> np.ndarray:
    """RTC-blended multi-chunk deploy: re-query every execution_horizon steps."""
    from phi0.inference.deploy_align import deploy_history_control_indices

    chunk_h = resolve_deploy_action_chunk_size(model)
    validate_rtc_params(chunk_h, inference_delay, execution_horizon)

    session = ActionInferenceSession(model, processor=processor, use_gt_history=True)
    session.prefill_from_clip_inputs(inputs)
    history_w = _history_window(model)
    device = model.device
    if hasattr(gt_norm_lut, "pin_device"):
        gt_norm_lut.pin_device(device)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16

    prev_chunk_norm: torch.Tensor | None = None
    chunks: list[np.ndarray] = []

    for seg_start in range(0, num_frames, execution_horizon):
        deploy_c = proprio_w + seg_start
        if history_w > 0:
            hist_idxs = deploy_history_control_indices(deploy_c, history_w)
            hist = torch.stack([gt_norm_lut[c] for c in hist_idxs], dim=0).to(
                device, non_blocking=True
            )
            session.set_history_gt(hist)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            if prev_chunk_norm is None:
                pred_norm = session.predict(chunk_h, denormalize=False)
            else:
                pred_norm = session.predict_rtc(
                    chunk_h,
                    prev_chunk_norm,
                    inference_delay=inference_delay,
                    execution_horizon=execution_horizon,
                    schedule=schedule,
                    denormalize=False,
                )

        # Shift for next query (align previous chunk to next time origin)
        prev_chunk_norm = _shift_chunk_rtc(
            pred_norm.detach().to(dtype=torch.float32), execution_horizon
        )

        # Take only execution_horizon steps from blended chunk
        chunk_len = min(execution_horizon, num_frames - seg_start)
        take = pred_norm[:chunk_len] if pred_norm.ndim == 2 else pred_norm[0, :chunk_len]
        if processor is not None:
            denormed = processor.postprocess(take.unsqueeze(0)).squeeze(0)
        else:
            denormed = take
        chunks.append(denormed.float().detach().cpu().numpy())

    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def _run_model_inference(
    *,
    cfg,
    ckpt_path: Path,
    device: str,
    episode_idx: int,
    num_frames: int,
    control_fps: float,
    rtc_cfg: dict | None = None,
    camera_host: str = "",
    camera_port: int = 5555,
    camera_wait_s: float = 30.0,
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
    if camera_host:
        inputs = _build_clip_inputs(
            model,
            processor,
            eval_ds,
            clip_row,
            collate_fn=collate_fn,
            camera_host=camera_host,
            camera_port=camera_port,
            camera_wait_s=camera_wait_s,
        )
    else:
        inputs = ClipInputsCache().get_or_build(
            model,
            processor,
            eval_ds,
            clip_row,
            prompt_cache=PromptEmbedCache(),
            cache_action_context=True,
            collate_fn=collate_fn,
        )
    if rtc_cfg and rtc_cfg.get("enabled"):
        logger.info(
            "RTC enabled: d=%d s=%d schedule=%s",
            rtc_cfg["inference_delay"],
            rtc_cfg["execution_horizon"],
            rtc_cfg["schedule"],
        )
        action_denorm = _predict_motion_deploy_rtc(
            model,
            processor,
            inputs,
            num_frames=num_frames,
            proprio_w=proprio_w,
            gt_norm_lut=gt_norm_lut,
            inference_delay=int(rtc_cfg["inference_delay"]),
            execution_horizon=int(rtc_cfg["execution_horizon"]),
            schedule=str(rtc_cfg["schedule"]),
        )
    else:
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
    stream_now: bool = False,
    record_mp4: str = "",
    record_fps: float = 30.0,
    record_motion_npz: str = "",
    camera_host: str = "",
    camera_port: int = 5555,
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

    if stream_now:
        time.sleep(start_delay_s)
        _send_deploy_start_commands(pub)
    else:
        if arm_flag:
            _wait_flag(Path(arm_flag), label="arm", timeout_s=arm_timeout_s)
            time.sleep(start_delay_s)
            _send_deploy_start_commands(pub)

        if ready_flag:
            _wait_flag(Path(ready_flag), label="ready", timeout_s=ready_timeout_s)
            time.sleep(start_delay_s)
            _send_deploy_start_commands(pub)

    period = 1.0 / float(control_fps)
    if record_motion_npz:
        _save_motion_trace(
            Path(record_motion_npz),
            tokens=tokens,
            left=left,
            right=right,
            episode_idx=episode_idx,
            control_fps=control_fps,
        )

    overlay = _StreamOverlayState(tokens, left, right, control_fps=control_fps)
    recorder: _EgoVideoRecorder | None = None
    if record_mp4:
        if not camera_host:
            raise ValueError("--record-mp4 requires --camera-host (live composed_camera)")
        recorder = _EgoVideoRecorder(
            camera_host,
            camera_port,
            Path(record_mp4),
            record_fps,
            overlay,
            include_plan_panel=True,
        )
        recorder.start()

    try:
        for i, msg in enumerate(messages):
            t0 = time.monotonic()
            pub.send(msg)
            overlay.set_frame(i)
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
    finally:
        if recorder is not None:
            recorder.stop()

    logger.info("done")
    pub.close()
    ctx.term()


def _resolve_rtc_cfg(cfg, args) -> dict:
    """Merge model-cfg rtc.* defaults with CLI overrides. CLI wins when non-default."""
    model_rtc = getattr(cfg, "rtc", None) or {}
    enabled = bool(getattr(model_rtc, "enabled", False)) or bool(args.rtc)
    d = int(args.rtc_inference_delay or getattr(model_rtc, "inference_delay", 2))
    s = int(args.rtc_execution_horizon or getattr(model_rtc, "execution_horizon", 4))
    schedule = str(args.rtc_schedule or getattr(model_rtc, "schedule", "exponential"))
    return {"enabled": enabled, "inference_delay": d, "execution_horizon": s, "schedule": schedule}


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
        rtc_cfg = _resolve_rtc_cfg(cfg, args)
        tokens, left, right = _run_model_inference(
            cfg=cfg,
            ckpt_path=ckpt_path,
            device=device,
            episode_idx=episode_idx,
            num_frames=num_frames,
            control_fps=timeline_fps,
            rtc_cfg=rtc_cfg,
            camera_host=args.camera_host.strip(),
            camera_port=int(args.camera_port),
            camera_wait_s=float(args.camera_wait_s),
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

    record_mp4 = args.record_mp4.strip()
    record_motion_npz = args.record_motion_npz.strip()
    if record_mp4 and not record_motion_npz:
        record_motion_npz = str(Path(record_mp4).with_name("deploy_motion.npz"))

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
        stream_now=bool(args.stream_now),
        record_mp4=record_mp4,
        record_fps=float(args.record_fps),
        record_motion_npz=record_motion_npz,
        camera_host=args.camera_host.strip(),
        camera_port=int(args.camera_port),
    )


if __name__ == "__main__":
    main()
