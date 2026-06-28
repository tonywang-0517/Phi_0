"""ZMQ wire format: Phi-0 unified -> G1 qpos (preferred) or legacy human_data."""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import msgpack
import numpy as np

from phi0.deploy.gmr_retarget import GMR_SMPLX_BODY_NAMES

ZMQ_TOPIC = b"phi0_gmr"
DEFAULT_PORT = 5560

STREAM_FORMAT_QPOS = "qpos"
STREAM_FORMAT_HUMAN_DATA = "human_data"

G1_QPOS_DIM = 36


def _pack_bodies(human_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    return {
        name: {
            "pos": np.asarray(human_data[name][0], dtype=np.float32).reshape(3).tolist(),
            "quat": np.asarray(human_data[name][1], dtype=np.float32).reshape(4).tolist(),
        }
        for name in GMR_SMPLX_BODY_NAMES
    }


def _unpack_bodies(bodies: Dict[str, Any]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in GMR_SMPLX_BODY_NAMES:
        entry = bodies[name]
        out[name] = (
            np.asarray(entry["pos"], dtype=np.float32).reshape(3),
            np.asarray(entry["quat"], dtype=np.float32).reshape(4),
        )
    return out


def encode_meta(
    *,
    clip_idx: int,
    num_frames: int,
    control_fps: float = 20.0,
    episode_idx: int | None = None,
    proprio_w: int | None = None,
    dataset: str | None = None,
    pick_tissue_root: str | None = None,
    pick_tissue_repo_id: str | None = None,
    stream_format: str = STREAM_FORMAT_QPOS,
    qpos_postprocessed: bool = True,
    ema_alpha: float | None = None,
    deploy_mode: str | None = None,
) -> bytes:
    msg: Dict[str, Any] = {
        "type": "meta",
        "clip_idx": int(clip_idx),
        "num_frames": int(num_frames),
        "control_fps": float(control_fps),
        "stream_format": str(stream_format),
        "t_wall": time.time(),
    }
    if stream_format == STREAM_FORMAT_QPOS:
        msg["qpos_dim"] = G1_QPOS_DIM
        msg["qpos_postprocessed"] = bool(qpos_postprocessed)
        if ema_alpha is not None:
            msg["ema_alpha"] = float(ema_alpha)
        if deploy_mode is not None:
            msg["deploy_mode"] = str(deploy_mode)
    else:
        msg["body_names"] = list(GMR_SMPLX_BODY_NAMES)
    if episode_idx is not None:
        msg["episode_idx"] = int(episode_idx)
    if proprio_w is not None:
        msg["proprio_w"] = int(proprio_w)
    if dataset is not None:
        msg["dataset"] = str(dataset)
    if pick_tissue_root is not None:
        msg["pick_tissue_root"] = str(pick_tissue_root)
    if pick_tissue_repo_id is not None:
        msg["pick_tissue_repo_id"] = str(pick_tissue_repo_id)
    return msgpack.packb(msg, use_bin_type=True)


def encode_qpos_frame(
    seq: int,
    qpos: np.ndarray,
    *,
    gripper_joints_14: np.ndarray | None = None,
    is_last: bool = False,
) -> bytes:
    msg: Dict[str, Any] = {
        "type": "frame",
        "seq": int(seq),
        "is_last": bool(is_last),
        "qpos": np.asarray(qpos, dtype=np.float32).reshape(G1_QPOS_DIM).tolist(),
        "t_wall": time.time(),
    }
    if gripper_joints_14 is not None:
        msg["gripper_joints_14"] = (
            np.asarray(gripper_joints_14, dtype=np.float32).reshape(14).tolist()
        )
    return msgpack.packb(msg, use_bin_type=True)


def encode_frame(
    seq: int,
    human_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    *,
    root_quat_wxyz: np.ndarray | None = None,
    gripper_joints_14: np.ndarray | None = None,
    is_last: bool = False,
) -> bytes:
    """Legacy wire format: 14-body human_data (subscriber runs GMR)."""
    msg: Dict[str, Any] = {
        "type": "frame",
        "seq": int(seq),
        "is_last": bool(is_last),
        "bodies": _pack_bodies(human_data),
        "t_wall": time.time(),
    }
    if root_quat_wxyz is not None:
        msg["root_quat"] = np.asarray(root_quat_wxyz, dtype=np.float32).reshape(4).tolist()
    if gripper_joints_14 is not None:
        msg["gripper_joints_14"] = (
            np.asarray(gripper_joints_14, dtype=np.float32).reshape(14).tolist()
        )
    return msgpack.packb(msg, use_bin_type=True)


def encode_done(*, num_frames: int) -> bytes:
    return msgpack.packb(
        {"type": "done", "num_frames": int(num_frames), "t_wall": time.time()},
        use_bin_type=True,
    )


def decode_message(payload: bytes) -> Dict[str, Any]:
    msg = msgpack.unpackb(payload, raw=False)
    if msg.get("type") == "frame":
        if "qpos" in msg:
            msg["qpos"] = np.asarray(msg["qpos"], dtype=np.float32).reshape(G1_QPOS_DIM)
        elif "bodies" in msg:
            msg["human_data"] = _unpack_bodies(msg["bodies"])
            del msg["bodies"]
        if "root_quat" in msg:
            msg["root_quat"] = np.asarray(msg["root_quat"], dtype=np.float32).reshape(4)
        if "gripper_joints_14" in msg:
            msg["gripper_joints_14"] = np.asarray(
                msg["gripper_joints_14"], dtype=np.float32
            ).reshape(14)
    return msg


def stream_format_from_meta(meta: Dict[str, Any] | None) -> str:
    if not meta:
        return STREAM_FORMAT_HUMAN_DATA
    fmt = str(meta.get("stream_format", STREAM_FORMAT_HUMAN_DATA))
    if fmt == STREAM_FORMAT_QPOS:
        return STREAM_FORMAT_QPOS
    return STREAM_FORMAT_HUMAN_DATA
