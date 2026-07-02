"""Build pick-tissue unified proprio from SONIC deploy ``g1_debug`` (ZMQ 5557)."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from phi0.data.xperience_unified_gt import write_root_trans_local
from phi0.deploy.dex3_gripper import NUM_GRIPPER_EACH, WBC_TO_DEPLOY_HAND7_IDX
from phi0.schema.unified_action_schema import (
    D_UNIFIED,
    SEMANTIC_DIM,
    SONIC_MOTION_TOKEN_DIM,
    write_g1_body_qpos_36,
    write_g1_gripper_joints_14,
    write_sonic_motion_token_64,
    zeros_unified,
)

_DEPLOY_TO_WBC_HAND7_IDX = np.argsort(np.asarray(WBC_TO_DEPLOY_HAND7_IDX, dtype=np.int64))


def deploy_hand7_to_wbc(hand7: np.ndarray) -> np.ndarray:
    """Inverse of ``wbc_hand7_to_deploy`` (deploy ZMQ order -> unified/WBC order)."""
    h = np.asarray(hand7, dtype=np.float32).reshape(NUM_GRIPPER_EACH)
    return h[np.asarray(_DEPLOY_TO_WBC_HAND7_IDX, dtype=np.int64)]


def _vec3(msg: Mapping[str, Any], *keys: str) -> np.ndarray:
    for key in keys:
        if key in msg:
            return np.asarray(msg[key], dtype=np.float32).reshape(3)
    raise KeyError(f"expected one of {keys!r} in g1_debug; have {list(msg.keys())[:16]}")


def _quat_wxyz(msg: Mapping[str, Any], *keys: str) -> np.ndarray:
    for key in keys:
        if key in msg:
            q = np.asarray(msg[key], dtype=np.float32).reshape(4)
            n = float(np.linalg.norm(q))
            if n < 1e-8:
                return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return (q / n).astype(np.float32)
    raise KeyError(f"expected one of {keys!r} in g1_debug; have {list(msg.keys())[:16]}")


def _body_q29(msg: Mapping[str, Any]) -> np.ndarray:
    for key in ("body_q_measured", "body_q"):
        if key in msg:
            return np.asarray(msg[key], dtype=np.float32).reshape(29)
    raise KeyError("g1_debug missing body_q_measured/body_q")


def _hand7(msg: Mapping[str, Any], side: str) -> np.ndarray:
    for key in (f"{side}_hand_q_measured", f"{side}_hand_q"):
        if key in msg:
            return np.asarray(msg[key], dtype=np.float32).reshape(7)
    raise KeyError(f"g1_debug missing {side}_hand_q_measured/{side}_hand_q")


def g1_body_qpos36_from_g1_debug(msg: Mapping[str, Any]) -> np.ndarray:
    """36-d body qpos: root xyz + quat wxyz + 29 DoF (deploy MuJoCo order)."""
    out = np.empty(36, dtype=np.float32)
    out[:3] = _vec3(msg, "base_trans_measured", "base_trans", "base_trans_target")
    out[3:7] = _quat_wxyz(msg, "base_quat_measured", "base_quat", "base_quat_target")
    out[7:] = _body_q29(msg)
    return out


def gripper14_wbc_from_g1_debug(msg: Mapping[str, Any]) -> np.ndarray:
    left = deploy_hand7_to_wbc(_hand7(msg, "left"))
    right = deploy_hand7_to_wbc(_hand7(msg, "right"))
    return np.concatenate([left, right], axis=0).astype(np.float32)


def token64_from_g1_debug(msg: Mapping[str, Any]) -> np.ndarray | None:
    raw = msg.get("token_state")
    if raw is None:
        return None
    tok = np.asarray(raw, dtype=np.float32).reshape(-1)
    if tok.size != SONIC_MOTION_TOKEN_DIM:
        return None
    return tok


def unified_from_g1_debug(
    msg: Mapping[str, Any],
    *,
    semantic_base: np.ndarray | None = None,
    anchor_root: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack deploy robot state into denormalized unified 512-d proprio.

    SMPL semantic ``0:346`` comes from ``semantic_base`` (hybrid / last prediction).
    Robot-measurable tail ``346:460`` is filled from ``g1_debug``.
    """
    if semantic_base is None:
        base = zeros_unified()
    else:
        base = np.asarray(semantic_base, dtype=np.float32).reshape(D_UNIFIED).copy()
        base[SEMANTIC_DIM:] = 0.0

    out = base.copy()
    write_g1_gripper_joints_14(out, gripper14_wbc_from_g1_debug(msg))
    write_g1_body_qpos_36(out, g1_body_qpos36_from_g1_debug(msg))
    tok = token64_from_g1_debug(msg)
    if tok is not None:
        write_sonic_motion_token_64(out, tok)

    target_root = _vec3(msg, "base_trans_measured", "base_trans", "base_trans_target")
    if anchor_root is None:
        anchor_root = target_root
    anchor = np.asarray(anchor_root, dtype=np.float32).reshape(3)
    d_raw = write_root_trans_local(out, target_root - anchor)
    return d_raw, target_root.copy()


def normalize_unified_proprio(processor, d_raw: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(d_raw, dtype=np.float32)).float().unsqueeze(0)
    return processor._normalize_action(t).reshape(-1)


class RobotProprioSource:
    """Poll ``g1_debug`` and expose normalized unified proprio for deploy inference."""

    def __init__(self, host: str, port: int) -> None:
        from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber

        self._sub = ZMQStateSubscriber(host=host, port=int(port))
        self._anchor_root: np.ndarray | None = None
        self._semantic_base: np.ndarray | None = None
        self._last_msg: dict[str, Any] | None = None

    def close(self) -> None:
        self._sub.close()

    def set_semantic_base(self, unified_denorm: np.ndarray | None) -> None:
        if unified_denorm is None:
            self._semantic_base = None
            return
        self._semantic_base = np.asarray(unified_denorm, dtype=np.float32).reshape(D_UNIFIED).copy()

    def poll(self) -> dict[str, Any] | None:
        msg = self._sub.get_msg(clear=True)
        if msg is not None:
            self._last_msg = msg
        return msg

    @property
    def ready(self) -> bool:
        return self._last_msg is not None

    def wait_first(self, timeout_s: float = 30.0) -> dict[str, Any]:
        deadline = __import__("time").monotonic() + float(timeout_s)
        while __import__("time").monotonic() < deadline:
            msg = self.poll()
            if msg is not None:
                return msg
            __import__("time").sleep(0.02)
        raise TimeoutError(f"no g1_debug within {timeout_s:.0f}s")

    def build_normalized(
        self,
        processor,
        *,
        msg: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        state = msg if msg is not None else self._last_msg
        if state is None:
            raise RuntimeError("no g1_debug received yet")
        d_raw, root = unified_from_g1_debug(
            state,
            semantic_base=self._semantic_base,
            anchor_root=self._anchor_root,
        )
        if self._anchor_root is None:
            self._anchor_root = root
        return normalize_unified_proprio(processor, d_raw)
