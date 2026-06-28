"""Unified action -> G1 qpos@20Hz for ZMQ (FK + GMR + postprocess in one place)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from phi0.deploy.gt_io import DeployGtBackend, denorm_to_human_frames
from phi0.schema.unified_action_schema import (
    D_UNIFIED,
    unpack_g1_body_qpos_36,
    unpack_g1_body_qpos_dof_29,
    unpack_root_quat_wxyz,
    root_trans_world_from_unified,
)

if TYPE_CHECKING:
    from phi0.deploy.gmr_retarget import GmrRetargetSession

# Same default as Humanoid-GPT tracking/constants.DEFAULT_QPOS (36-d body qpos).
DEFAULT_QPOS_FULL = np.float32([
    0, 0, 0.78, 1, 0, 0, 0,
    -0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0,
    0.2, 0.3, 0, 1.28, 0, 0, 0, 0.2, -0.3, 0, 1.28, 0, 0, 0,
])

G1_QPOS_DIM = 36

DEPLOY_MODE_SMPL = "smpl"
DEPLOY_MODE_QPOS = "qpos"
DEPLOY_MODES = (DEPLOY_MODE_SMPL, DEPLOY_MODE_QPOS)


class RetargetHumanDataFn(Protocol):
    def __call__(self, human_data: dict) -> np.ndarray: ...


def _quat_to_yaw_wxyz(q: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in q]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def align_qpos_trajectory_to_default(
    qpos: np.ndarray,
    default: np.ndarray | None = None,
) -> np.ndarray:
    """Rebase trajectory so frame-0 root matches ``default`` while preserving relative motion."""
    from scipy.spatial.transform import Rotation as R

    if default is None:
        default = DEFAULT_QPOS_FULL
    q = np.asarray(qpos, dtype=np.float32)
    if len(q) == 0:
        return q.copy()

    root0 = q[0, :3].copy()
    r0 = R.from_quat(q[0, 3:7][[1, 2, 3, 0]])
    r_def = R.from_quat(default[3:7][[1, 2, 3, 0]])
    r0_inv = r0.inv()
    yaw_fix = R.from_euler("z", _quat_to_yaw_wxyz(default[3:7]) - _quat_to_yaw_wxyz(q[0, 3:7]))

    out = np.empty_like(q)
    for i in range(len(q)):
        dxy = q[i, :2] - root0[:2]
        dxy_r = yaw_fix.apply(np.array([dxy[0], dxy[1], 0.0], dtype=np.float64))[:2]
        out[i, 0] = default[0] + dxy_r[0]
        out[i, 1] = default[1] + dxy_r[1]
        out[i, 2] = default[2] + (q[i, 2] - root0[2])

        ri = R.from_quat(q[i, 3:7][[1, 2, 3, 0]])
        r_new = r_def * r0_inv * ri
        xyzw = r_new.as_quat()
        out[i, 3:7] = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)
        out[i, 7:] = q[i, 7:]

    out[0, :7] = default[:7]
    return out


def inject_fk_root_quat(
    qpos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    *,
    default: np.ndarray | None = None,
) -> np.ndarray:
    """Replace GMR root orientation with FK root quats (aligned to default frame-0)."""
    from scipy.spatial.transform import Rotation as R

    if default is None:
        default = DEFAULT_QPOS_FULL
    q = np.asarray(qpos, dtype=np.float32).copy()
    rq = np.asarray(root_quat_wxyz, dtype=np.float32).reshape(-1, 4)
    if rq.shape[0] != q.shape[0]:
        raise ValueError(f"root_quat rows {rq.shape[0]} != qpos rows {q.shape[0]}")

    r0 = R.from_quat(rq[0][[1, 2, 3, 0]])
    r_def = R.from_quat(default[3:7][[1, 2, 3, 0]])
    r0_inv = r0.inv()
    for i in range(len(q)):
        ri = R.from_quat(rq[i][[1, 2, 3, 0]])
        r_new = r_def * r0_inv * ri
        xyzw = r_new.as_quat()
        q[i, 3:7] = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)
    q[0, 3:7] = default[3:7]
    return q


def apply_ema_qpos(qpos: np.ndarray, *, alpha: float = 0.55) -> np.ndarray:
    """EMA smoothing with quaternion sign continuity."""
    qpos_arr = np.asarray(qpos, dtype=np.float32)
    if qpos_arr.ndim != 2 or qpos_arr.shape[1] < 7:
        raise ValueError(f"expected (T, >=7), got {qpos_arr.shape}")
    if len(qpos_arr) <= 1:
        return qpos_arr.copy()

    smoothed = np.empty_like(qpos_arr)
    smoothed[0] = qpos_arr[0]
    for t in range(1, len(qpos_arr)):
        q_prev = smoothed[t - 1, 3:7]
        q_curr = qpos_arr[t, 3:7].copy()
        if np.dot(q_prev, q_curr) < 0:
            q_curr = -q_curr
        smoothed[t, :3] = smoothed[t - 1, :3] * alpha + qpos_arr[t, :3] * (1.0 - alpha)
        smoothed[t, 7:] = smoothed[t - 1, 7:] * alpha + qpos_arr[t, 7:] * (1.0 - alpha)
        q_blended = q_prev * alpha + q_curr * (1.0 - alpha)
        q_norm = float(np.linalg.norm(q_blended))
        smoothed[t, 3:7] = q_blended / max(q_norm, 1e-8)
    return smoothed


@dataclass(frozen=True)
class PostprocessConfig:
    align_to_default: bool = True
    inject_fk_root: bool = True
    ema_alpha: float | None = 0.55
    default_qpos: np.ndarray | None = None


def postprocess_qpos_20(
    qpos_20: np.ndarray,
    root_quat_wxyz: np.ndarray,
    cfg: PostprocessConfig | None = None,
) -> np.ndarray:
    """Apply the same 20Hz postprocess formerly done on the HGPT subscriber."""
    cfg = cfg or PostprocessConfig()
    default = cfg.default_qpos if cfg.default_qpos is not None else DEFAULT_QPOS_FULL
    q = np.asarray(qpos_20, dtype=np.float32)
    rq = np.asarray(root_quat_wxyz, dtype=np.float32)
    if cfg.align_to_default:
        q = align_qpos_trajectory_to_default(q, default=default)
    if cfg.inject_fk_root:
        q = inject_fk_root_quat(q, rq, default=default)
    if cfg.ema_alpha is not None:
        q = apply_ema_qpos(q, alpha=float(cfg.ema_alpha))
    return q


def human_frames_to_qpos20(
    human_frames: list[dict],
    retarget_fn: RetargetHumanDataFn,
) -> np.ndarray:
    """GMR retarget each human_data frame -> (T, 36)."""
    out = np.empty((len(human_frames), G1_QPOS_DIM), dtype=np.float32)
    for i, human_data in enumerate(human_frames):
        np.copyto(out[i], retarget_fn(human_data))
    return out


def denorm_to_gmr_qpos(
    action_denorm: np.ndarray,
    backend: DeployGtBackend,
    retarget_fn: RetargetHumanDataFn,
    *,
    proprio_w: int,
    chunk_h: int,
    constants: dict[str, np.ndarray],
    motion_deploy: bool,
    postprocess: PostprocessConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Full pipeline: unified -> human_data -> GMR qpos (+ optional postprocess)."""
    human_frames, root_quats = denorm_to_human_frames(
        action_denorm,
        backend,
        proprio_w=proprio_w,
        chunk_h=chunk_h,
        constants=constants,
        motion_deploy=motion_deploy,
    )
    qpos_raw = human_frames_to_qpos20(human_frames, retarget_fn)
    qpos = postprocess_qpos_20(qpos_raw, root_quats, postprocess)
    return qpos, root_quats


def has_g1_body_qpos_labels(action: np.ndarray, *, atol: float = 1e-5) -> bool:
    """True when ``g1_body_qpos_36`` DoF slice looks populated (not pre-rebuild zeros)."""
    actions = np.asarray(action, dtype=np.float32)
    row = actions if actions.ndim == 1 else actions[0]
    dof = np.asarray(unpack_g1_body_qpos_dof_29(row), dtype=np.float32).reshape(29)
    return bool(np.max(np.abs(dof)) > float(atol))


def _deploy_state_roots(
    backend: DeployGtBackend | None,
    *,
    num_frames: int,
    proprio_w: int,
    chunk_h: int,
    motion_deploy: bool,
) -> np.ndarray | None:
    if backend is None:
        return None
    roots: list[np.ndarray] = []
    for i in range(num_frames):
        seg_start = (i // chunk_h) * chunk_h if motion_deploy else 0
        control_state = proprio_w + seg_start if motion_deploy else 0
        control_idx = proprio_w + i if motion_deploy else i
        _, anchor = backend.pack_deploy_frame(
            control_idx=control_idx,
            state_control_idx=control_state,
        )
        roots.append(np.asarray(anchor, dtype=np.float32).reshape(3))
    return np.stack(roots, axis=0)


def assemble_g1_body_qpos36_from_unified(
    action_denorm: np.ndarray,
    *,
    state_roots: np.ndarray | None = None,
) -> np.ndarray:
    """Build deploy qpos: root from unified ``0:9`` + anchor; DoF29 from ``360:396``."""
    actions = np.asarray(action_denorm, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < D_UNIFIED:
        raise ValueError(f"expected (T, {D_UNIFIED}) unified actions, got {actions.shape}")
    if not has_g1_body_qpos_labels(actions[0]):
        raise ValueError(
            "g1_body_qpos_36 DoF slice is empty; rebuild dataset or use --deploy-mode smpl"
        )
    if state_roots is None:
        return np.stack(
            [unpack_g1_body_qpos_36(actions[i]).astype(np.float32) for i in range(actions.shape[0])],
            axis=0,
        )
    anchors = np.asarray(state_roots, dtype=np.float32).reshape(-1, 3)
    if anchors.shape[0] != actions.shape[0]:
        raise ValueError(f"state_roots T={anchors.shape[0]} != actions T={actions.shape[0]}")
    out = np.empty((actions.shape[0], G1_QPOS_DIM), dtype=np.float32)
    for i in range(actions.shape[0]):
        stored = unpack_g1_body_qpos_36(actions[i])
        out[i, :3] = root_trans_world_from_unified(actions[i], anchors[i])
        out[i, 3:7] = stored[3:7]
        out[i, 7:] = stored[7:]
    return out


def unified_actions_to_qpos36(
    action_denorm: np.ndarray,
    *,
    state_roots: np.ndarray | None = None,
    ema_alpha: float | None = None,
) -> np.ndarray:
    """Read deploy qpos from unified (root from action; DoF from ``360:396``)."""
    qpos = assemble_g1_body_qpos36_from_unified(action_denorm, state_roots=state_roots)
    if ema_alpha is not None:
        qpos = apply_ema_qpos(qpos, alpha=float(ema_alpha))
    return qpos


def denorm_to_deploy_qpos(
    action_denorm: np.ndarray,
    backend: DeployGtBackend,
    retarget_fn: RetargetHumanDataFn | None,
    *,
    deploy_mode: str,
    proprio_w: int,
    chunk_h: int,
    constants: dict[str, np.ndarray],
    motion_deploy: bool,
    postprocess: PostprocessConfig | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Unified denorm -> G1 qpos for ZMQ (SMPL+GMR or direct qpos slice)."""
    mode = str(deploy_mode).strip().lower()
    if mode not in DEPLOY_MODES:
        raise ValueError(f"deploy_mode must be one of {DEPLOY_MODES}, got {deploy_mode!r}")
    if mode == DEPLOY_MODE_QPOS:
        cfg = postprocess or PostprocessConfig()
        ema = cfg.ema_alpha
        state_roots = _deploy_state_roots(
            backend,
            num_frames=int(action_denorm.shape[0]),
            proprio_w=proprio_w,
            chunk_h=chunk_h,
            motion_deploy=motion_deploy,
        )
        qpos_raw = assemble_g1_body_qpos36_from_unified(action_denorm, state_roots=state_roots)
        if state_roots is not None and cfg.align_to_default:
            # ponytail: qpos/WBC path keeps robot root quat from 360:396; no SMPL FK inject
            cfg_qpos = PostprocessConfig(
                align_to_default=True,
                inject_fk_root=False,
                ema_alpha=cfg.ema_alpha,
                default_qpos=cfg.default_qpos,
            )
            root_quats = qpos_raw[:, 3:7].copy()
            qpos = postprocess_qpos_20(qpos_raw, root_quats, cfg_qpos)
            return qpos, root_quats
        if ema is not None:
            qpos_raw = apply_ema_qpos(qpos_raw, alpha=float(ema))
        return qpos_raw, None
    if retarget_fn is None:
        raise ValueError("deploy_mode=smpl requires a GMR retarget_fn")
    qpos, root_quats = denorm_to_gmr_qpos(
        action_denorm,
        backend,
        retarget_fn,
        proprio_w=proprio_w,
        chunk_h=chunk_h,
        constants=constants,
        motion_deploy=motion_deploy,
        postprocess=postprocess,
    )
    return qpos, root_quats


def create_gmr_retarget_session(
    *,
    constants: dict[str, np.ndarray] | None = None,
) -> GmrRetargetSession:
    """Lazy-import GMR (requires general_motion_retargeting in the active env)."""
    from phi0.deploy.gmr_retarget import GmrRetargetSession

    return GmrRetargetSession(constants=constants)


def root_quats_from_unified_chunk(chunk: np.ndarray) -> np.ndarray:
    """FK root quat per unified row."""
    return np.stack(
        [unpack_root_quat_wxyz(chunk[i]).astype(np.float32) for i in range(chunk.shape[0])],
        axis=0,
    )
