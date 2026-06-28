"""Phi-0 unified action -> GMR human_data (smplx_to_g1).

Requires GMR in the active env only for ``unified_to_g1_qpos`` / :class:`GmrRetargetSession`.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from phi0.schema.unified_action_schema import (
    JOINTS52_TO_SMPL24,
    SMPL24_JOINT_NAMES,
    matrix_to_quat_wxyz,
    root_trans_world_from_unified,
    rot6d_to_matrix,
    unpack_joint_rot6d_local_51,
    unpack_root_rot6d,
)
from phi0.viz.smplh_fk import get_skeleton, load_skeleton_constants

# Bodies referenced by general_motion_retargeting/ik_configs/smplx_to_g1.json
GMR_SMPLX_BODY_NAMES: Tuple[str, ...] = (
    "pelvis",
    "spine3",
    "left_hip",
    "left_knee",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_foot",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)

_SMPL24_NAME_TO_IDX = {n: i for i, n in enumerate(SMPL24_JOINT_NAMES)}


def _smpl24_name_to_j52_index(name: str) -> int:
    idx24 = _SMPL24_NAME_TO_IDX[name]
    return int(JOINTS52_TO_SMPL24[idx24])


def global_poses_52_from_unified(
    d: np.ndarray,
    *,
    state_root_trans_world: np.ndarray,
    betas: np.ndarray | None = None,
    constants: dict[str, np.ndarray] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """FK global (pos, rot_mat) for all 52 SMPL-H joints in world frame.

    Returns:
        pos: (T, 52, 3) or (52, 3)
        rot: (T, 52, 3, 3) or (52, 3, 3)
    """
    from phi0.schema.draw_schema import neutral_betas

    d_arr = np.asarray(d, dtype=np.float32)
    single = d_arr.ndim == 1
    if single:
        d_arr = d_arr[np.newaxis]

    if constants is None:
        constants = load_skeleton_constants()

    batch = d_arr.shape[0]
    if betas is None:
        betas_b = neutral_betas(batch)
    else:
        betas_b = np.asarray(betas, dtype=np.float32)
        if betas_b.ndim == 1:
            betas_b = np.broadcast_to(betas_b, (batch, betas_b.shape[0]))

    skeleton = get_skeleton(betas_b, constants)
    parents = constants["parents"]

    root6 = unpack_root_rot6d(d_arr)
    local51 = unpack_joint_rot6d_local_51(d_arr)
    rot6d_52 = np.concatenate([root6[..., np.newaxis, :], local51], axis=-2)
    rot_mats = rot6d_to_matrix(rot6d_52).astype(np.float32)

    # FK chain: same as batch_rigid_transform but keep global rotations.
    prefix = rot_mats.shape[:-3]
    j_count = rot_mats.shape[-3]
    if skeleton.shape[:-2] != prefix:
        skeleton = np.broadcast_to(skeleton, prefix + (j_count, 3))

    rel_joints = skeleton.copy()
    rel_joints[..., 1:, :] -= skeleton[..., parents[1:], :]

    transforms = np.zeros(prefix + (j_count, 4, 4), dtype=np.float32)
    transforms[..., :, :3, :3] = rot_mats
    transforms[..., :, :3, 3] = rel_joints
    transforms[..., :, 3, 3] = 1.0

    chain = [transforms[..., 0, :, :]]
    for i in range(1, j_count):
        parent = int(parents[i])
        chain.append(chain[parent] @ transforms[..., i, :, :])

    global_T = np.stack(chain, axis=-3)
    global_pos = global_T[..., :3, 3].copy()
    global_rot = global_T[..., :3, :3].copy()

    root_trans = root_trans_world_from_unified(d_arr, state_root_trans_world)
    if root_trans.ndim == 1:
        root_trans = root_trans[np.newaxis, :]
    global_pos = global_pos + root_trans[:, None, :]

    if single:
        return global_pos[0], global_rot[0]
    return global_pos, global_rot


def translate_human_data_sequence(frames: list[dict]) -> list[dict]:
    """Shift all body positions by first-frame pelvis origin; keep world orientations."""
    if not frames:
        return frames
    p0 = np.asarray(frames[0]["pelvis"][0], dtype=np.float32)
    out: list[dict] = []
    for frame in frames:
        nf = {}
        for name, (pos, quat) in frame.items():
            nf[name] = (np.asarray(pos, dtype=np.float32) - p0, np.asarray(quat, dtype=np.float32))
        out.append(nf)
    foot_z = [
        min(out[i]["left_foot"][0][2], out[i]["right_foot"][0][2]) for i in range(len(out))
    ]
    z_shift = float(min(foot_z))
    if abs(z_shift) > 1e-6:
        shift = np.array([0.0, 0.0, -z_shift], dtype=np.float32)
        for frame in out:
            for name in frame:
                pos, quat = frame[name]
                frame[name] = (pos + shift, quat)
    return out


def canonicalize_human_data_sequence(
    frames: list[dict],
) -> list[dict]:
    """Pelvis-canonical + ground align (reduces GMR world-frame offset from Xperience)."""
    from scipy.spatial.transform import Rotation as R

    if not frames:
        return frames
    p0, q0 = frames[0]["pelvis"]
    r0_inv = R.from_quat(np.asarray(q0)[[1, 2, 3, 0]]).inv()

    canonical: list[dict] = []
    for frame in frames:
        nf: dict = {}
        for name, (pos, quat) in frame.items():
            p = r0_inv.apply(np.asarray(pos, dtype=np.float64) - p0)
            rq = r0_inv * R.from_quat(np.asarray(quat)[[1, 2, 3, 0]])
            wxyz = np.roll(rq.as_quat(), 1).astype(np.float32)
            nf[name] = (p.astype(np.float32), wxyz)
        canonical.append(nf)

    foot_z = [
        min(canonical[i]["left_foot"][0][2], canonical[i]["right_foot"][0][2])
        for i in range(len(canonical))
    ]
    z_shift = float(min(foot_z))
    if abs(z_shift) > 1e-6:
        shift = np.array([0.0, 0.0, -z_shift], dtype=np.float32)
        for frame in canonical:
            for name in frame:
                pos, quat = frame[name]
                frame[name] = (pos + shift, quat)
    return canonical


def unified_chunk_to_gmr_human_data_list(
    d_chunk: np.ndarray,
    *,
    state_root_trans_world: np.ndarray,
    betas: np.ndarray | None = None,
    constants: dict[str, np.ndarray] | None = None,
    body_names: Tuple[str, ...] = GMR_SMPLX_BODY_NAMES,
) -> list[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Batch FK for an entire chunk; returns one human_data dict per frame."""
    pos52, rot52 = global_poses_52_from_unified(
        d_chunk,
        state_root_trans_world=state_root_trans_world,
        betas=betas,
        constants=constants,
    )
    if pos52.ndim == 2:
        pos52 = pos52[np.newaxis]
        rot52 = rot52[np.newaxis]
    j52_indices = [_smpl24_name_to_j52_index(n) for n in body_names]
    frames: list[Dict[str, Tuple[np.ndarray, np.ndarray]]] = []
    for t in range(pos52.shape[0]):
        frame: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name, j52 in zip(body_names, j52_indices):
            quat = matrix_to_quat_wxyz(rot52[t, j52])
            frame[name] = (pos52[t, j52].astype(np.float32, copy=False), quat.astype(np.float32))
        frames.append(frame)
    return frames


def unified_to_gmr_human_data(
    d: np.ndarray,
    *,
    state_root_trans_world: np.ndarray,
    betas: np.ndarray | None = None,
    constants: dict[str, np.ndarray] | None = None,
    body_names: Tuple[str, ...] = GMR_SMPLX_BODY_NAMES,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Build one GMR ``human_data`` frame from a denormalized unified action."""
    pos52, rot52 = global_poses_52_from_unified(
        d,
        state_root_trans_world=state_root_trans_world,
        betas=betas,
        constants=constants,
    )
    human_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in body_names:
        j52 = _smpl24_name_to_j52_index(name)
        quat = matrix_to_quat_wxyz(rot52[j52])
        human_data[name] = (pos52[j52].astype(np.float32), quat.astype(np.float32))
    return human_data


def human_height_from_betas(betas: np.ndarray) -> float:
    """Same heuristic as GMR ``general_motion_retargeting.utils.smpl``."""
    b = np.asarray(betas, dtype=np.float32).reshape(-1)
    return float(1.66 + 0.1 * b[0])


def unified_to_g1_qpos(
    d: np.ndarray,
    *,
    state_root_trans_world: np.ndarray,
    betas: np.ndarray | None = None,
    constants: dict[str, np.ndarray] | None = None,
    retarget=None,
):
    """FK -> GMR.retarget -> qpos (36,). Offline smoke test only.

    For online use, prefer :class:`GmrRetargetSession` (reuses one GMR instance).
    """
    if retarget is None:
        from general_motion_retargeting import GeneralMotionRetargeting as GMR

        height = human_height_from_betas(betas) if betas is not None else None
        retarget = GMR(
            src_human="smplx",
            tgt_robot="unitree_g1",
            actual_human_height=height,
            verbose=False,
        )

    human_data = unified_to_gmr_human_data(
        d,
        state_root_trans_world=state_root_trans_world,
        betas=betas,
        constants=constants,
    )
    return retarget.retarget(human_data).astype(np.float32)


class GmrRetargetSession:
    """Reusable GMR wrapper for 20Hz online loops (init once, retarget many)."""

    def __init__(
        self,
        *,
        betas: np.ndarray | None = None,
        actual_human_height: float | None = None,
        constants: dict[str, np.ndarray] | None = None,
    ) -> None:
        from general_motion_retargeting import GeneralMotionRetargeting as GMR

        if actual_human_height is None and betas is not None:
            actual_human_height = human_height_from_betas(betas)
        self._constants = constants if constants is not None else load_skeleton_constants()
        self._retarget = GMR(
            src_human="smplx",
            tgt_robot="unitree_g1",
            actual_human_height=actual_human_height,
            verbose=False,
        )
        # Reuse output buffer to avoid per-frame allocations in hot loops.
        self._qpos_buf = np.empty(36, dtype=np.float32)

    def retarget_human_data(self, human_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        qpos = self._retarget.retarget(human_data)
        np.copyto(self._qpos_buf, qpos)
        return self._qpos_buf

    def retarget(
        self,
        d: np.ndarray,
        *,
        state_root_trans_world: np.ndarray,
        betas: np.ndarray | None = None,
    ) -> np.ndarray:
        human_data = unified_to_gmr_human_data(
            d,
            state_root_trans_world=state_root_trans_world,
            betas=betas,
            constants=self._constants,
        )
        return self.retarget_human_data(human_data)

    def retarget_chunk(
        self,
        d_chunk: np.ndarray,
        *,
        state_root_trans_world: np.ndarray,
        betas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Retarget (T,512) chunk; returns (T,36) without retaining per-frame dicts."""
        human_frames = unified_chunk_to_gmr_human_data_list(
            d_chunk,
            state_root_trans_world=state_root_trans_world,
            betas=betas,
            constants=self._constants,
        )
        out = np.empty((len(human_frames), 36), dtype=np.float32)
        for i, human_data in enumerate(human_frames):
            np.copyto(out[i], self._retarget.retarget(human_data))
        return out


def upsample_qpos_20_to_50(qpos_20hz: np.ndarray) -> np.ndarray:
    """Linear pos/dof + SLERP root quat: (T20, 36) -> (T50, 36) with ratio 2.5."""
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp

    q = np.asarray(qpos_20hz, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 36:
        raise ValueError(f"expected (T, 36), got {q.shape}")
    if q.shape[0] < 2:
        return q.copy()

    t20 = np.arange(q.shape[0], dtype=np.float64)
    t50 = np.linspace(0, q.shape[0] - 1, int(round((q.shape[0] - 1) * 2.5) + 1))

    out = np.empty((t50.shape[0], 36), dtype=np.float32)
    lin_cols = np.r_[0:3, 7:36]
    out[:, lin_cols] = interp1d(t20, q[:, lin_cols], axis=0, kind="linear")(t50).astype(np.float32)

    rots = R.from_quat(q[:, [4, 5, 6, 3]])  # wxyz -> xyzw for scipy
    xyzw = Slerp(t20, rots)(t50).as_quat()
    out[:, 3] = xyzw[:, 3]
    out[:, 4:7] = xyzw[:, 0:3]
    return out
