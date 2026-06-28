"""Phi_0 unified action token buffer (D_UNIFIED=512) with 6D rotations.

Semantic SMPL-H payload occupies ``0:346``. ``346:360`` holds G1 three-finger
gripper joint angles (7 per hand, from ``action.wbc``). ``360:396`` holds G1
MuJoCo body ``qpos`` (root xyz + quat wxyz + 29 DoF): on pick-tissue, filled from
recorded ``action.wbc`` + ``observation.root_orientation`` at dataset rebuild
(see ``g1_qpos_from_wbc``); at deploy, readable via ``DEPLOY_MODE=qpos``.
``396:460`` holds SONIC ``motion_token`` (64-d, from deploy encoder at record time).
``460:512`` is remaining reserved padding (zeroed, masked). Loss uses
``action_dim_is_pad`` — unsupervised dims contribute zero gradient (see
``Phi0._compute_action_loss``).

**Root (pelvis):**
* ``root_trans_local`` — pelvis translation **delta** vs input state ``State_t``
  (world-aligned axes): ``target_root_trans_world - state_t.root_trans_world``.
* ``root_rot6d`` — pelvis global orientation as 6D rotation (Zhou et al., continuous).

**Parent-local rotations:**
* ``joint_rot6d_local_51`` — parent-local 6D rotations for joints 1–51
  (21 body + 15 left hand + 15 right hand). Joint 0 (pelvis) is **not** duplicated here.
* ``contacts_body21``, ``tactile_fingertips_10``.

**Not stored in action (derive via FK or dataset GT):**
* ``joints_pelvis_local_52`` / ``joints_world_52`` — ``joints_*_from_unified()`` + betas + State_t.
* ``root_trans_world`` — ``root_trans_world_from_unified(action, state_t)``.
* ``smpl_pose_aa`` — from ``body_rot6d_local`` via rotation matrix → axis-angle.
* ``body_quat_wxyz`` — from ``root_rot6d``.
* ``smpl24_joints_local`` — FK subset of pelvis-local 52 joints.
* ``betas`` — FK only; not part of the action buffer.

6D convention: first two **columns** of the 3×3 rotation matrix, flattened
``[R[:,0], R[:,1]]`` (6,). Recover ``R`` via Gram–Schmidt (``rot6d_to_matrix``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

D_UNIFIED = 512
ACTION_TOKEN_DIM = D_UNIFIED  # model I/O token width

NUM_JOINTS_SMPLH = 52
NUM_LOCAL_JOINTS = NUM_JOINTS_SMPLH - 1  # 51 (exclude pelvis root rot)
NUM_BODY_JOINTS = 21
NUM_HAND_JOINTS_EACH = 15
NUM_JOINTS_SMPL24 = 24
NUM_CONTACTS_BODY = 21
NUM_TACTILE_FINGERTIPS = 10

ROT6D_DIM = 6
JOINT_ROT6D_LOCAL_DIM = NUM_LOCAL_JOINTS * ROT6D_DIM  # 306
BODY_ROT6D_DIM = NUM_BODY_JOINTS * ROT6D_DIM  # 126
HAND_ROT6D_DIM = NUM_HAND_JOINTS_EACH * ROT6D_DIM  # 90
SMPL_POSE_AA_DIM = NUM_BODY_JOINTS * 3  # export-only

ROOT_TRANS_DIM = 3
SEMANTIC_DIM = 346  # end of SMPL-H semantic fields (before robot-specific tail)

# G1 Dex3: WBC/unified stores index×2 + middle×2 + thumb×3 per hand (wbc 22–28, 36–42).
NUM_G1_GRIPPER_JOINTS_EACH = 7
NUM_G1_GRIPPER_JOINTS = NUM_G1_GRIPPER_JOINTS_EACH * 2  # 14

# Humanoid-GPT tracker body qpos: [root_xyz(3), root_quat_wxyz(4), dof29(29)].
NUM_G1_BODY_QPOS = 36
G1_BODY_QPOS_ROOT_DIM = 7
G1_BODY_QPOS_DOF_DIM = 29

SONIC_MOTION_TOKEN_DIM = 64  # matches phi0.data.sonic_unified_io
RESERVED_TAIL_DIM = (
    D_UNIFIED - SEMANTIC_DIM - NUM_G1_GRIPPER_JOINTS - NUM_G1_BODY_QPOS
)  # 116
RESERVED_PADDING_DIM = RESERVED_TAIL_DIM - SONIC_MOTION_TOKEN_DIM  # 52

# Identity rotation in 6D (first two columns of I₃).
ROT6D_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

# ---------------------------------------------------------------------------
# Flat buffer slices  [start, end)
# ---------------------------------------------------------------------------

SLICES: Dict[str, Tuple[int, int]] = {
    "root_trans_local": (0, 3),
    "root_rot6d": (3, 3 + ROT6D_DIM),  # 3:9
    "joint_rot6d_local_51": (9, 9 + JOINT_ROT6D_LOCAL_DIM),  # 9:315
    "contacts_body21": (315, 315 + NUM_CONTACTS_BODY),  # 315:336
    "tactile_fingertips_10": (336, 336 + NUM_TACTILE_FINGERTIPS),  # 336:346
    "g1_gripper_joints_14": (SEMANTIC_DIM, SEMANTIC_DIM + NUM_G1_GRIPPER_JOINTS),  # 346:360
    "g1_body_qpos_36": (
        SEMANTIC_DIM + NUM_G1_GRIPPER_JOINTS,
        SEMANTIC_DIM + NUM_G1_GRIPPER_JOINTS + NUM_G1_BODY_QPOS,
    ),  # 360:396
    "sonic_motion_token_64": (
        SEMANTIC_DIM + NUM_G1_GRIPPER_JOINTS + NUM_G1_BODY_QPOS,
        SEMANTIC_DIM + NUM_G1_GRIPPER_JOINTS + NUM_G1_BODY_QPOS + SONIC_MOTION_TOKEN_DIM,
    ),  # 396:460
    "reserved_52": (
        SEMANTIC_DIM + NUM_G1_GRIPPER_JOINTS + NUM_G1_BODY_QPOS + SONIC_MOTION_TOKEN_DIM,
        D_UNIFIED,
    ),  # 460:512
}

SUPERVISED_POSE_DIM = SEMANTIC_DIM  # 346

_R6_0 = SLICES["joint_rot6d_local_51"][0]
SLICES_ROT6D_SUB = {
    "body_rot6d_local": (_R6_0, _R6_0 + BODY_ROT6D_DIM),
    "left_hand_rot6d_local": (_R6_0 + BODY_ROT6D_DIM, _R6_0 + BODY_ROT6D_DIM + HAND_ROT6D_DIM),
    "right_hand_rot6d_local": (
        _R6_0 + BODY_ROT6D_DIM + HAND_ROT6D_DIM,
        SLICES["joint_rot6d_local_51"][1],
    ),
}

# ---------------------------------------------------------------------------
# Joint naming & SMPL-24 mapping
# ---------------------------------------------------------------------------

SMPL24_JOINT_NAMES: Tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)

JOINTS52_TO_SMPL24: np.ndarray = np.array(
    list(range(22)) + [20, 21],
    dtype=np.int32,
)

BODY21_JOINT_NAMES: Tuple[str, ...] = SMPL24_JOINT_NAMES[1:22]

MANO_TIP_INDICES: Tuple[int, ...] = (4, 8, 12, 16, 20)

TACTILE_FINGER_NAMES: Tuple[str, ...] = (
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
)

_DATASET_UNIFIED: Dict[str, bool] = {
    "root_trans_local": True,
    "root_rot6d": True,
    "joint_rot6d_local_51": True,
    "contacts_body21": True,
    "tactile_fingertips_10": False,
    "g1_gripper_joints_14": True,
    "g1_body_qpos_36": True,
    "sonic_motion_token_64": False,
    "reserved_52": False,
}

_DATASET_G1_SONIC = dict(_DATASET_UNIFIED)
_DATASET_G1_SONIC["root_trans_local"] = False  # pick-tissue: no mocap pelvis odometry GT
_DATASET_G1_SONIC["sonic_motion_token_64"] = True

_DATASET_XPERIENCE = dict(_DATASET_UNIFIED)
_DATASET_XPERIENCE["tactile_fingertips_10"] = True
_DATASET_XPERIENCE["g1_gripper_joints_14"] = False
_DATASET_XPERIENCE["g1_body_qpos_36"] = False
_DATASET_XPERIENCE["sonic_motion_token_64"] = False


@dataclass(frozen=True)
class UnifiedActionSchema:
    rep: str
    dim: int
    pose_dim_end: int
    slices: Dict[str, Tuple[int, int]]
    dataset_dim_available: Dict[str, Dict[str, bool]]


SCHEMA_UNIFIED = UnifiedActionSchema(
    rep="unified_smplh_sonic_rot6d",
    dim=D_UNIFIED,
    pose_dim_end=SUPERVISED_POSE_DIM,
    slices=dict(SLICES),
    dataset_dim_available={
        "xperience": dict(_DATASET_XPERIENCE),
        "g1_sonic": dict(_DATASET_G1_SONIC),
    },
)


def get_unified_action_schema() -> UnifiedActionSchema:
    return SCHEMA_UNIFIED


# ---------------------------------------------------------------------------
# 6D rotation math (Zhou et al.)
# ---------------------------------------------------------------------------


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3) rotation matrix (columns orthonormal)."""
    x = np.asarray(rot6d, dtype=np.float64)
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6) using the first two columns."""
    m = np.asarray(matrix, dtype=np.float64)
    return np.concatenate([m[..., :, 0], m[..., :, 1]], axis=-1).astype(np.float32)


def quat_wxyz_to_matrix(quats_wxyz: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    q = np.asarray(quats_wxyz, dtype=np.float64)
    single = q.ndim == 1
    if single:
        q = q.reshape(1, 4)
    xyzw = q[..., [1, 2, 3, 0]]
    mat = R.from_quat(xyzw.reshape(-1, 4)).as_matrix().reshape(*q.shape[:-1], 3, 3)
    return mat[0] if single else mat


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    m = np.asarray(matrix, dtype=np.float64)
    single = m.ndim == 2
    if single:
        m = m.reshape(1, 3, 3)
    xyzw = R.from_matrix(m.reshape(-1, 3, 3)).as_quat()
    wxyz = np.roll(xyzw, 1, axis=-1).reshape(*m.shape[:-2], 4)
    return wxyz[0].astype(np.float32) if single else wxyz.astype(np.float32)


def quat_wxyz_to_rot6d(quats_wxyz: np.ndarray) -> np.ndarray:
    return matrix_to_rot6d(quat_wxyz_to_matrix(quats_wxyz))


def rot6d_to_quat_wxyz(rot6d: np.ndarray) -> np.ndarray:
    return matrix_to_quat_wxyz(rot6d_to_matrix(rot6d))


def rot6d_to_axis_angle(rot6d: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    m = rot6d_to_matrix(rot6d)
    single = m.ndim == 2
    if single:
        m = m.reshape(1, 3, 3)
    aa = R.from_matrix(m.reshape(-1, 3, 3)).as_rotvec().astype(np.float32)
    aa = aa.reshape(*m.shape[:-2], 3)
    return aa[0] if single else aa


def body_rot6d_to_smpl_pose_aa(body_rot6d: np.ndarray) -> np.ndarray:
    """(21, 6) or (..., 21, 6) -> (..., 21, 3) axis-angle."""
    r = np.asarray(body_rot6d, dtype=np.float32)
    if r.ndim == 2:
        return rot6d_to_axis_angle(r.reshape(NUM_BODY_JOINTS, 6)).reshape(NUM_BODY_JOINTS, 3)
    return rot6d_to_axis_angle(r)


# ---------------------------------------------------------------------------
# Pack / unpack
# ---------------------------------------------------------------------------


def _slice_field(d: np.ndarray, name: str) -> np.ndarray:
    s, e = SLICES[name]
    return np.asarray(d, dtype=np.float32)[..., s:e]


def zeros_unified() -> np.ndarray:
    out = np.zeros(D_UNIFIED, dtype=np.float32)
    out[SLICES["root_rot6d"][0] : SLICES["root_rot6d"][1]] = ROT6D_IDENTITY
    s, e = SLICES["joint_rot6d_local_51"]
    out[s:e] = np.tile(ROT6D_IDENTITY, NUM_LOCAL_JOINTS)
    return out


def world_joints_to_pelvis_local_52(
    joints_world_52: np.ndarray,
    root_trans_world: np.ndarray | None = None,
) -> np.ndarray:
    """Utility for FK-loss GT targets (not stored in action buffer)."""
    j = np.asarray(joints_world_52, dtype=np.float32)
    single = j.ndim == 2
    if single:
        j = j[np.newaxis, ...]
    if root_trans_world is not None:
        anchor = np.asarray(root_trans_world, dtype=np.float32).reshape(-1, 1, 3)
    else:
        anchor = j[..., 0:1, :]
    local = j - anchor
    out = local.copy()
    out[..., 0, :] = 0.0
    return out[0] if single else out


def pelvis_local_joints_to_world(
    joints_pelvis_local_52: np.ndarray,
    root_trans_world: np.ndarray,
) -> np.ndarray:
    local = np.asarray(joints_pelvis_local_52, dtype=np.float32)
    root = np.asarray(root_trans_world, dtype=np.float32).reshape(-1, 3)
    single = local.ndim == 2
    if single:
        local = local[np.newaxis, ...]
        root = root.reshape(1, 3)
    world = local + root[:, None, :]
    return world[0] if single else world


def _write_rot6d_local_51(
    out: np.ndarray,
    body: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
) -> None:
    parts = [
        np.asarray(body, dtype=np.float32).reshape(NUM_BODY_JOINTS, ROT6D_DIM),
        np.asarray(left_hand, dtype=np.float32).reshape(NUM_HAND_JOINTS_EACH, ROT6D_DIM),
        np.asarray(right_hand, dtype=np.float32).reshape(NUM_HAND_JOINTS_EACH, ROT6D_DIM),
    ]
    out[SLICES["joint_rot6d_local_51"][0] : SLICES["joint_rot6d_local_51"][1]] = np.concatenate(
        [p.reshape(-1) for p in parts]
    )


def pack_unified(
    *,
    root_trans_local: np.ndarray | None = None,
    # Ingest: absolute target minus State_t anchor (world-aligned delta).
    root_trans_world: np.ndarray | None = None,
    state_root_trans_world: np.ndarray | None = None,
    root_rot6d: np.ndarray | None = None,
    joint_rot6d_local_51: np.ndarray | None = None,
    body_rot6d_local: np.ndarray | None = None,
    left_hand_rot6d_local: np.ndarray | None = None,
    right_hand_rot6d_local: np.ndarray | None = None,
    # Ingest helpers (converted to 6D; not stored as quats):
    root_quat_wxyz: np.ndarray | None = None,
    body_quats_wxyz: np.ndarray | None = None,
    left_hand_quats_wxyz: np.ndarray | None = None,
    right_hand_quats_wxyz: np.ndarray | None = None,
    contacts_body21: np.ndarray | None = None,
    tactile_fingertips_10: np.ndarray | None = None,
) -> np.ndarray:
    """Pack into D_UNIFIED. Rotations must be 6D; quat args are converted on ingest."""
    out = zeros_unified()

    if root_trans_local is not None:
        if root_trans_world is not None or state_root_trans_world is not None:
            raise ValueError("pass only root_trans_local or (root_trans_world + state_root_trans_world)")
        local = np.asarray(root_trans_local, dtype=np.float32).reshape(3)
    elif root_trans_world is not None and state_root_trans_world is not None:
        target = np.asarray(root_trans_world, dtype=np.float32).reshape(3)
        anchor = np.asarray(state_root_trans_world, dtype=np.float32).reshape(3)
        local = target - anchor
    else:
        raise ValueError("provide root_trans_local or root_trans_world + state_root_trans_world")
    out[SLICES["root_trans_local"][0] : SLICES["root_trans_local"][1]] = local

    if root_rot6d is not None:
        out[SLICES["root_rot6d"][0] : SLICES["root_rot6d"][1]] = np.asarray(
            root_rot6d, dtype=np.float32
        ).reshape(ROT6D_DIM)
    elif root_quat_wxyz is not None:
        out[SLICES["root_rot6d"][0] : SLICES["root_rot6d"][1]] = quat_wxyz_to_rot6d(
            np.asarray(root_quat_wxyz, dtype=np.float32).reshape(4)
        )
    else:
        raise ValueError("provide root_rot6d or root_quat_wxyz")

    if joint_rot6d_local_51 is not None:
        flat = np.asarray(joint_rot6d_local_51, dtype=np.float32).reshape(JOINT_ROT6D_LOCAL_DIM)
        out[SLICES["joint_rot6d_local_51"][0] : SLICES["joint_rot6d_local_51"][1]] = flat
    else:
        if body_rot6d_local is not None:
            body6 = body_rot6d_local
            lh6 = left_hand_rot6d_local
            rh6 = right_hand_rot6d_local
        elif body_quats_wxyz is not None:
            body6 = quat_wxyz_to_rot6d(np.asarray(body_quats_wxyz, dtype=np.float32))
            lh6 = quat_wxyz_to_rot6d(np.asarray(left_hand_quats_wxyz, dtype=np.float32))
            rh6 = quat_wxyz_to_rot6d(np.asarray(right_hand_quats_wxyz, dtype=np.float32))
        else:
            raise ValueError(
                "provide joint_rot6d_local_51 or body/lhand/rhand rot6d or quat triple"
            )
        _write_rot6d_local_51(out, body6, lh6, rh6)

    if contacts_body21 is not None:
        out[SLICES["contacts_body21"][0] : SLICES["contacts_body21"][1]] = np.asarray(
            contacts_body21, dtype=np.float32
        ).reshape(NUM_CONTACTS_BODY)

    if tactile_fingertips_10 is not None:
        out[SLICES["tactile_fingertips_10"][0] : SLICES["tactile_fingertips_10"][1]] = np.asarray(
            tactile_fingertips_10, dtype=np.float32
        ).reshape(NUM_TACTILE_FINGERTIPS)

    return out


def write_g1_gripper_joints_14(out: np.ndarray, joints: np.ndarray) -> None:
    """Write 14-d G1 three-finger gripper angles into unified buffer in-place."""
    j = np.asarray(joints, dtype=np.float32).reshape(NUM_G1_GRIPPER_JOINTS)
    s, e = SLICES["g1_gripper_joints_14"]
    out[s:e] = j


def write_g1_body_qpos_36(out: np.ndarray, qpos: np.ndarray) -> None:
    """Write 36-d G1 body qpos (root + 29 DoF) into unified buffer in-place."""
    q = np.asarray(qpos, dtype=np.float32).reshape(NUM_G1_BODY_QPOS)
    s, e = SLICES["g1_body_qpos_36"]
    out[s:e] = q


def write_sonic_motion_token_64(out: np.ndarray, token: np.ndarray) -> None:
    """Write 64-d SONIC motion_token latent into unified buffer in-place."""
    t = np.asarray(token, dtype=np.float32).reshape(SONIC_MOTION_TOKEN_DIM)
    s, e = SLICES["sonic_motion_token_64"]
    out[s:e] = t


def pack_unified_with_g1_gripper(
    unified: np.ndarray,
    *,
    g1_gripper_joints_14: np.ndarray | None = None,
    g1_body_qpos_36: np.ndarray | None = None,
) -> np.ndarray:
    """Attach G1 gripper / body-qpos slices to an existing unified row (copy)."""
    out = np.asarray(unified, dtype=np.float32).reshape(D_UNIFIED).copy()
    if g1_gripper_joints_14 is not None:
        write_g1_gripper_joints_14(out, g1_gripper_joints_14)
    if g1_body_qpos_36 is not None:
        write_g1_body_qpos_36(out, g1_body_qpos_36)
    return out


def unpack_root_rot6d(d: np.ndarray) -> np.ndarray:
    return _slice_field(d, "root_rot6d").reshape(*_slice_field(d, "root_rot6d").shape[:-1], ROT6D_DIM)


def unpack_joint_rot6d_local_51(d: np.ndarray) -> np.ndarray:
    flat = _slice_field(d, "joint_rot6d_local_51")
    return flat.reshape(*flat.shape[:-1], NUM_LOCAL_JOINTS, ROT6D_DIM)


def unpack_body_rot6d_local(d: np.ndarray) -> np.ndarray:
    return unpack_joint_rot6d_local_51(d)[..., :NUM_BODY_JOINTS, :]


def unpack_left_hand_rot6d_local(d: np.ndarray) -> np.ndarray:
    return unpack_joint_rot6d_local_51(d)[..., NUM_BODY_JOINTS : NUM_BODY_JOINTS + NUM_HAND_JOINTS_EACH, :]


def unpack_right_hand_rot6d_local(d: np.ndarray) -> np.ndarray:
    return unpack_joint_rot6d_local_51(d)[..., NUM_BODY_JOINTS + NUM_HAND_JOINTS_EACH :, :]


def unpack_root_quat_wxyz(d: np.ndarray) -> np.ndarray:
    """Derived quat for SONIC ``body_quat_w`` (not stored in buffer)."""
    return rot6d_to_quat_wxyz(unpack_root_rot6d(d))


def unpack_body_quats_wxyz(d: np.ndarray) -> np.ndarray:
    """Derived body quats from 6D (debug / legacy compare only)."""
    body6 = unpack_body_rot6d_local(d)
    if body6.ndim == 2:
        return rot6d_to_quat_wxyz(body6.reshape(NUM_BODY_JOINTS, 6)).reshape(NUM_BODY_JOINTS, 4)
    return rot6d_to_quat_wxyz(body6)


def unpack_contacts_body21(d: np.ndarray) -> np.ndarray:
    return _slice_field(d, "contacts_body21")


def unpack_tactile_fingertips_10(d: np.ndarray) -> np.ndarray:
    return _slice_field(d, "tactile_fingertips_10")


def unpack_g1_gripper_joints_14(d: np.ndarray) -> np.ndarray:
    return _slice_field(d, "g1_gripper_joints_14")


def unpack_g1_body_qpos_36(d: np.ndarray) -> np.ndarray:
    return _slice_field(d, "g1_body_qpos_36")


def unpack_g1_body_qpos_root_7(d: np.ndarray) -> np.ndarray:
    return unpack_g1_body_qpos_36(d)[..., :G1_BODY_QPOS_ROOT_DIM]


def unpack_g1_body_qpos_dof_29(d: np.ndarray) -> np.ndarray:
    return unpack_g1_body_qpos_36(d)[..., G1_BODY_QPOS_ROOT_DIM:]


def unpack_sonic_motion_token_64(d: np.ndarray) -> np.ndarray:
    return _slice_field(d, "sonic_motion_token_64")


def unpack_root_trans_local(d: np.ndarray) -> np.ndarray:
    return _slice_field(d, "root_trans_local")


def root_trans_world_from_unified(
    d: np.ndarray,
    state_root_trans_world: np.ndarray,
) -> np.ndarray:
    """Recover absolute pelvis translation: ``State_t.root_trans + root_trans_local``."""
    local = unpack_root_trans_local(d)
    anchor = np.asarray(state_root_trans_world, dtype=np.float32)
    if local.ndim == 1 and anchor.ndim == 1:
        return local + anchor
    if anchor.ndim == 1:
        anchor = anchor.reshape(*(1,) * (local.ndim - 1), 3)
    return local + anchor


def dim_mask_for_dataset(dataset_name: str) -> np.ndarray:
    """Per-dim supervision mask: ``True`` = has GT and receives action loss.

    Datasets set ``action_dim_is_pad = ~dim_mask_for_dataset(name)`` in the
    loader; ``Phi0._compute_action_loss`` zeroes MSE on padded dims.
    """
    schema = get_unified_action_schema()
    avail = schema.dataset_dim_available.get(dataset_name, _DATASET_UNIFIED)
    mask = np.zeros(D_UNIFIED, dtype=bool)
    for field, (s, e) in SLICES.items():
        if avail.get(field, False):
            mask[s:e] = True
    return mask


def unsupervised_dim_mask_for_dataset(dataset_name: str) -> np.ndarray:
    """Dims that must stay zero in packed GT (no loss, no semantic payload)."""
    return ~dim_mask_for_dataset(dataset_name)


def zero_unsupervised_unified_action_dims(
    action: "torch.Tensor",
    *,
    dataset_name: str = "g1_sonic",
) -> "torch.Tensor":
    """Zero only unsupervised unified dims (not ``pose_dim_end:`` tail wipe)."""
    import torch

    if not isinstance(action, torch.Tensor):
        raise TypeError("zero_unsupervised_unified_action_dims expects a torch.Tensor")
    mask = unsupervised_dim_mask_for_dataset(dataset_name)
    out = action.clone()
    out[..., mask] = 0.0
    return out


def zero_unsupervised_unified_action_dims_np(
    d_raw: np.ndarray,
    *,
    dataset_name: str = "g1_sonic",
) -> np.ndarray:
    out = np.asarray(d_raw, dtype=np.float32).copy()
    out[..., unsupervised_dim_mask_for_dataset(dataset_name)] = 0.0
    return out


# ---------------------------------------------------------------------------
# FK-derived joint positions (not stored in action buffer)
# ---------------------------------------------------------------------------


def joints_world_52_from_unified(
    d: np.ndarray,
    *,
    state_root_trans_world: np.ndarray,
    betas: np.ndarray | None = None,
    constants: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """FK global 52-joint positions from unified action + State_t anchor + betas."""
    from phi0.schema.draw_schema import neutral_betas
    from phi0.viz.smplh_fk import batch_rigid_transform, get_skeleton, load_skeleton_constants

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
    root6 = unpack_root_rot6d(d_arr)
    local51 = unpack_joint_rot6d_local_51(d_arr)
    rot6d_52 = np.concatenate([root6[..., np.newaxis, :], local51], axis=-2)
    rot_mats = rot6d_to_matrix(rot6d_52).astype(np.float32)
    posed = batch_rigid_transform(rot_mats, skeleton, constants["parents"])
    root_trans = root_trans_world_from_unified(d_arr, state_root_trans_world)
    if root_trans.ndim == 1:
        root_trans = root_trans[np.newaxis, :]
    posed = posed + root_trans[:, None, :]
    return posed[0] if single else posed


def joints_pelvis_local_52_from_unified(
    d: np.ndarray,
    *,
    state_root_trans_world: np.ndarray,
    betas: np.ndarray | None = None,
    constants: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """FK pelvis-local 52-joint offsets (joint 0 = 0)."""
    world = joints_world_52_from_unified(
        d,
        state_root_trans_world=state_root_trans_world,
        betas=betas,
        constants=constants,
    )
    root_trans = root_trans_world_from_unified(d, state_root_trans_world)
    return world_joints_to_pelvis_local_52(world, root_trans)


def smpl24_joints_local_from_unified(
    d: np.ndarray,
    *,
    state_root_trans_world: np.ndarray,
    betas: np.ndarray | None = None,
    constants: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    j52 = joints_pelvis_local_52_from_unified(
        d,
        state_root_trans_world=state_root_trans_world,
        betas=betas,
        constants=constants,
    )
    if j52.ndim == 2:
        return j52[JOINTS52_TO_SMPL24]
    return j52[..., JOINTS52_TO_SMPL24, :]


def smpl_pose_aa_from_unified(d: np.ndarray) -> np.ndarray:
    return body_rot6d_to_smpl_pose_aa(unpack_body_rot6d_local(d))


# ---------------------------------------------------------------------------
# SONIC export / HDF5 pack
# ---------------------------------------------------------------------------


def tactile_proxy_from_mano_joints(joints_3d: np.ndarray) -> np.ndarray:
    from phi0.schema.draw_schema import compute_tactile_proxy_from_mano_joints

    return compute_tactile_proxy_from_mano_joints(joints_3d)


def export_sonic_zmq_v3_fields(
    d: np.ndarray,
    *,
    state_root_trans_world: np.ndarray,
    betas: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    return {
        "root_trans": root_trans_world_from_unified(d, state_root_trans_world).astype(np.float32),
        "body_quat_w": unpack_root_quat_wxyz(d).astype(np.float32),
        "smpl_joints": smpl24_joints_local_from_unified(
            d,
            state_root_trans_world=state_root_trans_world,
            betas=betas,
        ).astype(np.float32),
        "smpl_pose": smpl_pose_aa_from_unified(d).astype(np.float32),
    }


def pack_from_xperience_hdf5_frame(
    f: Mapping,
    t: int,
    *,
    state_t: int | None = None,
) -> np.ndarray:
    """Build unified buffer from Xperience HDF5 (quats converted to 6D on ingest).

    ``root_trans_local`` is ``Ts_world_root[t] - Ts_world_root[state_t]`` (default ``state_t=t``).
    """
    state_idx = t if state_t is None else state_t
    root_target = f["full_body_mocap/Ts_world_root"][t].astype(np.float32)
    root_state = f["full_body_mocap/Ts_world_root"][state_idx].astype(np.float32)
    body = f["full_body_mocap/body_quats"][t].astype(np.float32)
    lh = f["full_body_mocap/left_hand_quats"][t].astype(np.float32)
    rh = f["full_body_mocap/right_hand_quats"][t].astype(np.float32)
    contacts = f["full_body_mocap/contacts"][t].astype(np.float32)

    tactile = np.zeros(NUM_TACTILE_FINGERTIPS, dtype=np.float32)
    if "hand_mocap/left_joints_3d" in f and "hand_mocap/right_joints_3d" in f:
        l_mano = f["hand_mocap/left_joints_3d"][t].astype(np.float32)
        r_mano = f["hand_mocap/right_joints_3d"][t].astype(np.float32)
        tactile[:5] = tactile_proxy_from_mano_joints(l_mano)
        tactile[5:] = tactile_proxy_from_mano_joints(r_mano)

    return pack_unified(
        root_trans_local=root_target[:3] - root_state[:3],
        root_quat_wxyz=root_target[3:7],
        body_quats_wxyz=body,
        left_hand_quats_wxyz=lh,
        right_hand_quats_wxyz=rh,
        contacts_body21=contacts,
        tactile_fingertips_10=tactile,
    )


def schema_field_table() -> str:
    lines = [
        f"D_UNIFIED={D_UNIFIED}  (rep={SCHEMA_UNIFIED.rep})",
        f"  Semantic SMPL-H: [0:{SEMANTIC_DIM}); G1 gripper [{SEMANTIC_DIM}:{SEMANTIC_DIM + NUM_G1_GRIPPER_JOINTS}); "
        f"G1 body qpos [{SEMANTIC_DIM + NUM_G1_GRIPPER_JOINTS}:"
        f"{SEMANTIC_DIM + NUM_G1_GRIPPER_JOINTS + NUM_G1_BODY_QPOS}); "
        f"sonic_motion_token [{SEMANTIC_DIM + NUM_G1_GRIPPER_JOINTS + NUM_G1_BODY_QPOS}:"
        f"{SEMANTIC_DIM + NUM_G1_GRIPPER_JOINTS + NUM_G1_BODY_QPOS + SONIC_MOTION_TOKEN_DIM}); "
        f"reserved [{SEMANTIC_DIM + NUM_G1_GRIPPER_JOINTS + NUM_G1_BODY_QPOS + SONIC_MOTION_TOKEN_DIM}:"
        f"{D_UNIFIED}).",
        "  Root trans: local delta vs State_t (world-aligned). Root rot: global 6D.",
        "  Positions: FK-derived (not in buffer); use joints_*_from_unified() + State_t + betas.",
        "  Derived at export: root_trans_world, body_quat_w, smpl_pose_aa, smpl24_joints_local.",
        "-" * 72,
    ]
    for name, (s, e) in SLICES.items():
        lines.append(f"  [{s:3d}:{e:3d}]  {e - s:4d}  {name}")
    s, e = SLICES_ROT6D_SUB["body_rot6d_local"]
    lines.append(f"       └ body_rot6d_local      [{s}:{e}]")
    s, e = SLICES_ROT6D_SUB["left_hand_rot6d_local"]
    lines.append(f"       └ left_hand_rot6d_local [{s}:{e}]")
    s, e = SLICES_ROT6D_SUB["right_hand_rot6d_local"]
    lines.append(f"       └ right_hand_rot6d_local[{s}:{e}]")
    return "\n".join(lines)
