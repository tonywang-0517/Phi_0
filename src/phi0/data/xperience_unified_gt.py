"""Xperience HDF5 → unified action ground truth (D_UNIFIED=512) with FK validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from phi0.schema.unified_action_schema import (
    D_UNIFIED,
    NUM_CONTACTS_BODY,
    NUM_TACTILE_FINGERTIPS,
    ROT6D_DIM,
    SLICES,
    dim_mask_for_dataset,
    joints_world_52_from_unified,
    pack_from_xperience_hdf5_frame,
    pack_unified,
    root_trans_world_from_unified,
    unsupervised_dim_mask_for_dataset,
    unpack_contacts_body21,
    unpack_joint_rot6d_local_51,
    unpack_root_rot6d,
    unpack_root_trans_local,
    unpack_tactile_fingertips_10,
)

STATS_SEMANTICS_XPERIENCE_UNIFIED = "xperience_unified_smplh_512"


@dataclass(frozen=True)
class XperienceUnifiedFrameGT:
    """One frame of unified action GT plus metadata for FK / State_t."""

    action: np.ndarray  # (D_UNIFIED,)
    frame_index: int
    state_frame_index: int
    state_root_trans_world: np.ndarray  # (3,) State_t anchor when packing
    target_root_trans_world: np.ndarray  # (3,) world pelvis at frame_index
    betas: np.ndarray  # (16,)


def read_xperience_root_trans_world(f: Mapping, t: int) -> np.ndarray:
    return f["full_body_mocap/Ts_world_root"][t][:3].astype(np.float32)


def read_xperience_betas(f: Mapping, t: int) -> np.ndarray:
    return f["full_body_mocap/betas"][t].astype(np.float32)


def pack_xperience_unified_frame_gt(
    f: Mapping,
    t: int,
    *,
    state_t: int | None = None,
) -> XperienceUnifiedFrameGT:
    """Pack unified action GT from Xperience HDF5 at frame ``t`` relative to ``state_t``."""
    state_idx = int(t if state_t is None else state_t)
    action = pack_from_xperience_hdf5_frame(f, t, state_t=state_idx)
    return XperienceUnifiedFrameGT(
        action=action,
        frame_index=int(t),
        state_frame_index=state_idx,
        state_root_trans_world=read_xperience_root_trans_world(f, state_idx),
        target_root_trans_world=read_xperience_root_trans_world(f, t),
        betas=read_xperience_betas(f, t),
    )


def write_root_trans_local(action: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """In-place safe copy with updated ``root_trans_local`` slice."""
    out = np.asarray(action, dtype=np.float32).reshape(D_UNIFIED).copy()
    out[SLICES["root_trans_local"][0] : SLICES["root_trans_local"][1]] = np.asarray(
        delta, dtype=np.float32
    ).reshape(3)
    return out


def repack_clip_root_trans_local(
    actions: np.ndarray,
    root_trans_worlds: np.ndarray,
    *,
    anchor_index: int = 0,
) -> np.ndarray:
    """Repack ``root_trans_local`` for a clip: ``root[t_i] - root[t_anchor]``."""
    actions = np.asarray(actions, dtype=np.float32)
    roots = np.asarray(root_trans_worlds, dtype=np.float32).reshape(-1, 3)
    anchor = roots[int(anchor_index)]
    out = actions.copy()
    for i in range(out.shape[0]):
        out[i] = write_root_trans_local(out[i], roots[i] - anchor)
    return out


def fk_joints_world_from_gt(
    gt: XperienceUnifiedFrameGT,
    *,
    constants: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """52 joint world positions from unified GT + State_t + betas."""
    return joints_world_52_from_unified(
        gt.action,
        state_root_trans_world=gt.state_root_trans_world,
        betas=gt.betas,
        constants=constants,
    )


def reference_joints_world_from_hdf5_quat(
    f: Mapping,
    t: int,
    *,
    constants: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Independent FK reference from HDF5 quats (not via unified pack)."""
    from phi0.viz.smplh_fk import joints_from_d_raw_batch, load_skeleton_constants, pack_xperience_frame_quat

    root = f["full_body_mocap/Ts_world_root"][t].astype(np.float32)
    body = f["full_body_mocap/body_quats"][t].astype(np.float32)
    lh = f["full_body_mocap/left_hand_quats"][t].astype(np.float32)
    rh = f["full_body_mocap/right_hand_quats"][t].astype(np.float32)
    betas = read_xperience_betas(f, t)
    d_quat = pack_xperience_frame_quat(root, body, lh, rh, betas, np.zeros(10, np.float32))
    if constants is None:
        constants = load_skeleton_constants()
    return joints_from_d_raw_batch(d_quat[None], constants, use_d_raw_betas=True)[0]


def validate_unified_gt_fk_matches_hdf5_quat(
    f: Mapping,
    t: int,
    *,
    state_t: int | None = None,
    atol: float = 1e-3,
    constants: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Return FK error metrics; raises ``AssertionError`` on failure."""
    gt = pack_xperience_unified_frame_gt(f, t, state_t=state_t)
    pred = fk_joints_world_from_gt(gt, constants=constants)
    ref = reference_joints_world_from_hdf5_quat(f, t, constants=constants)
    err = np.abs(pred - ref)
    metrics = {
        "max_abs_m": float(err.max()),
        "mean_abs_m": float(err.mean()),
        "pelvis_abs_m": float(np.linalg.norm(pred[0] - ref[0])),
    }
    if metrics["max_abs_m"] > atol:
        raise AssertionError(
            f"unified GT FK mismatch at t={t} state_t={gt.state_frame_index}: {metrics}"
        )
    return metrics


def validate_packed_action_structure(action: np.ndarray) -> None:
    """Sanity-check unified buffer layout after Xperience ingest."""
    action = np.asarray(action, dtype=np.float32).reshape(D_UNIFIED)
    unsup = unsupervised_dim_mask_for_dataset("xperience")
    if not np.allclose(action[unsup], 0.0, atol=1e-7):
        raise AssertionError("unsupervised dims must be zero for xperience")
    mask = dim_mask_for_dataset("xperience")
    rot6d = unpack_root_rot6d(action)
    if not np.all(np.isfinite(rot6d)):
        raise AssertionError("root_rot6d must be finite")
    local51 = unpack_joint_rot6d_local_51(action)
    if local51.shape != (51, ROT6D_DIM):
        raise AssertionError(f"expected joint rot6d (51, 6), got {local51.shape}")
    contacts = unpack_contacts_body21(action)
    if contacts.shape != (NUM_CONTACTS_BODY,):
        raise AssertionError(f"contacts shape {contacts.shape}")
    tactile = unpack_tactile_fingertips_10(action)
    if tactile.shape != (NUM_TACTILE_FINGERTIPS,):
        raise AssertionError(f"tactile shape {tactile.shape}")
    if not np.all(np.isfinite(action[mask])):
        raise AssertionError("supervised dims must be finite")


def validate_root_trans_local_consistency(gt: XperienceUnifiedFrameGT, atol: float = 1e-6) -> None:
    local = unpack_root_trans_local(gt.action)
    expected = gt.target_root_trans_world - gt.state_root_trans_world
    if not np.allclose(local, expected, atol=atol):
        raise AssertionError(
            f"root_trans_local mismatch: got {local}, expected {expected}"
        )
    world = root_trans_world_from_unified(gt.action, gt.state_root_trans_world)
    if not np.allclose(world, gt.target_root_trans_world, atol=atol):
        raise AssertionError(
            f"root_trans_world roundtrip failed: {world} vs {gt.target_root_trans_world}"
        )


def validate_contacts_match_hdf5(f: Mapping, t: int, action: np.ndarray, atol: float = 1e-6) -> None:
    expected = f["full_body_mocap/contacts"][t].astype(np.float32)
    got = unpack_contacts_body21(action)
    if not np.allclose(got, expected, atol=atol):
        raise AssertionError(f"contacts mismatch at t={t}")


def validate_rot6d_matches_hdf5_quats(f: Mapping, t: int, action: np.ndarray, atol: float = 1e-4) -> None:
    root_q = f["full_body_mocap/Ts_world_root"][t][3:7].astype(np.float32)
    body_q = f["full_body_mocap/body_quats"][t].astype(np.float32)
    lh_q = f["full_body_mocap/left_hand_quats"][t].astype(np.float32)
    rh_q = f["full_body_mocap/right_hand_quats"][t].astype(np.float32)
    expected = pack_unified(
        root_trans_local=np.zeros(3, dtype=np.float32),
        root_quat_wxyz=root_q,
        body_quats_wxyz=body_q,
        left_hand_quats_wxyz=lh_q,
        right_hand_quats_wxyz=rh_q,
    )
    s_rot, e_rot = SLICES["root_rot6d"]
    s_j, e_j = SLICES["joint_rot6d_local_51"]
    if not np.allclose(action[s_rot:e_rot], expected[s_rot:e_rot], atol=atol):
        raise AssertionError("root_rot6d does not match HDF5 quat ingest")
    if not np.allclose(action[s_j:e_j], expected[s_j:e_j], atol=atol):
        raise AssertionError("joint_rot6d_local does not match HDF5 quat ingest")
