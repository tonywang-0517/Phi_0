"""Isaac-GR00T pick-tissue teleop → Phi_0 unified 512-d action packing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from phi0.schema.draw_schema import neutral_betas
from phi0.schema.unified_action_schema import (
    D_UNIFIED,
    NUM_BODY_JOINTS,
    NUM_HAND_JOINTS_EACH,
    pack_unified,
    write_g1_gripper_joints_14,
)

from phi0.data.robot_action_norm import STATS_SEMANTICS_XPERIENCE_UNIFIED
from phi0.data.sonic_unified_io import resolve_hands

STATS_SEMANTICS_PICK_TISSUE_UNIFIED = STATS_SEMANTICS_XPERIENCE_UNIFIED
IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Fields zeroed by gear_sonic run_data_exporter when SMPL ZMQ tick is missing.
_TELEOP_SMPL_FORWARD_FILL_KEYS: tuple[str, ...] = (
    "teleop.smpl_joints",
    "teleop.smpl_pose",
    "teleop.body_quat_w",
    "teleop.target_body_orientation",
    "teleop.left_wrist_joints",
    "teleop.right_wrist_joints",
    "teleop.smpl_frame_index",
)


def is_invalid_smpl_teleop_row(row: Mapping[str, Any]) -> bool:
    """True when exporter had no SMPL message (``smpl_frame_index==0`` sentinel).

    Upstream ``run_data_exporter.py`` sets this when ``use_smpl`` is false or
    ``smpl_msg`` / ``frame_index`` is missing — not a valid SMPL stream index 0.
    """
    raw = row.get("teleop.smpl_frame_index")
    if raw is None:
        return True
    idx = int(np.asarray(raw).reshape(-1)[0])
    return idx == 0


def forward_fill_smpl_teleop_row(
    row: Mapping[str, Any],
    last_valid_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Hold last valid SMPL teleop fields; keep robot WBC/observation from ``row``."""
    out = dict(row)
    for key in _TELEOP_SMPL_FORWARD_FILL_KEYS:
        if key in last_valid_row:
            val = last_valid_row[key]
            out[key] = val.copy() if isinstance(val, np.ndarray) else val
    return out


def prepare_groot_row_for_unified(
    row: Mapping[str, Any],
    last_valid_row: Mapping[str, Any] | None,
    *,
    bootstrap_row: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Mapping[str, Any] | None, bool]:
    """Return row ready for packing; forward-fill SMPL when exporter tick was missing.

    ``bootstrap_row`` seeds leading invalid ticks from the first valid SMPL row in the
    episode (exporter warm-up before the first ZMQ SMPL message).

    Returns:
        prepared row dict, updated last_valid_row, whether SMPL was repaired.
    """
    if is_invalid_smpl_teleop_row(row):
        seed = last_valid_row if last_valid_row is not None else bootstrap_row
        if seed is None:
            raise ValueError("first row has invalid SMPL teleop; cannot forward-fill")
        prepared = forward_fill_smpl_teleop_row(row, seed)
        return prepared, last_valid_row, True
    prepared = dict(row)
    return prepared, prepared, False


def read_groot_pelvis_world(row: Mapping[str, Any]) -> np.ndarray:
    """Legacy pelvis proxy from ``teleop.smpl_joints`` joint-0 (not G1 base odometry)."""
    joints = np.asarray(row["teleop.smpl_joints"], dtype=np.float32).reshape(24, 3)
    return joints[0].copy()


def read_groot_base_trans_world(row: Mapping[str, Any]) -> np.ndarray:
    """G1 base xyz world from ``observation.base_trans`` (g1_debug ``base_trans_measured``)."""
    if "observation.base_trans" in row:
        return np.asarray(row["observation.base_trans"], dtype=np.float32).reshape(3).copy()
    # ponytail: legacy parquets pre-exporter patch; upgrade path = rebuild after re-export
    return read_groot_pelvis_world(row)


def has_groot_base_trans_odometry(row: Mapping[str, Any]) -> bool:
    """True when parquet has real G1 base xyz (not SMPL pelvis proxy)."""
    return "observation.base_trans" in row


def smpl_pose_aa_to_body_quats_wxyz(smpl_pose_aa: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    aa = np.asarray(smpl_pose_aa, dtype=np.float64).reshape(NUM_BODY_JOINTS, 3)
    xyzw = R.from_rotvec(aa).as_quat()
    wxyz = np.roll(xyzw, 1, axis=-1)
    return wxyz.astype(np.float32)


def identity_hand_quats_wxyz() -> np.ndarray:
    return np.tile(IDENTITY_QUAT_WXYZ, (NUM_HAND_JOINTS_EACH, 1))


@dataclass(frozen=True)
class GrootUnifiedFrameGT:
    action: np.ndarray
    state_root_trans_world: np.ndarray
    target_root_trans_world: np.ndarray
    betas: np.ndarray


def pack_from_groot_teleop_row(
    row: Mapping[str, Any],
    *,
    state_t_row: Mapping[str, Any] | None = None,
) -> GrootUnifiedFrameGT:
    """Pack one GR00T teleop row into unified 512-d action relative to ``state_t_row``."""
    anchor_row = row if state_t_row is None else state_t_row
    state_root = read_groot_base_trans_world(anchor_row)
    target_root = read_groot_base_trans_world(row)

    if has_groot_base_trans_odometry(anchor_row) and has_groot_base_trans_odometry(row):
        root_local = target_root - state_root
    else:
        # pick-tissue: no SMPL pelvis odometry GT; g1_sonic masks 0:3 out of loss
        root_local = np.zeros(3, dtype=np.float32)

    root_quat = np.asarray(row["teleop.body_quat_w"], dtype=np.float32).reshape(4)
    smpl_pose_aa = np.asarray(row["teleop.smpl_pose"], dtype=np.float32).reshape(-1)
    body_quats = smpl_pose_aa_to_body_quats_wxyz(smpl_pose_aa)
    left_hand_quats = identity_hand_quats_wxyz()
    right_hand_quats = identity_hand_quats_wxyz()
    contacts = np.zeros(21, dtype=np.float32)

    action = pack_unified(
        root_trans_local=root_local,
        root_quat_wxyz=root_quat,
        body_quats_wxyz=body_quats,
        left_hand_quats_wxyz=left_hand_quats,
        right_hand_quats_wxyz=right_hand_quats,
        contacts_body21=contacts,
        tactile_fingertips_10=None,
    )
    wbc = np.asarray(row.get("action.wbc"), dtype=np.float32).reshape(-1)
    fallback_wbc = None
    if state_t_row is not None and state_t_row is not row:
        fallback_wbc = np.asarray(state_t_row.get("action.wbc"), dtype=np.float32).reshape(-1)
    write_g1_gripper_joints_14(action, resolve_hands(row, wbc, fallback=fallback_wbc))
    if action.shape != (D_UNIFIED,):
        raise ValueError(f"expected unified dim {D_UNIFIED}, got {action.shape}")

    return GrootUnifiedFrameGT(
        action=action,
        state_root_trans_world=state_root,
        target_root_trans_world=target_root,
        betas=neutral_betas(1).reshape(-1),
    )


def pack_groot_unified_frame_lists(row: Mapping[str, Any]) -> dict[str, list[float]]:
    """List form for LeRobot parquet export (anchor = same frame)."""
    gt = pack_from_groot_teleop_row(row, state_t_row=row)
    return {
        "unified_action": gt.action.astype(np.float32).tolist(),
        "state_root_trans_world": gt.state_root_trans_world.astype(np.float32).tolist(),
        "target_root_trans_world": gt.target_root_trans_world.astype(np.float32).tolist(),
        "betas": gt.betas.astype(np.float32).tolist(),
    }
