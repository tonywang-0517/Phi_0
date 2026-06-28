"""Pack recorded G1 WBC joints (pick-tissue GR00T export) into HGPT 36-d body qpos."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from phi0.data.sonic_unified_io import QPOS_SLICES, take_slices
from phi0.schema.unified_action_schema import (
    NUM_G1_BODY_QPOS,
    root_trans_world_from_unified,
    unpack_root_quat_wxyz,
)


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def body_dof29_from_wbc43(wbc43: np.ndarray) -> np.ndarray:
    """29 body DoF: legs(12) + waist(3) + arms(14); hands live in ``346:360``."""
    vec = np.asarray(wbc43, dtype=np.float32).reshape(43)
    return take_slices(vec, QPOS_SLICES).astype(np.float32)


def g1_body_qpos36_from_unified_frame(
    unified_action: np.ndarray,
    wbc43: np.ndarray,
    *,
    state_root_trans_world: np.ndarray,
    root_quat_wxyz: np.ndarray | None = None,
) -> np.ndarray:
    """Build HGPT 36-d qpos: root xyz from unified ``0:9``, quat from robot WBC when given."""
    unified = np.asarray(unified_action, dtype=np.float32).reshape(-1)
    anchor = np.asarray(state_root_trans_world, dtype=np.float32).reshape(3)
    wbc = np.asarray(wbc43, dtype=np.float32).reshape(43)
    out = np.empty(NUM_G1_BODY_QPOS, dtype=np.float32)
    out[:3] = root_trans_world_from_unified(unified, anchor)
    if root_quat_wxyz is not None:
        out[3:7] = _normalize_quat_wxyz(root_quat_wxyz)
    else:
        out[3:7] = unpack_root_quat_wxyz(unified)
    out[7:] = body_dof29_from_wbc43(wbc)
    return out


def g1_body_qpos36_from_groot_row(
    row: Mapping[str, Any],
    *,
    use_action_wbc: bool = True,
    unified_action: np.ndarray | None = None,
    state_root_trans_world: np.ndarray | None = None,
) -> np.ndarray:
    """Build HGPT ``g1_body_qpos_36`` from one GR00T teleop row (+ optional unified)."""
    key = "action.wbc" if use_action_wbc else "observation.state"
    wbc = np.asarray(row[key], dtype=np.float32).reshape(43)
    if unified_action is not None and state_root_trans_world is not None:
        return g1_body_qpos36_from_unified_frame(
            unified_action,
            wbc,
            state_root_trans_world=state_root_trans_world,
        )
    from phi0.data.groot_unified_io import read_groot_base_trans_world

    root_quat = _normalize_quat_wxyz(row.get("teleop.body_quat_w", row["observation.root_orientation"]))
    out = np.empty(NUM_G1_BODY_QPOS, dtype=np.float32)
    out[:3] = read_groot_base_trans_world(row)
    out[3:7] = root_quat
    out[7:] = body_dof29_from_wbc43(wbc)
    return out


def g1_body_qpos36_batch_from_groot_rows(
    rows: list[Mapping[str, Any]],
    *,
    use_action_wbc: bool = True,
) -> np.ndarray:
    return np.stack(
        [g1_body_qpos36_from_groot_row(r, use_action_wbc=use_action_wbc) for r in rows],
        axis=0,
    )
