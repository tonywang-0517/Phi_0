"""Offline G1 body qpos@36 labels: recorded robot joints into unified ``360:396``.

Uses ``observation.state`` (proprio feedback) for DoF29; ``action.wbc`` is the WBC
command and can differ by ~0.8 rad — wrong for GT replay ghost validation.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from phi0.data.g1_qpos_from_wbc import g1_body_qpos36_from_unified_frame
from phi0.data.sonic_unified_io import SONIC_MOTION_TOKEN_DIM
from phi0.schema.unified_action_schema import write_g1_body_qpos_36, write_sonic_motion_token_64

# GT replay / sim validation: actual measured joints, not WBC command.
_DEFAULT_USE_ACTION_WBC = False


def attach_g1_qpos_to_unified_actions(
    actions: np.ndarray,
    groot_rows: list[Mapping[str, Any]],
    *,
    state_roots: list[np.ndarray] | None = None,
    use_action_wbc: bool = _DEFAULT_USE_ACTION_WBC,
) -> np.ndarray:
    """Write recorded WBC qpos into ``actions[:, 360:396]`` in-place."""
    actions = np.asarray(actions, dtype=np.float32)
    if len(groot_rows) != len(actions):
        raise ValueError(f"rows T={len(groot_rows)} != actions T={len(actions)}")
    if state_roots is not None and len(state_roots) != len(actions):
        raise ValueError(f"state_roots T={len(state_roots)} != actions T={len(actions)}")
    key = "action.wbc" if use_action_wbc else "observation.state"
    for i in range(len(actions)):
        wbc = np.asarray(groot_rows[i][key], dtype=np.float32).reshape(43)
        if state_roots is not None:
            anchor = np.asarray(state_roots[i], dtype=np.float32).reshape(3)
        else:
            from phi0.data.groot_unified_io import read_groot_base_trans_world

            anchor = read_groot_base_trans_world(groot_rows[i])
        qpos = g1_body_qpos36_from_unified_frame(
            actions[i],
            wbc,
            state_root_trans_world=anchor,
            root_quat_wxyz=groot_rows[i]["observation.root_orientation"],
        )
        write_g1_body_qpos_36(actions[i], qpos)
    return actions


def attach_g1_qpos_to_parquet_rows(
    rows: list[Mapping[str, Any]],
    groot_rows: list[Mapping[str, Any]],
    *,
    use_action_wbc: bool = _DEFAULT_USE_ACTION_WBC,
) -> None:
    """Fill ``unified_action[360:396]`` from GR00T joints (mutates rows)."""
    if not rows:
        return
    actions = np.stack(
        [np.asarray(r["unified_action"], dtype=np.float32) for r in rows],
        axis=0,
    )
    state_roots = [np.asarray(r["state_root_trans_world"], dtype=np.float32) for r in rows]
    attach_g1_qpos_to_unified_actions(
        actions,
        groot_rows,
        state_roots=state_roots,
        use_action_wbc=use_action_wbc,
    )
    for i, row in enumerate(rows):
        row["unified_action"] = actions[i].astype(np.float32).tolist()


def attach_g1_qpos_to_single_row(
    export_row: Mapping[str, Any],
    groot_row: Mapping[str, Any],
    *,
    use_action_wbc: bool = _DEFAULT_USE_ACTION_WBC,
) -> None:
    """Convenience for one LeRobot export row + source GR00T row."""
    action = np.asarray(export_row["unified_action"], dtype=np.float32)
    anchor = np.asarray(export_row["state_root_trans_world"], dtype=np.float32).reshape(3)
    key = "action.wbc" if use_action_wbc else "observation.state"
    wbc = np.asarray(groot_row[key], dtype=np.float32).reshape(43)
    write_g1_body_qpos_36(
        action,
        g1_body_qpos36_from_unified_frame(
            action,
            wbc,
            state_root_trans_world=anchor,
            root_quat_wxyz=groot_row["observation.root_orientation"],
        ),
    )
    export_row["unified_action"] = action.astype(np.float32).tolist()  # type: ignore[index]


def attach_sonic_motion_token_to_unified_actions(
    actions: np.ndarray,
    groot_rows: list[Mapping[str, Any]],
) -> np.ndarray:
    """Write ``action.motion_token`` into ``actions[:, 396:460]`` in-place."""
    actions = np.asarray(actions, dtype=np.float32)
    if len(groot_rows) != len(actions):
        raise ValueError(f"rows T={len(groot_rows)} != actions T={len(actions)}")
    for i in range(len(actions)):
        token = np.asarray(groot_rows[i]["action.motion_token"], dtype=np.float32).reshape(-1)
        if token.shape != (SONIC_MOTION_TOKEN_DIM,):
            raise ValueError(
                f"frame {i}: expected motion_token dim {SONIC_MOTION_TOKEN_DIM}, got {token.shape}"
            )
        write_sonic_motion_token_64(actions[i], token)
    return actions


def attach_sonic_motion_token_to_parquet_rows(
    rows: list[Mapping[str, Any]],
    groot_rows: list[Mapping[str, Any]],
) -> None:
    """Fill ``unified_action[396:460]`` from GR00T motion_token (mutates rows)."""
    if not rows:
        return
    actions = np.stack(
        [np.asarray(r["unified_action"], dtype=np.float32) for r in rows],
        axis=0,
    )
    attach_sonic_motion_token_to_unified_actions(actions, groot_rows)
    for i, row in enumerate(rows):
        row["unified_action"] = actions[i].astype(np.float32).tolist()
