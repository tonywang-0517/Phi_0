"""Normalize / denormalize SONIC unified 43-d state and 100-d action."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from phi0.data.sonic_unified_io import (
    SONIC_ACTION_DIM,
    SONIC_STATE_DIM,
    STATS_SEMANTICS_SONIC_UNIFIED,
)
from phi0.data.simple_action_norm import (
    denormalize_robot_nd,
    normalize_robot_nd,
    stats_view_for_robot_nd,
)

__all__ = [
    "SONIC_STATE_DIM",
    "SONIC_ACTION_DIM",
    "STATS_SEMANTICS_SONIC_UNIFIED",
    "load_sonic_stats_json",
    "normalize_robot_nd",
    "denormalize_robot_nd",
    "stats_view_for_robot_nd",
]


def load_sonic_stats_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    action = raw.get("action") or raw.get("actions") or {}
    state = raw.get("states") or raw.get("state") or {}
    norm_mode = "bounds"
    if "q01" in action and "q99" in action:
        norm_mode = "bounds_q99"
    return {
        "version": 2,
        "robot_action_semantics": STATS_SEMANTICS_SONIC_UNIFIED,
        "norm_mode": norm_mode,
        "normalize_gripper": True,
        "robot_dim": SONIC_ACTION_DIM,
        "action_dim": SONIC_ACTION_DIM,
        "state_dim": SONIC_STATE_DIM,
        "num_frames": int(raw.get("num_frames", 0)),
        "mean": action.get("mean", [0.0] * SONIC_ACTION_DIM),
        "std": action.get("std", [1.0] * SONIC_ACTION_DIM),
        "q01": action.get("q01", action.get("min", [0.0] * SONIC_ACTION_DIM)),
        "q99": action.get("q99", action.get("max", [1.0] * SONIC_ACTION_DIM)),
        "state_mean": state.get("mean", [0.0] * SONIC_STATE_DIM),
        "state_std": state.get("std", [1.0] * SONIC_STATE_DIM),
        "state_q01": state.get("q01", state.get("min", [0.0] * SONIC_STATE_DIM)),
        "state_q99": state.get("q99", state.get("max", [1.0] * SONIC_STATE_DIM)),
    }
