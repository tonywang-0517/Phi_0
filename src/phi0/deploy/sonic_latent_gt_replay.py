"""Load pick-tissue parquet rows for SONIC motion_token ZMQ v4 GT replay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from phi0.deploy.sonic_zmq_io import unified_action_denorm_to_zmq_arrays
from phi0.schema.unified_action_schema import SLICES, SONIC_MOTION_TOKEN_DIM

from gear_sonic.utils.teleop.zmq.v4_latent_replay import (
    prebuild_latent_action_messages,
)

_TOKEN_SLICE = slice(*SLICES["sonic_motion_token_64"])


def _stack_list_column(table, col_name: str) -> np.ndarray:
    return np.stack(table.column(col_name).to_numpy()).astype(np.float32, copy=False)


def resolve_token_source(df_columns: set[str] | frozenset[str], token_source: str) -> str:
    if token_source != "auto":
        return token_source
    if "unified_action" in df_columns:
        return "unified_slice"
    if "action.motion_token" in df_columns:
        return "valid_column"
    raise ValueError("parquet has neither unified_action nor action.motion_token")


def load_tokens_from_table(table, token_source: str) -> np.ndarray:
    cols = set(table.column_names)
    source = resolve_token_source(cols, token_source)
    if source == "valid_column":
        tokens = _stack_list_column(table, "action.motion_token")
    elif source == "unified_slice":
        ua = _stack_list_column(table, "unified_action")
        tokens = ua[:, _TOKEN_SLICE]
    else:
        raise ValueError(f"unknown token_source: {source}")
    if tokens.shape[1] != SONIC_MOTION_TOKEN_DIM:
        raise ValueError(f"expected token dim {SONIC_MOTION_TOKEN_DIM}, got {tokens.shape[1]}")
    return tokens


def load_hands_from_table(table) -> tuple[np.ndarray, np.ndarray]:
    left = _stack_list_column(table, "teleop.left_hand_joints")
    right = _stack_list_column(table, "teleop.right_hand_joints")
    return left, right


def load_hands_from_unified_table(table) -> tuple[np.ndarray, np.ndarray]:
    """Dex3 7+7 from ``unified_action`` gripper slice, reordered for deploy ZMQ."""
    ua = _stack_list_column(table, "unified_action")
    _, left, right = unified_action_denorm_to_zmq_arrays(ua)
    return left, right


def _hands_degenerate(left: np.ndarray, right: np.ndarray, *, eps: float = 1e-6) -> bool:
    return bool(np.max(np.abs(left)) < eps and np.max(np.abs(right)) < eps)


def load_sonic_latent_replay_arrays(
    parquet: Path,
    *,
    token_source: str = "auto",
    valid_parquet_for_hands: Path | None = None,
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return (tokens, left_hands, right_hands, resolved_token_source)."""
    parquet = parquet.resolve()
    schema_cols = set(pq.read_schema(parquet).names)
    resolved = resolve_token_source(schema_cols, token_source)
    read_cols = ["unified_action"] if resolved == "unified_slice" else ["action.motion_token"]
    token_table = pq.read_table(parquet, columns=read_cols)
    tokens = load_tokens_from_table(token_table, resolved)

    hand_source = "teleop"
    if resolved == "unified_slice":
        left, right = load_hands_from_unified_table(token_table)
        hand_source = "unified_gripper"
    elif valid_parquet_for_hands is not None:
        hand_table = pq.read_table(
            valid_parquet_for_hands.resolve(),
            columns=["teleop.left_hand_joints", "teleop.right_hand_joints"],
        )
        if hand_table.num_rows != token_table.num_rows:
            raise ValueError(
                f"hands parquet T={hand_table.num_rows} != token parquet T={token_table.num_rows}"
            )
        left, right = load_hands_from_table(hand_table)
        if _hands_degenerate(left, right):
            unified_table = pq.read_table(parquet, columns=["unified_action"])
            if unified_table.num_rows != token_table.num_rows:
                raise ValueError("unified_action row count mismatch for gripper fallback")
            left, right = load_hands_from_unified_table(unified_table)
            hand_source = "unified_gripper_fallback"
    else:
        hand_table = pq.read_table(
            parquet,
            columns=["teleop.left_hand_joints", "teleop.right_hand_joints"],
        )
        left, right = load_hands_from_table(hand_table)
        if _hands_degenerate(left, right) and resolved == "unified_slice":
            left, right = load_hands_from_unified_table(token_table)
            hand_source = "unified_gripper_fallback"

    if max_frames is not None:
        n = min(len(tokens), max_frames)
        tokens, left, right = tokens[:n], left[:n], right[:n]
    return tokens, left, right, resolved if hand_source == "teleop" else f"{resolved}+{hand_source}"


def build_replay_messages(
    tokens: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    hand_ramp_frames: int = 40,
) -> list[bytes]:
    return prebuild_latent_action_messages(
        tokens, left, right, hand_ramp_frames=hand_ramp_frames
    )
