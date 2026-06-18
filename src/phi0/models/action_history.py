"""Action history window: split clip into observed history + future horizon."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

DEFAULT_ACTION_HISTORY_WINDOW = 29
DEFAULT_ACTION_FUTURE_HORIZON = 29


def split_history_future(
    action: torch.Tensor,
    history_window: int,
    future_horizon: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split ``[B,T,D]`` into history ``[B,H,D]`` and future ``[B,F,D]``."""
    h = int(history_window)
    if h <= 0:
        raise ValueError(f"history_window must be positive, got {h}")
    f = int(future_horizon) if future_horizon is not None else action.shape[1] - h
    if f <= 0:
        raise ValueError(f"future_horizon must be positive, got {f}")
    if action.shape[1] < h + f:
        raise ValueError(
            f"action T={action.shape[1]} must be >= history_window + future_horizon "
            f"({h} + {f} = {h + f})"
        )
    return action[:, :h], action[:, h : h + f]


def split_history_future_pad(
    action_is_pad: Optional[torch.Tensor],
    history_window: int,
) -> Optional[torch.Tensor]:
    if action_is_pad is None or int(history_window) <= 0:
        return action_is_pad
    return action_is_pad[:, int(history_window) :]


def split_history_future_dim_pad(
    action_dim_is_pad: Optional[torch.Tensor],
    history_window: int,
) -> Optional[torch.Tensor]:
    if action_dim_is_pad is None or int(history_window) <= 0:
        return action_dim_is_pad
    pad = action_dim_is_pad
    if pad.ndim == 3:
        return pad[:, int(history_window) :]
    if pad.ndim == 2 and pad.shape[0] > int(history_window):
        return pad[int(history_window) :]
    return pad


def history_to_flow_source(
    history: torch.Tensor,
    future_horizon: int,
) -> torch.Tensor:
    """Map history ``[B,H,D]`` to FM source prior ``[B,T,D]`` (hold-last current frame)."""
    if history.ndim != 3:
        raise ValueError(f"history must be [B,H,D], got {tuple(history.shape)}")
    horizon = int(future_horizon)
    if horizon <= 0:
        raise ValueError(f"future_horizon must be positive, got {horizon}")
    current = history[:, -1:, :]
    return current.expand(-1, horizon, -1).contiguous()


def encode_action_embeddings(
    action_encoder: nn.Module,
    action_tokens: torch.Tensor,
    *,
    position_embedding: Optional[nn.Embedding] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Encode an action token sequence with optional absolute position embedding."""
    batch_size, action_seq_len, _ = action_tokens.shape
    action_emb = action_encoder(action_tokens)
    if position_embedding is not None:
        pos_ids = torch.arange(action_seq_len, device=action_emb.device, dtype=torch.long)
        action_emb = action_emb + position_embedding(pos_ids).unsqueeze(0)

    meta = {
        "batch_size": batch_size,
        "action_seq_len": action_seq_len,
        "total_seq_len": action_seq_len,
    }
    return action_emb, meta
