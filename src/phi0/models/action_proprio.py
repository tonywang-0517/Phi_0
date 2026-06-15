"""DiT4DiT-style proprio prefix: past actions encoded and prepended to the action sequence."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def split_proprio_future(
    action: torch.Tensor,
    past_window: int,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Split ``[B,T,D]`` into proprio ``[B,W,D]`` and future ``[B,T-W,D]``."""
    w = int(past_window)
    if w <= 0:
        return None, action
    if action.shape[1] <= w:
        raise ValueError(f"action T={action.shape[1]} must exceed past_action_window_size={w}")
    return action[:, :w], action[:, w:]


def split_proprio_future_pad(
    action_is_pad: Optional[torch.Tensor],
    past_window: int,
) -> Optional[torch.Tensor]:
    if action_is_pad is None or int(past_window) <= 0:
        return action_is_pad
    return action_is_pad[:, int(past_window) :]


def split_proprio_future_dim_pad(
    action_dim_is_pad: Optional[torch.Tensor],
    past_window: int,
) -> Optional[torch.Tensor]:
    if action_dim_is_pad is None or int(past_window) <= 0:
        return action_dim_is_pad
    pad = action_dim_is_pad
    if pad.ndim == 3:
        return pad[:, int(past_window) :]
    if pad.ndim == 2 and pad.shape[0] > int(past_window):
        return pad[int(past_window) :]
    return pad


def merge_proprio_action_embeddings(
    proprio_encoder: nn.Module,
    action_encoder: nn.Module,
    proprio_tokens: Optional[torch.Tensor],
    action_tokens: torch.Tensor,
    *,
    position_embedding: Optional[nn.Embedding] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Encode action chunk; optionally prepend proprio tokens (no pos embed on proprio)."""
    batch_size, action_seq_len, _ = action_tokens.shape
    action_emb = action_encoder(action_tokens)
    if position_embedding is not None:
        pos_ids = torch.arange(action_seq_len, device=action_emb.device, dtype=torch.long)
        action_emb = action_emb + position_embedding(pos_ids).unsqueeze(0)

    proprio_len = 0
    if proprio_tokens is not None and proprio_tokens.shape[1] > 0:
        proprio_emb = proprio_encoder(proprio_tokens)
        tokens = torch.cat([proprio_emb, action_emb], dim=1)
        proprio_len = int(proprio_tokens.shape[1])
    else:
        tokens = action_emb

    meta = {
        "batch_size": batch_size,
        "action_seq_len": action_seq_len,
        "proprio_len": proprio_len,
        "total_seq_len": tokens.shape[1],
    }
    return tokens, meta
