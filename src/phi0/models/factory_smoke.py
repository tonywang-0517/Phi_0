"""Lightweight Phi_0 factory for action-only smoke tests (no Cosmos download)."""

from __future__ import annotations

import torch

from phi0.models.cosmos.video_tower import SmokeVideoTower
from phi0.models.phi0 import Phi0, build_action_expert
from phi0.schema.draw_schema import D_RAW


def create_phi0_action_only_smoke(
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.float32,
    raw_action_dim: int = D_RAW,
    num_layers: int = 2,
    hidden_dim: int = 1024,
    text_dim: int = 512,
    action_head: str = "fm",
    past_action_window_size: int = 0,
) -> Phi0:
    """Minimal Phi0 for CPU smoke tests without HuggingFace downloads."""
    action_dit_config = {
        "hidden_dim": hidden_dim,
        "ffn_dim": 2048,
        "num_heads": 4,
        "attn_head_dim": 128,
        "num_layers": num_layers,
        "text_dim": text_dim,
        "freq_dim": 256,
        "eps": 1e-6,
        "use_gradient_checkpointing": False,
        "interleave_self_attention": True,
        "proprio_window": int(past_action_window_size),
    }
    video_tower = SmokeVideoTower(
        action_context_dim=text_dim,
        num_context_tokens=16,
        device=device,
        torch_dtype=torch_dtype,
    )
    action_expert = build_action_expert(
        action_head,
        action_dit_config,
        raw_action_dim=raw_action_dim,
        device=device,
        torch_dtype=torch_dtype,
    )
    return Phi0(
        video_tower=video_tower,
        action_expert=action_expert,
        device=device,
        torch_dtype=torch_dtype,
        loss_lambda_video=0.0,
        loss_lambda_action=1.0,
        loss_lambda_bone=0.1,
        action_head=action_head,
        past_action_window_size=int(past_action_window_size),
    )
