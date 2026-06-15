"""Tests for dual VGGT integration helpers."""

from __future__ import annotations

import torch

from phi0.models.action_act_dit import ActionACTDiT
from phi0.models.vggt.tower import VGGT_REGISTER_DIM


def test_vggt_embedding_default_init_is_active():
    """VGGT branch should contribute non-zero signal from step 0 (no zero-init residual)."""
    model = ActionACTDiT(
        hidden_dim=64,
        raw_action_dim=16,
        ffn_dim=128,
        text_dim=32,
        eps=1e-6,
        num_heads=4,
        attn_head_dim=16,
        num_layers=2,
        action_cross_attn_mode="dual_cosmos_vggt",
        vggt_dim=VGGT_REGISTER_DIM,
    )
    ctx = torch.randn(2, 32, VGGT_REGISTER_DIM)
    emb = model.vggt_embedding(ctx)
    assert emb.shape == (2, 32, 64)
    assert emb.abs().max().item() > 0.0


def test_build_inputs_video_pad_matches_latent_source():
    """Pad replacement on returned video must match what Cosmos VAE would encode."""
    from phi0.data.video_pad import apply_video_pad_replacement

    video = torch.arange(1 * 3 * 3 * 2 * 2, dtype=torch.float32).view(1, 3, 3, 2, 2)
    pad = torch.tensor([[False, True, True]])
    fixed = apply_video_pad_replacement(video, pad)
    assert not torch.equal(fixed[0, :, 1], video[0, :, 1])
    assert torch.allclose(fixed[0, :, 1], fixed[0, :, 0])
    assert torch.allclose(fixed[0, :, 2], fixed[0, :, 0])
