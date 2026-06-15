"""Unit tests for action cross-attention ablation modes."""

from __future__ import annotations

import pytest
import torch

from phi0.models.action_cross_attn import cross_attn_target, resolve_action_cross_attn_mode
from phi0.models.action_fm_dit import ActionFMDiT
from phi0.models.vggt.tower import VGGT_REGISTER_DIM


def test_resolve_modes():
    assert resolve_action_cross_attn_mode("interleave_cosmos") == "interleave_cosmos"
    assert resolve_action_cross_attn_mode("dual_cosmos_vggt") == "dual_cosmos_vggt"
    assert resolve_action_cross_attn_mode("all_cosmos") == "all_cosmos"
    assert resolve_action_cross_attn_mode(None, interleave_self_attention=False) == "all_cosmos"
    assert resolve_action_cross_attn_mode(None, interleave_self_attention=True) == "interleave_cosmos"


def test_cross_attn_targets_interleave():
    assert cross_attn_target("interleave_cosmos", 0) == "cosmos"
    assert cross_attn_target("interleave_cosmos", 1) is None
    assert cross_attn_target("interleave_cosmos", 2) == "cosmos"


def test_cross_attn_targets_dual():
    assert cross_attn_target("dual_cosmos_vggt", 0) == "cosmos"
    assert cross_attn_target("dual_cosmos_vggt", 1) == "vggt"
    assert cross_attn_target("dual_cosmos_vggt", 2) == "cosmos"
    assert cross_attn_target("dual_cosmos_vggt", 3) == "vggt"


def test_cross_attn_targets_all_cosmos():
    for i in range(4):
        assert cross_attn_target("all_cosmos", i) == "cosmos"


@pytest.mark.parametrize("mode", ["interleave_cosmos", "dual_cosmos_vggt", "all_cosmos"])
def test_action_fm_dit_forward_shapes(mode: str):
    b, t_act, s_c, s_v = 2, 8, 32, 16
    hidden = 256
    model = ActionFMDiT(
        hidden_dim=hidden,
        raw_action_dim=256,
        ffn_dim=512,
        text_dim=512,
        vggt_dim=VGGT_REGISTER_DIM,
        freq_dim=64,
        eps=1e-6,
        num_heads=4,
        attn_head_dim=64,
        num_layers=4,
        max_seq_len=64,
        proprio_window=0,
        action_cross_attn_mode=mode,
    )
    action = torch.randn(b, t_act, 256)
    timestep = torch.randint(0, 1000, (b,))
    cosmos_ctx = torch.randn(b, s_c, 512)
    vggt_ctx = torch.randn(b, s_v, VGGT_REGISTER_DIM) if mode == "dual_cosmos_vggt" else None

    out = model(
        action,
        timestep,
        cosmos_ctx,
        vggt_context=vggt_ctx,
    )
    assert out.shape == (b, t_act, 256)


def test_dual_mode_requires_vggt():
    model = ActionFMDiT(
        hidden_dim=256,
        raw_action_dim=256,
        ffn_dim=512,
        text_dim=512,
        freq_dim=64,
        eps=1e-6,
        num_heads=4,
        attn_head_dim=64,
        num_layers=2,
        action_cross_attn_mode="dual_cosmos_vggt",
    )
    with pytest.raises(ValueError, match="vggt_context"):
        model(
            torch.randn(1, 4, 256),
            torch.tensor([0]),
            torch.randn(1, 8, 512),
        )
