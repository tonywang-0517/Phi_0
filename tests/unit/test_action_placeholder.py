"""Unit tests for VLA-Adapter-aligned future action placeholders."""

from __future__ import annotations

import torch

from phi0.models.action_act_dit import ActionACTDiT
from phi0.models.action_placeholder import (
    FUTURE_PLACEHOLDER_NOISE_STD,
    apply_vla_future_placeholder_noise,
    make_future_action_placeholder,
)


def test_make_future_action_placeholder_zeros():
    x = make_future_action_placeholder(2, 5, 16, device="cpu", dtype=torch.float32)
    assert x.shape == (2, 5, 16)
    assert torch.allclose(x, torch.zeros_like(x))


def test_apply_vla_noise_matches_std():
    base = make_future_action_placeholder(4, 8, 32, device="cpu", dtype=torch.float32)
    out = apply_vla_future_placeholder_noise(base, noise_std=0.02)
    assert out.shape == base.shape
    assert not torch.allclose(out, base)
    # Shared (T,D) noise across batch — row 0 and 1 identical per timestep
    assert torch.allclose(out[0], out[1])


def test_act_expert_train_adds_noise_eval_zeros_path():
    expert = ActionACTDiT(
        hidden_dim=64,
        raw_action_dim=16,
        ffn_dim=128,
        text_dim=32,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=32,
        num_layers=1,
        proprio_window=2,
        future_placeholder_noise_std=FUTURE_PLACEHOLDER_NOISE_STD,
    )
    b, t, d = 2, 3, 16
    ctx = torch.randn(b, 4, 32)
    future = make_future_action_placeholder(b, t, d, device="cpu", dtype=torch.float32)
    proprio = torch.randn(b, 2, d)

    expert.eval()
    out_eval_a = expert(future, ctx, proprio_tokens=proprio)
    out_eval_b = expert(future, ctx, proprio_tokens=proprio)
    assert torch.allclose(out_eval_a, out_eval_b)

    expert.train()
    out_train_a = expert(future, ctx, proprio_tokens=proprio)
    out_train_b = expert(future, ctx, proprio_tokens=proprio)
    assert out_train_a.shape == (b, t, d)
    # Fixed learnable perturbation — same input => same output.
    assert torch.allclose(out_train_a, out_train_b)


def test_act_expert_gradient_checkpointing_forward():
    expert = ActionACTDiT(
        hidden_dim=64,
        raw_action_dim=16,
        ffn_dim=128,
        text_dim=32,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=32,
        num_layers=2,
        use_gradient_checkpointing=True,
    )
    b, t, d = 1, 3, 16
    ctx = torch.randn(b, 4, 32)
    future = make_future_action_placeholder(b, t, d, device="cpu", dtype=torch.float32)
    expert.train()
    out = expert(future, ctx)
    assert out.shape == (b, t, d)


def test_act_expert_noise_disabled_when_std_zero():
    expert = ActionACTDiT(
        hidden_dim=64,
        raw_action_dim=16,
        ffn_dim=128,
        text_dim=32,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=32,
        num_layers=1,
        proprio_window=0,
        future_placeholder_noise_std=0.0,
    )
    expert.train()
    b, t, d = 1, 2, 16
    ctx = torch.randn(b, 4, 32)
    future = make_future_action_placeholder(b, t, d, device="cpu", dtype=torch.float32)
    assert torch.allclose(expert(future, ctx), expert(future, ctx))
