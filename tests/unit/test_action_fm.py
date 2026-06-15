"""Unit tests for DiT4DiT-style action flow matching."""

from __future__ import annotations

import torch

from phi0.models.action_fm_dit import ActionFMDiT
from phi0.models.action_fm_scheduler import ActionFlowMatching, ActionFMConfig
from phi0.models.factory_smoke import create_phi0_action_only_smoke


def test_fm_corrupt_and_target():
    fm = ActionFlowMatching(ActionFMConfig())
    clean = torch.zeros(2, 5, 4)
    noise = torch.ones(2, 5, 4)
    t = torch.tensor([0.5, 0.25])
    noisy = fm.corrupt(clean, noise, t)
    target = fm.training_target(clean, noise)
    assert torch.allclose(noisy[0], 0.5 * noise[0])
    assert torch.allclose(target, noise - clean)


def test_action_fm_dit_forward_shape():
    expert = ActionFMDiT(
        hidden_dim=64,
        raw_action_dim=16,
        ffn_dim=128,
        text_dim=32,
        freq_dim=64,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=32,
        num_layers=2,
        max_seq_len=32,
    )
    b, t, d = 2, 7, 16
    ctx = torch.randn(b, 8, 32)
    noisy = torch.randn(b, t, d)
    ts = torch.randint(0, 1000, (b,))
    out = expert(noisy, ts, ctx)
    assert out.shape == (b, t, d)


def test_training_loss_fm():
    model = create_phi0_action_only_smoke(device="cpu", torch_dtype=torch.float32)
    model.loss_lambda_bone = 0.0
    model.loss_lambda_bone_hand = 0.0
    model.loss_lambda_bone_dir = 0.0
    model.train()
    b, t, d = 1, 5, model.action_expert.raw_action_dim
    sample = {
        "video": torch.rand(b, 3, t, 480, 640) * 2.0 - 1.0,
        "context": torch.randn(1, 4, model.text_dim),
        "context_mask": torch.ones(1, 4, dtype=torch.bool),
        "action": torch.randn(b, t, d),
        "action_is_pad": torch.zeros(b, t, dtype=torch.bool),
        "action_dim_is_pad": None,
    }
    loss, loss_dict = model.training_loss(sample)
    assert float(loss.item()) > 0
    assert "loss_action" in loss_dict


def test_fm_euler_step_recovers_clean_at_t1():
    """At t=1, one full Euler step x - v equals x0 (DiT4DiT rectified flow)."""
    fm = ActionFlowMatching(ActionFMConfig())
    clean = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    noise = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
    t = torch.tensor([1.0])
    x_t = fm.corrupt(clean, noise, t)
    velocity = fm.training_target(clean, noise)
    assert torch.allclose(x_t, noise)
    recovered = x_t - velocity
    assert torch.allclose(recovered, clean)


def test_action_fm_dit_bf16_forward():
    expert = ActionFMDiT(
        hidden_dim=64,
        raw_action_dim=16,
        ffn_dim=128,
        text_dim=32,
        freq_dim=64,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=32,
        num_layers=2,
        max_seq_len=32,
    ).to(dtype=torch.bfloat16)
    b, t, d = 2, 7, 16
    ctx = torch.randn(b, 8, 32, dtype=torch.bfloat16)
    noisy = torch.randn(b, t, d, dtype=torch.bfloat16)
    ts = torch.randint(0, 1000, (b,))
    out = expert(noisy, ts, ctx)
    assert out.shape == (b, t, d)
    assert out.dtype == torch.bfloat16


def test_fm_discretize_clamps_to_bucket_range():
    fm = ActionFlowMatching(ActionFMConfig(num_timestep_buckets=1000))
    t = torch.tensor([0.0, 1.0, 1.5])
    disc = fm.discretize_t(t)
    assert disc.tolist() == [0, 999, 999]
