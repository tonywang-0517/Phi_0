"""Unit tests for DiT4DiT-style action flow matching."""

from __future__ import annotations

import torch

from phi0.models.action_fm_dit import ActionFMDiT
from phi0.models.action_fm_scheduler import ActionFlowMatching, ActionFMConfig
from phi0.models.factory_smoke import create_phi0_action_only_smoke


def test_fm_corrupt_and_target():
    fm = ActionFlowMatching(ActionFMConfig())
    clean = torch.zeros(2, 5, 4)
    source = torch.ones(2, 5, 4)
    t = torch.tensor([0.5, 0.25])
    noisy = fm.corrupt(clean, source, t)
    target = fm.training_target(clean, source)
    assert torch.allclose(noisy[0], 0.5 * source[0])
    assert torch.allclose(target, source - clean)


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
    model = create_phi0_action_only_smoke(device="cpu", torch_dtype=torch.float32, past_action_window_size=2)
    model.loss_lambda_bone = 0.0
    model.loss_lambda_bone_hand = 0.0
    model.loss_lambda_bone_dir = 0.0
    model.train()
    b, t, d = 1, 4, model.action_expert.raw_action_dim
    sample = {
        "input_ids": torch.ones(b, 8, dtype=torch.long),
        "attention_mask": torch.ones(b, 8, dtype=torch.bool),
        "pixel_values": torch.zeros(b, 16, 512),
        "image_grid_thw": torch.tensor([[1, 16, 16]] * b),
        "action": torch.randn(b, t, d),
        "action_is_pad": torch.zeros(b, t, dtype=torch.bool),
        "action_dim_is_pad": None,
    }
    loss, loss_dict = model.training_loss(sample)
    assert float(loss.item()) > 0
    assert "loss_action" in loss_dict


def test_fm_euler_step_recovers_clean_at_t1():
    """At t=1, one full Euler step x - v equals x0 (history-conditioned rectified flow)."""
    fm = ActionFlowMatching(ActionFMConfig())
    clean = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    source = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
    t = torch.tensor([1.0])
    x_t = fm.corrupt(clean, source, t)
    velocity = fm.training_target(clean, source)
    assert torch.allclose(x_t, source)
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


def test_fm_denoise_euler_gaussian_init():
    fm = ActionFlowMatching(ActionFMConfig(num_inference_timesteps=4))

    def const_velocity(actions, t_disc):
        del t_disc
        return torch.zeros_like(actions)

    out = fm.denoise_euler(
        const_velocity,
        batch_size=2,
        seq_len=5,
        action_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert out.shape == (2, 5, 4)


def test_predict_action_fm_smoke():
    model = create_phi0_action_only_smoke(device="cpu", torch_dtype=torch.float32, past_action_window_size=2)
    model.eval()
    ctx = torch.zeros(1, 16, model.text_dim, device=model.device, dtype=model.torch_dtype)
    ctx_mask = torch.ones(1, 16, device=model.device, dtype=torch.bool)
    proprio = torch.zeros(1, 2, model.action_expert.raw_action_dim, device=model.device, dtype=model.torch_dtype)
    out = model.predict_action_fm(ctx, ctx_mask, 4, proprio_tokens=proprio)
    assert out.shape == (1, 4, model.action_expert.raw_action_dim)


def test_flow_action_encoder_uses_timestep():
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
    b, t, d = 1, 5, 16
    ctx = torch.randn(b, 8, 32)
    noisy = torch.randn(b, t, d)
    out_low = expert(noisy, torch.tensor([0]), ctx)
    out_high = expert(noisy, torch.tensor([999]), ctx)
    assert not torch.allclose(out_low, out_high)
