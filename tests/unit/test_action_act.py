"""Unit tests for ACT action chunk head."""

from __future__ import annotations

import torch

from phi0.inference.session import ActionInferenceSession
from phi0.models.action_act_dit import ActionACTDiT
from phi0.models.factory_smoke import create_phi0_action_only_smoke


def test_action_act_dit_forward_shape():
    expert = ActionACTDiT(
        hidden_dim=64,
        raw_action_dim=16,
        ffn_dim=128,
        text_dim=32,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=32,
        num_layers=2,
        max_seq_len=32,
    )
    b, t, d = 2, 7, 16
    ctx = torch.randn(b, 8, 32)
    history = torch.randn(b, t, d)
    out = expert(history, ctx)
    assert out.shape == (b, t, d)


def test_training_loss_act():
    model = create_phi0_action_only_smoke(
        device="cpu", torch_dtype=torch.float32, action_head="act", past_action_window_size=2
    )
    model.loss_lambda_bone = 0.0
    model.loss_lambda_bone_hand = 0.0
    model.loss_lambda_bone_dir = 0.0
    model.train()
    assert model.action_head == "act"
    assert model.action_fm is None
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


def test_act_predict_routes_through_session():
    model = create_phi0_action_only_smoke(
        device="cpu", torch_dtype=torch.float32, action_head="act", past_action_window_size=4
    )
    model.eval()
    session = ActionInferenceSession(model)
    img0 = torch.rand(1, 3, 480, 640) * 2.0 - 1.0
    session.prefill_from_image(img0, "pick up cup")
    session.seed_proprio_from_normalized(torch.randn(model.action_expert.raw_action_dim))
    pred = session.predict(7)
    assert pred.shape == (7, model.action_expert.raw_action_dim)


def test_action_head_switch_invalid():
    try:
        create_phi0_action_only_smoke(action_head="invalid")
    except ValueError as exc:
        assert "action_head" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown action_head")
