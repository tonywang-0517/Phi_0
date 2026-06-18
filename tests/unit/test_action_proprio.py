"""DiT4DiT-style proprio prefix split + proprio-conditioned flow prior."""

from __future__ import annotations

import torch

from phi0.models.action_history import history_to_flow_source
from phi0.models.action_proprio import split_proprio_future
from phi0.models.factory_smoke import create_phi0_action_only_smoke


def test_split_proprio_future():
    action = torch.arange(30, dtype=torch.float32).reshape(1, 10, 3)
    proprio, future = split_proprio_future(action, 3)
    assert proprio.shape == (1, 3, 3)
    assert future.shape == (1, 7, 3)
    assert proprio[0, 0, 0].item() == 0.0
    assert future[0, 0, 0].item() == 9.0


def test_history_to_flow_source_hold_last():
    history = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    source = history_to_flow_source(history, 4)
    assert source.shape == (1, 4, 2)
    assert torch.allclose(source, torch.tensor([[[5.0, 6.0]] * 4]))


def test_fm_training_with_proprio_window():
    model = create_phi0_action_only_smoke(device="cpu", torch_dtype=torch.float32, past_action_window_size=2)
    model.loss_lambda_bone = 0.0
    model.loss_lambda_bone_hand = 0.0
    model.loss_lambda_bone_dir = 0.0
    model.train()
    b, t, d = 1, 4, model.action_expert.raw_action_dim
    sample = {
        "video": torch.rand(b, 3, 3, 480, 640) * 2.0 - 1.0,
        "context": torch.randn(1, 4, model.text_dim),
        "context_mask": torch.ones(1, 4, dtype=torch.bool),
        "action": torch.randn(b, t, d),
        "action_is_pad": torch.zeros(b, t, dtype=torch.bool),
        "action_dim_is_pad": None,
    }
    loss, loss_dict = model.training_loss(sample)
    assert float(loss.item()) >= 0
    assert "loss_action" in loss_dict
