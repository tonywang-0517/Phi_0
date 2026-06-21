"""Action-only training path: freeze VLM, train action head only."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from phi0.models.factory_smoke import create_phi0_action_only_smoke
from phi0.runtime import build_optimizer, create_phi0


def _mock_phi0_for_optimizer() -> nn.Module:
    """Minimal module mimicking frozen VLM + trainable action expert naming."""

    class MockPhi0(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vlm_tower = nn.Linear(4, 4)
            self.action_expert = nn.Linear(4, 4)
            for p in self.vlm_tower.parameters():
                p.requires_grad = False

    return MockPhi0()


def test_build_optimizer_action_only_excludes_vlm():
    model = _mock_phi0_for_optimizer()
    cfg = OmegaConf.create(
        {
            "learning_rate": 1e-5,
            "learning_rate_backbone": 1e-5,
            "learning_rate_action": 1e-4,
            "weight_decay": 0.0,
        }
    )
    optim = build_optimizer(model, cfg)
    assert len(optim.param_groups) == 1
    assert optim.param_groups[0]["lr"] == 1e-4
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable_names == {"action_expert.weight", "action_expert.bias"}


def test_action_only_training_loss_has_no_video_term():
    model = create_phi0_action_only_smoke(device="cpu", torch_dtype=torch.float32)
    model.loss_lambda_bone = 0.0
    model.loss_lambda_bone_hand = 0.0
    model.train()
    b, t, d = 1, 5, model.action_expert.raw_action_dim
    sample = {
        "input_ids": torch.ones(b, 8, dtype=torch.long),
        "attention_mask": torch.ones(b, 8, dtype=torch.bool),
        "pixel_values": torch.zeros(b, 16, 512),
        "image_grid_thw": torch.tensor([[1, 16, 16]] * b),
        "action": torch.zeros(b, t, d),
        "action_is_pad": torch.zeros(b, t, dtype=torch.bool),
        "action_dim_is_pad": None,
    }
    loss, loss_dict = model.training_loss(sample)
    assert float(loss.item()) >= 0
    assert "loss_action" in loss_dict
    assert "loss_video" not in loss_dict
    assert model.loss_lambda_video == 0.0


def test_phi0_full_config_action_only_flags():
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
        cfg = compose(config_name="train_full")
    assert float(cfg.model.loss.lambda_video) == 0.0
    assert float(cfg.model.loss.lambda_action) == 1.0
    assert bool(cfg.model.vlm.freeze) is True
    assert bool(cfg.get("save_action_expert_only", False)) is True
