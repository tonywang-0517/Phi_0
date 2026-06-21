from __future__ import annotations

import torch

from phi0.benchmark.bridge_head import (
    BridgeHeadConfig,
    bridge_logits_to_action,
    build_bridge_head,
    load_bridge_checkpoint,
    save_bridge_checkpoint,
)


def test_bridge_mlp_output_shape():
    cfg = BridgeHeadConfig(input_dim=156, hidden_dim=64, num_layers=2, head_type="mlp")
    model = build_bridge_head(cfg)
    x = torch.randn(8, 156)
    y = model(x)
    assert y.shape == (8, 7)


def test_bridge_logits_to_action_gripper_range():
    logits = torch.randn(6, 7)
    action = bridge_logits_to_action(logits)
    assert action.shape == logits.shape
    assert torch.all(action[:, 6] >= 0.0)
    assert torch.all(action[:, 6] <= 1.0)


def test_bridge_checkpoint_roundtrip(tmp_path):
    cfg = BridgeHeadConfig(input_dim=32, hidden_dim=16, num_layers=1, head_type="mlp")
    model = build_bridge_head(cfg)
    ckpt = tmp_path / "bridge.pt"
    save_bridge_checkpoint(ckpt, model, config=cfg, input_mode="latent_norm", extra={"tag": "unit"})
    loaded, payload = load_bridge_checkpoint(ckpt)
    assert payload["input_mode"] == "latent_norm"
    assert payload["extra"]["tag"] == "unit"
    for k, v in model.state_dict().items():
        assert torch.allclose(v, loaded.state_dict()[k])
