"""Cosmos DiT hook-layer early exit (action-only / frozen transformer)."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from phi0.models.cosmos.hook_forward import forward_transformer_to_hook_layer
from phi0.models.cosmos.video_tower import CosmosVideoTower


class _CountBlock(nn.Module):
    def __init__(self, idx: int, dim: int = 4) -> None:
        super().__init__()
        self.idx = idx
        self.linear = nn.Linear(dim, dim, bias=False)
        self.call_count = 0

    def forward(self, hidden_states, *_args, **_kwargs):
        self.call_count += 1
        return self.linear(hidden_states)


def _minimal_mock_transformer(num_layers: int = 6) -> nn.Module:
    """Tiny stand-in with the same forward contract as hook_forward expects."""
    from types import SimpleNamespace

    class _Patch(nn.Module):
        def forward(self, x):
            b, c, t, h, w = x.shape
            return x.permute(0, 2, 3, 4, 1)  # [B, T, H, W, C] before flatten

    class _TimeEmbed(nn.Module):
        def forward(self, hidden_states, timestep):
            b, seq, c = hidden_states.shape
            del timestep
            return hidden_states.new_zeros(b, seq, c), hidden_states.new_zeros(b, seq, c)

    class _Rope(nn.Module):
        def forward(self, hidden_states, fps=None):
            del fps
            return None

    tr = nn.Module()
    tr.config = SimpleNamespace(
        patch_size=(1, 1, 1),
        concat_padding_mask=False,
        extra_pos_embed_type=None,
        use_crossattn_projection=False,
        img_context_dim_in=None,
        controlnet_block_every_n=1,
    )
    tr.rope = _Rope()
    tr.patch_embed = _Patch()
    tr.time_embed = _TimeEmbed()
    tr.transformer_blocks = nn.ModuleList([_CountBlock(i) for i in range(num_layers)])
    tr.gradient_checkpointing = False
    return tr


def test_should_hook_early_exit_policy():
    tower = CosmosVideoTower.__new__(CosmosVideoTower)
    tower.hook_early_exit = True
    tower.transformer = MagicMock()
    tower.transformer.transformer_blocks = list(range(28))

    assert CosmosVideoTower._should_hook_early_exit(tower, compute_video_loss=False) is True
    assert CosmosVideoTower._should_hook_early_exit(tower, compute_video_loss=True) is False

    tower.hook_early_exit = False
    assert CosmosVideoTower._should_hook_early_exit(tower, compute_video_loss=False) is False


def test_forward_transformer_stops_at_extract_layer():
    tr = _minimal_mock_transformer(num_layers=6)
    for block in tr.transformer_blocks:
        nn.init.eye_(block.linear.weight)

    x = torch.randn(1, 4, 2, 2, 2)
    enc = torch.randn(1, 3, 4)
    out = forward_transformer_to_hook_layer(
        tr,
        extract_layer=2,
        hidden_states=x,
        timestep=torch.zeros(1),
        encoder_hidden_states=enc,
    )
    assert out.shape[-1] == 4
    for i, block in enumerate(tr.transformer_blocks):
        expected_calls = 1 if i <= 2 else 0
        assert block.call_count == expected_calls, f"block {i} called {block.call_count} times"

    for block in tr.transformer_blocks:
        block.call_count = 0

    full = forward_transformer_to_hook_layer(
        tr,
        extract_layer=5,
        hidden_states=x,
        timestep=torch.zeros(1),
        encoder_hidden_states=enc,
    )
    assert full.shape == out.shape
    assert all(block.call_count == 1 for block in tr.transformer_blocks)
