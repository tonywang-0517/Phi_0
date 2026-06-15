"""Deterministic prompt embeddings for CPU smoke tests (no Qwen/T5 download)."""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn


class SmokeTextEncoder(nn.Module):
    """Sentinel module marking smoke text path as active (no learned weights)."""

    def __init__(self, embed_dim: int = 1024):
        super().__init__()
        self.embed_dim = int(embed_dim)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use SmokeVideoTower.encode_prompt for smoke tests.")


def hash_prompt_embeddings(
    prompts: Sequence[str],
    embed_dim: int,
    seq_len: int = 8,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map each prompt to a fixed pseudo-embedding (differs across strings)."""
    rows = []
    for text in prompts:
        seed = abs(hash(str(text))) % (2**31 - 1)
        rng = np.random.RandomState(seed)
        rows.append(rng.randn(seq_len, embed_dim).astype(np.float32))
    embeds = torch.from_numpy(np.stack(rows, axis=0)).to(device=device, dtype=dtype)
    mask = torch.ones((len(prompts), seq_len), device=embeds.device, dtype=torch.bool)
    return embeds, mask


def encode_smoke_prompt(
    prompt: Union[str, Sequence[str]],
    embed_dim: int,
    max_sequence_length: int = 128,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_list = [prompt] if isinstance(prompt, str) else list(prompt)
    seq_len = min(8, max(1, int(max_sequence_length // 16)))
    return hash_prompt_embeddings(prompt_list, embed_dim, seq_len=seq_len, device=device, dtype=dtype)
