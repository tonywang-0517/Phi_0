"""DiT4DiT-style action token encoder (action MLP + sinusoidal timestep embedding)."""

from __future__ import annotations

import torch
import torch.nn as nn


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal encoding of shape (B, T, dim) from timesteps (B, T)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        batch_size, seq_len = timesteps.shape
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0, device=device)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class Dit4DiTActionEncoder(nn.Module):
    """Encode action tokens with DiT4DiT ActionEncoder (MLP + sin time embed)."""

    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = actions.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == batch_size:
            timesteps = timesteps.unsqueeze(1).expand(-1, seq_len)
        elif timesteps.shape != (batch_size, seq_len):
            raise ValueError(
                f"Expected timesteps (B,) or (B,T); got {tuple(timesteps.shape)} "
                f"for actions (B,T,D)=({batch_size},{seq_len},*)"
            )

        action_emb = self.layer1(actions)
        time_emb = self.pos_encoding(timesteps).to(dtype=action_emb.dtype)
        x = torch.cat([action_emb, time_emb], dim=-1)
        x = swish(self.layer2(x))
        return self.layer3(x)
