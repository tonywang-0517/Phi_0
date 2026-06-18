"""Conditional rectified flow: history prior -> future GT (no Gaussian source)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Beta


@dataclass
class ActionFMConfig:
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    num_inference_timesteps: int = 4


class ActionFlowMatching:
    """Rectified flow: x_t = (1-t)*x0 + t*source, target velocity v = source - x0."""

    def __init__(self, cfg: ActionFMConfig | None = None):
        self.cfg = cfg or ActionFMConfig()
        self._beta = Beta(self.cfg.noise_beta_alpha, self.cfg.noise_beta_beta)

    def sample_training_t(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Continuous t in (0, 1] per batch item (DiT4DiT Beta / noise_s)."""
        sample = self._beta.sample([batch_size]).to(device=device, dtype=dtype)
        return sample / float(self.cfg.noise_s)

    @staticmethod
    def corrupt(clean: torch.Tensor, source: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.view(-1, *([1] * (clean.ndim - 1)))
        return (1.0 - t) * clean + t * source

    @staticmethod
    def training_target(clean: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return source - clean

    def discretize_t(self, t_cont: torch.Tensor) -> torch.Tensor:
        """Map continuous t -> integer bucket ids for timestep embedding."""
        if t_cont.ndim > 1:
            t_cont = t_cont.reshape(t_cont.shape[0])
        return (t_cont * float(self.cfg.num_timestep_buckets)).long().clamp(
            0, self.cfg.num_timestep_buckets - 1
        )

    def denoise_euler(
        self,
        predict_velocity,
        *,
        initial_state: torch.Tensor | None = None,
        batch_size: int = 1,
        seq_len: int = 1,
        action_dim: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Euler integration from t=1 source to clean actions at t=0.

        When ``initial_state`` is omitted, starts from Gaussian noise (DiT4DiT-style).
        When provided (e.g. history hold-last prior), starts from that tensor.
        """
        if initial_state is None:
            if device is None:
                raise ValueError("denoise_euler requires device when initial_state is None")
            actions = torch.randn(
                int(batch_size),
                int(seq_len),
                int(action_dim),
                device=device,
                dtype=dtype,
            )
        else:
            actions = initial_state.clone()
        batch_size = int(actions.shape[0])
        num_steps = max(1, int(self.cfg.num_inference_timesteps))
        dt = 1.0 / float(num_steps)
        device = actions.device
        dtype = actions.dtype
        for step in range(num_steps):
            t_cont = 1.0 - step / float(num_steps)
            t_batch = torch.full((batch_size,), t_cont, device=device, dtype=dtype)
            t_disc = self.discretize_t(t_batch)
            pred_v = predict_velocity(actions, t_disc)
            actions = actions - dt * pred_v
        return actions
