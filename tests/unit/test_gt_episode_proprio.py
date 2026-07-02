"""GT episode proprio for closed-loop deploy fallback."""

from __future__ import annotations

import numpy as np
import torch

from phi0.deploy.gt_io import GtEpisodeProprioSource


class _FakeBackend:
    def pack_deploy_frame(self, *, control_idx: int, state_control_idx: int):
        del state_control_idx
        row = np.zeros(512, dtype=np.float32)
        row[0] = float(control_idx)
        return row, np.zeros(3, dtype=np.float32)


class _FakeProcessor:
    mean = torch.zeros(512)
    std = torch.ones(512)

    def _normalize_action(self, t: torch.Tensor) -> torch.Tensor:
        return t


class _FakeModel:
    past_action_window_size = 1

    def uses_history_action_input(self) -> bool:
        return False


class _FakeSession:
    def __init__(self):
        self.model = _FakeModel()
        self.last = None

    def set_proprio_gt(self, proprio: torch.Tensor) -> None:
        self.last = proprio.detach().cpu()


def test_gt_episode_proprio_uses_control_idx():
    src = GtEpisodeProprioSource(backend=_FakeBackend())
    proc = _FakeProcessor()
    sess = _FakeSession()
    src.apply_to_session(sess, proc, 12)
    assert sess.last is not None
    assert int(sess.last.shape[0]) == 1
    assert float(sess.last[0, 0, 0]) == 12.0
