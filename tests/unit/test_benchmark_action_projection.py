from __future__ import annotations

import numpy as np
import torch

from phi0.benchmark.action_projection import KeypointToArmActionProjector


class _DummyProcessor:
    def postprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_project_chunk_output_shape():
    projector = KeypointToArmActionProjector()
    proc = _DummyProcessor()
    pred = torch.zeros(8, 256)
    out = projector.project_chunk(pred, proc)
    assert out.shape == (8, 7)


def test_project_chunk_gripper_is_binary():
    projector = KeypointToArmActionProjector()
    proc = _DummyProcessor()
    pred = torch.zeros(4, 256)
    # force large wrist-tip distance to trigger open
    # wrist idx=21 -> dims 63:66, tip idx=51 -> dims 153:156
    pred[:, 153] = 0.2
    out = projector.project_chunk(pred, proc)
    assert np.all(np.isin(out[:, 6], [0.0, 1.0]))

