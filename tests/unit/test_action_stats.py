"""Unit tests for action normalization statistics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from phi0.data.action_stats import compute_action_stats_from_datasets, stats_to_tensors
from phi0.data.xperience import XperienceDataset
from phi0.schema.draw_schema import D_RAW


class _TinyDataset:
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        x = np.zeros(D_RAW, dtype=np.float32)
        x[0] = float(idx + 1)
        x[3] = float(idx + 2)
        pad = np.zeros(D_RAW, dtype=bool)
        pad[100:] = True
        return {
            "action": torch.from_numpy(x).unsqueeze(0),
            "action_dim_is_pad": torch.from_numpy(pad),
        }


def test_compute_action_stats_masked_dims():
    stats = compute_action_stats_from_datasets([_TinyDataset()])
    assert stats["num_frames"] == 2
    assert stats["mean"][0] == pytest.approx(1.5, abs=1e-5)
    assert stats["mean"][3] == pytest.approx(2.5, abs=1e-5)
    assert stats["std"][0] == pytest.approx(0.5, abs=1e-5)
    assert stats["count_per_dim"][100] == 0
    assert stats["mean"][100] == pytest.approx(0.0, abs=1e-6)
    assert stats["std"][100] == pytest.approx(1.0, abs=1e-6)


def test_stats_to_tensors():
    stats = {"mean": [0.0] * D_RAW, "std": [2.0] * D_RAW}
    mean, std = stats_to_tensors(stats)
    assert mean.shape == (D_RAW,)
    assert std[0].item() == pytest.approx(2.0)


@pytest.mark.skipif(
    not __import__("pathlib").Path(
        "/mnt/data2/wpy/workspace/Isaac-GR00T/demo_data/xperience-10m-sample/annotation.hdf5"
    ).exists(),
    reason="Xperience demo HDF5 not available",
)
def test_xperience_stats_keypoints_have_variance():
    ds = XperienceDataset(max_frames=8, cache_video=False)
    stats = compute_action_stats_from_datasets([ds])
    mean = np.array(stats["mean"][:156])
    std = np.array(stats["std"][:156])
    assert std.max() > 0.01
    assert not np.allclose(mean, 0.0)
