"""LIBERO delta-EEF chunk + absolute-EEF proprio prefix alignment."""

from __future__ import annotations

import numpy as np
import torch

from phi0.benchmark.rlds_adapters import (
    libero_rlds_action_to_train,
    libero_rlds_state_to_eef_7d,
)
from phi0.benchmark.rlds_io import RldsStep
from phi0.data.sequence import SequenceDataset
from phi0.models.action_proprio import split_proprio_future


class _FakeLiberoFrameDataset:
    DATASET_NAME = "libero_spatial"

    def __init__(self, frames: list[dict]) -> None:
        self._frames = frames

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> dict:
        return self._frames[idx]


def _frame(state: np.ndarray, action: np.ndarray) -> dict:
    return {
        "dataset": "libero_spatial",
        "idx": 0,
        "task": "task",
        "robot_proprio_7d": torch.from_numpy(libero_rlds_state_to_eef_7d(state)).view(1, 7),
        "robot_delta_7d": torch.from_numpy(libero_rlds_action_to_train(action)).view(1, 7),
        "action": torch.zeros(1, 256),
        "action_dim_is_pad": torch.zeros(256, dtype=torch.bool),
        "images": {"ego_view": torch.zeros(1, 3, 8, 8)},
    }


def test_sequence_split_proprio_delta_chunk():
    states = [np.array([i, 0, 1, 0, 0, 0, 0.04, -0.04], dtype=np.float32) for i in range(13)]
    actions = [np.array([0.01, 0, 0, 0, 0, 0, 1.0 - (i % 2)], dtype=np.float32) for i in range(13)]
    ds = _FakeLiberoFrameDataset([_frame(s, a) for s, a in zip(states, actions)])
    seq = SequenceDataset(
        ds,
        seq_len=13,
        stride=1,
        future_action_steps=8,
        native_fps={"libero_spatial": 20.0},
    )
    item = seq.sample_at_start(0)
    assert item["robot_proprio_7d"].shape == (5, 7)
    assert item["robot_future_delta_7d"].shape == (8, 7)
    assert torch.allclose(item["robot_proprio_7d"][-1, 0], torch.tensor(4.0))
    expected_first = torch.from_numpy(libero_rlds_action_to_train(actions[4])).float()
    assert torch.allclose(item["robot_future_delta_7d"][0], expected_first)
    expected_last = torch.from_numpy(libero_rlds_action_to_train(actions[11])).float()
    assert torch.allclose(item["robot_future_delta_7d"][-1], expected_last)


def test_proprio_future_split_on_7d_tokens():
    proprio = torch.randn(1, 5, 7)
    future = torch.randn(1, 8, 7)
    merged = torch.cat([proprio, future], dim=1)
    p, f = split_proprio_future(merged, 5)
    assert p.shape == (1, 5, 7)
    assert f.shape == (1, 8, 7)


def test_rlds_action_is_delta_command():
    step = RldsStep(
        rgb_static=np.zeros((8, 8, 3), dtype=np.uint8),
        rgb_gripper=np.zeros((8, 8, 3), dtype=np.uint8),
        state=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.04, -0.04], dtype=np.float32),
        action=np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        language="task",
    )
    delta = libero_rlds_action_to_train(step.action)
    assert np.isclose(delta[0], 0.01)
    assert delta[6] == 1.0


def test_sequence_past_w1_proprio_at_clip_start():
    """past_w=1: proprio + first future delta anchor at control index 0 (VLM-aligned)."""
    from phi0.data.temporal_align import observation_subsampled_frame_index, video_sample_control_indices

    states = [np.array([i, 0, 1, 0, 0, 0, 0.04, -0.04], dtype=np.float32) for i in range(9)]
    actions = [np.array([0.01, 0, 0, 0, 0, 0, 1.0 - (i % 2)], dtype=np.float32) for i in range(9)]
    ds = _FakeLiberoFrameDataset([_frame(s, a) for s, a in zip(states, actions)])
    seq = SequenceDataset(
        ds,
        seq_len=9,
        stride=1,
        future_action_steps=8,
        native_fps={"libero_spatial": 20.0},
    )
    item = seq.sample_at_start(0)
    assert item["robot_proprio_7d"].shape == (1, 7)
    assert item["robot_future_delta_7d"].shape == (8, 7)
    assert torch.allclose(item["robot_proprio_7d"][0, 0], torch.tensor(0.0))
    expected_first = torch.from_numpy(libero_rlds_action_to_train(actions[0])).float()
    assert torch.allclose(item["robot_future_delta_7d"][0], expected_first)
    subsampled = video_sample_control_indices(9, seq.action_video_freq_ratio)
    assert observation_subsampled_frame_index(1, subsampled) == 0
    assert subsampled[0] == 0
