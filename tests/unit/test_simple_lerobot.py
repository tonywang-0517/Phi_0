"""Unit tests for SimpleG1ClipDataset collate shapes."""

from __future__ import annotations

import torch

from phi0.data.simple_action_norm import SIMPLE_G1_DIM
from phi0.data.simple_lerobot import SimpleG1ClipDataset
from phi0.schema.draw_schema import D_RAW


def _fake_sample(future_steps: int = 30):
    seq_len = 1 + future_steps
    img = torch.rand(1, 3, 180, 320)
    mask = torch.zeros(D_RAW, dtype=torch.bool)
    mask[:SIMPLE_G1_DIM] = True
    return {
        "dataset": "simple_g1",
        "idx": 0,
        "task": "pick",
        "images": {"ego_view": img},
        "image_is_pad": torch.zeros(1, dtype=torch.bool),
        "action": torch.zeros(seq_len, D_RAW),
        "action_is_pad": torch.zeros(seq_len, dtype=torch.bool),
        "action_dim_is_pad": mask,
        "robot_proprio_36d": torch.randn(1, SIMPLE_G1_DIM),
        "robot_future_36d": torch.randn(future_steps, SIMPLE_G1_DIM),
        "control_fps": 20.0,
        "action_video_freq_ratio": 1,
        "video_control_indices": torch.tensor([0], dtype=torch.long),
    }


def test_collate_stacks_robot_tensors():
    batch = SimpleG1ClipDataset.collate_fn([_fake_sample(), _fake_sample()])
    assert batch["robot_proprio_36d"].shape == (2, 1, SIMPLE_G1_DIM)
    assert batch["robot_future_36d"].shape == (2, 30, SIMPLE_G1_DIM)
    assert batch["action_is_pad"].shape == (2, 31)
