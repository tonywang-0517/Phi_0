"""Unit tests for SIMPLE G1 36-d bounds normalization."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from phi0.data.simple_action_norm import (
    SIMPLE_G1_DIM,
    denormalize_robot_nd,
    load_simple_stats_json,
    normalize_robot_nd,
)


def _fake_stats() -> dict:
    return {
        "version": 2,
        "robot_action_semantics": "simple_g1_wholebody_36d",
        "norm_mode": "bounds_q99",
        "normalize_gripper": True,
        "robot_dim": SIMPLE_G1_DIM,
        "action_dim": SIMPLE_G1_DIM,
        "mean": [0.0] * SIMPLE_G1_DIM,
        "std": [1.0] * SIMPLE_G1_DIM,
        "q01": [-1.0] * SIMPLE_G1_DIM,
        "q99": [1.0] * SIMPLE_G1_DIM,
        "state_mean": [0.0] * SIMPLE_G1_DIM,
        "state_std": [1.0] * SIMPLE_G1_DIM,
        "state_q01": [-0.5] * SIMPLE_G1_DIM,
        "state_q99": [0.5] * SIMPLE_G1_DIM,
    }


def test_bounds_roundtrip():
    stats = _fake_stats()
    x = torch.linspace(-0.8, 0.8, SIMPLE_G1_DIM)
    y = denormalize_robot_nd(normalize_robot_nd(x, stats), stats)
    assert torch.allclose(x, y, atol=1e-5)


def test_proprio_uses_state_stats():
    stats = _fake_stats()
    x = torch.zeros(SIMPLE_G1_DIM)
    norm = normalize_robot_nd(x, stats, proprio=True)
    assert norm.shape == (SIMPLE_G1_DIM,)
    assert torch.allclose(norm, torch.zeros(SIMPLE_G1_DIM))


def test_load_stats_json_file(tmp_path: Path):
    payload = {
        "action": {"q01": [0.0] * SIMPLE_G1_DIM, "q99": [2.0] * SIMPLE_G1_DIM},
        "states": {"q01": [-1.0] * SIMPLE_G1_DIM, "q99": [1.0] * SIMPLE_G1_DIM},
    }
    path = tmp_path / "stats_psi0.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_simple_stats_json(path)
    assert loaded["norm_mode"] == "bounds_q99"
    assert loaded["robot_action_semantics"] == "simple_g1_wholebody_36d"
