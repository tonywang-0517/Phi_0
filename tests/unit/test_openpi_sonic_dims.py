"""Unit tests for openpi state/action padding with SONIC unified dims."""

from __future__ import annotations

import numpy as np

from openpi.transforms import PadStatesAndActions


def test_pad_states_and_actions_sonic_unified():
    t = PadStatesAndActions(model_action_dim=100, model_state_dim=43)
    data = {
        "state": np.ones(43, dtype=np.float32),
        "actions": np.ones((30, 100), dtype=np.float32),
    }
    out = t(data)
    assert out["state"].shape == (43,)
    assert out["actions"].shape == (30, 100)
    assert np.allclose(out["state"], 1.0)
    assert np.allclose(out["actions"], 1.0)


def test_pad_short_state_and_action():
    t = PadStatesAndActions(model_action_dim=100, model_state_dim=43)
    data = {
        "state": np.array([1.0, 2.0], dtype=np.float32),
        "actions": np.array([[3.0]], dtype=np.float32),
    }
    out = t(data)
    assert out["state"].shape == (43,)
    assert out["actions"].shape == (1, 100)
    assert out["state"][0] == 1.0
    assert out["state"][1] == 2.0
    assert np.all(out["state"][2:] == 0.0)
    assert out["actions"][0, 0] == 3.0
    assert np.all(out["actions"][0, 1:] == 0.0)
