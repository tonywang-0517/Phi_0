"""Unit tests for OpenPI HfmInputs + SONIC unified policy I/O."""

from __future__ import annotations

import numpy as np

from openpi.models import model as _model
from openpi.policies.psi_policy import HfmInputs
from phi0.data.sonic_unified_io import SONIC_ACTION_DIM, SONIC_STATE_DIM


def _rgb(h: int = 64, w: int = 64, val: int = 128) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


def test_hfm_inputs_sonic_unified_dual_camera():
    transform = HfmInputs(model_type=_model.ModelType.PI05)
    states = np.arange(SONIC_STATE_DIM, dtype=np.float32)
    actions = np.ones((30, SONIC_ACTION_DIM), dtype=np.float32)
    out = transform(
        {
            "observation/image": _rgb(val=10),
            "observation/wrist_image": _rgb(val=20),
            "states": states,
            "actions": actions,
            "prompt": "pick tissue",
        }
    )
    assert out["state"].shape == (SONIC_STATE_DIM,)
    assert np.allclose(out["state"], states)
    assert out["actions"].shape == (30, SONIC_ACTION_DIM)
    assert out["image_mask"]["base_0_rgb"]
    assert out["image_mask"]["left_wrist_0_rgb"]
    assert not out["image_mask"]["right_wrist_0_rgb"]
    assert out["image"]["left_wrist_0_rgb"][0, 0, 0] == 20
    assert out["image"]["right_wrist_0_rgb"].shape == out["image"]["base_0_rgb"].shape


def test_hfm_inputs_mono_fallback_masks_left_wrist():
    transform = HfmInputs(model_type=_model.ModelType.PI05)
    out = transform(
        {
            "observation/image": _rgb(val=10),
            "states": np.zeros(SONIC_STATE_DIM, dtype=np.float32),
            "actions": np.zeros((4, SONIC_ACTION_DIM), dtype=np.float32),
            "prompt": "demo",
        }
    )
    assert not out["image_mask"]["left_wrist_0_rgb"]
    assert not out["image_mask"]["right_wrist_0_rgb"]
