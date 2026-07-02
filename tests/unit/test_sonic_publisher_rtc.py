"""Unit tests for SONIC publisher RTC blending logic.

Tests cover:
- _shift_chunk_rtc: chunk rolling helper
- _resolve_rtc_cfg: CLI / model-cfg merge
- _predict_motion_deploy_rtc: full RTC deploy path with mock session predict
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Stub heavy/unavailable runtime deps before importing the publisher script.
for _mod in (
    "zmq",
    "gear_sonic",
    "gear_sonic.utils",
    "gear_sonic.utils.teleop",
    "gear_sonic.utils.teleop.zmq",
    "gear_sonic.utils.teleop.zmq.v4_latent_replay",
    "gear_sonic.utils.teleop.zmq.zmq_planner_sender",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import phi0_sonic_latent_zmq_publisher as pub  # noqa: E402


# ---------------------------------------------------------------------------
# _shift_chunk_rtc
# ---------------------------------------------------------------------------

def test_shift_chunk_2d():
    chunk = torch.arange(12).float().view(4, 3)  # [H=4, D=3]
    shifted = pub._shift_chunk_rtc(chunk, s=2)
    assert shifted.shape == (4, 3)
    assert torch.allclose(shifted[:2], chunk[2:4])
    assert torch.allclose(shifted[2], chunk[3])
    assert torch.allclose(shifted[3], chunk[3])


def test_shift_chunk_3d():
    chunk = torch.arange(12).float().view(1, 4, 3)  # [1, H=4, D=3]
    shifted = pub._shift_chunk_rtc(chunk, s=1)
    assert shifted.shape == (1, 4, 3)
    assert torch.allclose(shifted[0, 0], chunk[0, 1])
    assert torch.allclose(shifted[0, -1], chunk[0, -1])


def test_shift_full_horizon_pads_all():
    chunk = torch.ones(6, 2) * 5.0
    chunk[0] = 0.0
    shifted = pub._shift_chunk_rtc(chunk, s=6)
    assert torch.allclose(shifted, torch.ones(6, 2) * 5.0)


# ---------------------------------------------------------------------------
# _resolve_rtc_cfg
# ---------------------------------------------------------------------------

def _make_args(**kwargs):
    args = MagicMock()
    args.rtc = kwargs.get("rtc", False)
    args.rtc_inference_delay = kwargs.get("rtc_inference_delay", 0)
    args.rtc_execution_horizon = kwargs.get("rtc_execution_horizon", 0)
    args.rtc_schedule = kwargs.get("rtc_schedule", "")
    return args


def _make_cfg(enabled=False, d=2, s=4, schedule="exponential"):
    rtc = MagicMock()
    rtc.enabled = enabled
    rtc.inference_delay = d
    rtc.execution_horizon = s
    rtc.schedule = schedule
    cfg = MagicMock()
    cfg.rtc = rtc
    return cfg


def test_resolve_rtc_cfg_cli_enables():
    cfg = _make_cfg(enabled=False)
    args = _make_args(rtc=True)
    result = pub._resolve_rtc_cfg(cfg, args)
    assert result["enabled"] is True


def test_resolve_rtc_cfg_model_cfg_wins_when_cli_off():
    cfg = _make_cfg(enabled=True, d=3, s=5, schedule="linear")
    args = _make_args(rtc=False)
    result = pub._resolve_rtc_cfg(cfg, args)
    assert result["enabled"] is True
    assert result["inference_delay"] == 3
    assert result["execution_horizon"] == 5
    assert result["schedule"] == "linear"


def test_resolve_rtc_cfg_cli_overrides_values():
    cfg = _make_cfg(enabled=False, d=2, s=4, schedule="exponential")
    args = _make_args(rtc=True, rtc_inference_delay=1, rtc_execution_horizon=3, rtc_schedule="hard")
    result = pub._resolve_rtc_cfg(cfg, args)
    assert result["inference_delay"] == 1
    assert result["execution_horizon"] == 3
    assert result["schedule"] == "hard"


def test_resolve_rtc_cfg_defaults_when_no_model_rtc():
    cfg = MagicMock()
    cfg.rtc = None
    args = _make_args(rtc=True)
    result = pub._resolve_rtc_cfg(cfg, args)
    assert result["enabled"] is True
    assert result["inference_delay"] == 2
    assert result["execution_horizon"] == 4
    assert result["schedule"] == "exponential"


# ---------------------------------------------------------------------------
# _predict_motion_deploy_rtc (mock model → deterministic tensors)
# ---------------------------------------------------------------------------

def _make_smoke_model(action_dim: int = 4, history_w: int = 1):
    model = MagicMock()
    model.past_action_window_size = history_w
    model.action_history_window = 0
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.uses_history_action_input = lambda: False
    model.uses_vlm_tower = lambda: False
    model.uses_dual_vggt_cross_attn = lambda: False
    model.uses_cross_attn_context = lambda: False
    model.text_dim = 8

    def _dummy_action_context(batch_size, device, dtype, text_dim):
        return (
            torch.zeros(batch_size, 4, text_dim, device=device, dtype=dtype),
            torch.ones(batch_size, 4, dtype=torch.bool, device=device),
        )

    def predict_action(ctx, ctx_mask, num_frames, **kwargs):
        return torch.ones(1, num_frames, action_dim)

    model._dummy_action_context = _dummy_action_context
    model.predict_action = predict_action
    return model


class _FakeLut(dict):
    def pin_device(self, device):
        pass


def _make_lut(n: int, action_dim: int) -> _FakeLut:
    lut = _FakeLut({i: torch.zeros(action_dim) for i in range(n)})
    return lut


def test_predict_motion_deploy_rtc_output_shape():
    """RTC deploy → correct (num_frames, action_dim) shape."""
    chunk_h, action_dim, history_w = 8, 4, 1
    num_frames, d, s = 12, 2, 4

    model = _make_smoke_model(action_dim=action_dim, history_w=history_w)
    processor = MagicMock()
    processor.postprocess = lambda x: x  # identity denorm
    inputs = {
        "action_ctx": torch.zeros(1, 4, 8),
        "action_ctx_mask": torch.ones(1, 4, dtype=torch.bool),
    }
    lut = _make_lut(history_w + 30, action_dim)

    with patch.object(pub, "resolve_deploy_action_chunk_size", return_value=chunk_h):
        result = pub._predict_motion_deploy_rtc(
            model, processor, inputs,
            num_frames=num_frames, proprio_w=1, gt_norm_lut=lut,
            inference_delay=d, execution_horizon=s,
        )

    assert result.shape == (num_frames, action_dim), f"expected ({num_frames},{action_dim}), got {result.shape}"


def test_predict_motion_deploy_rtc_single_query():
    """When execution_horizon == num_frames, only one query happens (no prev_chunk blending)."""
    chunk_h, action_dim, history_w = 8, 3, 1
    num_frames = 6  # < chunk_h, single query

    model = _make_smoke_model(action_dim=action_dim, history_w=history_w)
    processor = MagicMock()
    processor.postprocess = lambda x: x
    inputs = {
        "action_ctx": torch.zeros(1, 4, 8),
        "action_ctx_mask": torch.ones(1, 4, dtype=torch.bool),
    }
    lut = _make_lut(history_w + 20, action_dim)

    with patch.object(pub, "resolve_deploy_action_chunk_size", return_value=chunk_h):
        result = pub._predict_motion_deploy_rtc(
            model, processor, inputs,
            num_frames=num_frames, proprio_w=1, gt_norm_lut=lut,
            inference_delay=2, execution_horizon=num_frames,
        )

    assert result.shape == (num_frames, action_dim)


# ---------------------------------------------------------------------------
# RTC blend math (pure, no model dependency)
# ---------------------------------------------------------------------------

def test_rtc_blend_hard_schedule_freezes_first_d_steps():
    """prev=0, new=1 → first d steps frozen to 0, last s steps = 1."""
    from phi0.inference.rtc import blend_action_chunks_rtc, create_rtc_soft_mask

    H, D, d, s = 8, 3, 2, 4
    mask = create_rtc_soft_mask(H, d, s, schedule="hard")
    blended = blend_action_chunks_rtc(
        torch.ones(1, H, D), torch.zeros(1, H, D), mask
    )
    assert torch.allclose(blended[0, :d], torch.zeros(d, D)), "frozen prefix must stay prev"
    assert torch.allclose(blended[0, H - s:], torch.ones(s, D)), "free suffix must be new"
