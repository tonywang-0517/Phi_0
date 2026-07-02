"""Training-time RTC postfix mask (Psi0-style random prefix delay)."""

from __future__ import annotations

import torch

from phi0.models.phi0 import Phi0


class _StubExpert:
    raw_action_dim = 512
    text_dim = 512
    action_cross_attn_mode = "interleave_vlm"


def _stub_phi0(*, rtc_enabled: bool) -> Phi0:
    model = Phi0.__new__(Phi0)
    torch.nn.Module.__init__(model)
    model.action_expert = _StubExpert()
    model.action_head = "act"
    model.action_fm = None
    model.rtc_enabled = rtc_enabled
    model.rtc_max_delay = 4
    return model


def test_rtc_postfix_mask_disabled():
    model = _stub_phi0(rtc_enabled=False)
    assert model._rtc_postfix_mask(2, 8, torch.device("cpu")) is None


def test_rtc_postfix_mask_zeros_prefix():
    torch.manual_seed(0)
    model = _stub_phi0(rtc_enabled=True)
    mask = model._rtc_postfix_mask(16, 10, torch.device("cpu"))
    assert mask is not None
    assert mask.shape == (16, 10, 1)
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    # With max_delay=4, delay in [0,3); some rows should have frozen prefix steps.
    assert (mask[:, :3, 0].sum(dim=1) < 3).any()


def test_compute_action_loss_respects_rtc_mask():
    model = _stub_phi0(rtc_enabled=True)
    pred = torch.ones(2, 4, 3)
    target = torch.zeros(2, 4, 3)
    rtc_mask = torch.tensor([[[0.0], [1.0], [1.0], [1.0]], [[1.0], [1.0], [1.0], [1.0]]])
    loss = model._compute_action_loss(pred, target, None, None, rtc_postfix_mask=rtc_mask)
    # Row0: only steps 1..3 contribute (3 steps * 3 dims)
    expected = (3 * 3) / (3 * 3)
    assert torch.allclose(loss, torch.tensor(expected))


def test_resolve_rtc_deploy_cfg_reads_model_section():
    from types import SimpleNamespace

    from phi0.inference.rtc import resolve_rtc_deploy_cfg

    cfg = SimpleNamespace(
        model=SimpleNamespace(
            rtc=SimpleNamespace(
                enabled=True,
                inference_delay=2,
                execution_horizon=4,
                schedule="exponential",
            )
        )
    )
    out = resolve_rtc_deploy_cfg(cfg)
    assert out["enabled"] is True
    assert out["inference_delay"] == 2
    assert out["execution_horizon"] == 4
    assert out["schedule"] == "exponential"
