"""Unit tests for deploy ActionInferenceSession video refresh + chunk predict."""

from __future__ import annotations

import pytest
import torch

from phi0.inference.session import ActionInferenceSession, PromptEmbedCache, resolve_deploy_action_chunk_size
from phi0.models.factory_smoke import create_phi0_action_only_smoke
from phi0.schema.draw_schema import D_RAW


def test_resolve_deploy_action_chunk_size():
    model = create_phi0_action_only_smoke(device="cpu", past_action_window_size=4)
    assert resolve_deploy_action_chunk_size(model, seq_len=33) == 29


def test_proprio_cold_start_replicates_current_frame_not_zeros():
    model = create_phi0_action_only_smoke(
        device="cpu", torch_dtype=torch.float32, past_action_window_size=4, action_head="act"
    )
    model.eval()
    session = ActionInferenceSession(model)
    anchor = torch.arange(D_RAW, dtype=torch.float32) + 1.0
    session.seed_proprio_from_normalized(anchor)
    proprio = session._proprio_tensor()
    assert proprio is not None
    assert proprio.shape == (1, 4, D_RAW)
    assert not torch.allclose(proprio, torch.zeros_like(proprio))
    assert torch.allclose(proprio, anchor.view(1, 1, -1).expand(1, 4, -1))


def test_proprio_requires_seed_before_predict():
    model = create_phi0_action_only_smoke(
        device="cpu", torch_dtype=torch.float32, past_action_window_size=4, action_head="act"
    )
    model.eval()
    session = ActionInferenceSession(model)
    img0 = torch.rand(1, 3, 480, 640) * 2.0 - 1.0
    session.prefill_from_image(img0, "pick up cup")
    with pytest.raises(RuntimeError, match="set_proprio_gt"):
        session.predict(3)


def test_refresh_video_updates_context():
    model = create_phi0_action_only_smoke(device="cpu", torch_dtype=torch.float32)
    model.eval()
    session = ActionInferenceSession(model, deploy_seq_len=33, action_video_freq_ratio=2)
    prompt_cache = PromptEmbedCache()
    img0 = torch.rand(1, 3, 480, 640) * 2.0 - 1.0
    clip0 = img0.unsqueeze(2).expand(1, 3, 17, 480, 640)
    session.prefill_from_video_clip(clip0, "pick up cup", prompt_cache=prompt_cache)
    ctx0 = session.context_emb.clone()

    img1 = torch.rand(1, 3, 480, 640) * 2.0 - 1.0
    clip1 = img1.unsqueeze(2).expand(1, 3, 17, 480, 640)
    session.refresh_video_context_from_clip(clip1)
    assert session.video_refresh_count == 2
    assert session._video_clip is not None
    assert session._video_clip.shape[2] == 17
    assert not torch.allclose(ctx0, session.context_emb)


def test_set_proprio_gt_overrides_history():
    model = create_phi0_action_only_smoke(
        device="cpu", torch_dtype=torch.float32, past_action_window_size=4, action_head="act"
    )
    model.eval()
    session = ActionInferenceSession(model, use_gt_proprio=True)
    gt = torch.stack([torch.full((256,), float(i)) for i in range(4)], dim=0)
    session.set_proprio_gt(gt)
    proprio = session._proprio_tensor()
    assert proprio is not None
    assert torch.allclose(proprio[0, 0, 0], torch.tensor(0.0))
    assert torch.allclose(proprio[0, 3, 0], torch.tensor(3.0))


def test_predict_shape():
    model = create_phi0_action_only_smoke(device="cpu", torch_dtype=torch.float32)
    model.eval()
    session = ActionInferenceSession(model)
    img0 = torch.rand(1, 3, 480, 640) * 2.0 - 1.0
    session.prefill_from_image(img0, "pick up cup")
    pred = session.predict(7)
    assert pred.shape == (7, model.action_expert.raw_action_dim)
