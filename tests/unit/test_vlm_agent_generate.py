"""VLM agent speech: eval-only, once per session, opt-in."""

from __future__ import annotations

import torch

from phi0.inference.session import ActionInferenceSession
from phi0.models.factory_smoke import create_phi0_action_only_smoke
from phi0.models.vlm.tower import GenerateTextConfig, decode_generated_text


class _StubTokenizer:
    def batch_decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return [f"tok_{int(row[0])}" if row.numel() else "" for row in token_ids]


def test_decode_generated_text_trims_prompt():
    processor = type("P", (), {"tokenizer": _StubTokenizer()})()
    input_ids = torch.tensor([[1, 2, 3]])
    generated_ids = torch.tensor([[1, 2, 3, 9, 10]])
    assert decode_generated_text(processor, input_ids, generated_ids) == ["tok_9"]


def test_agent_speech_default_off():
    model = create_phi0_action_only_smoke(device="cpu", text_dim=2048)
    session = ActionInferenceSession(model)
    video = torch.zeros(1, 3, 1, 64, 64)
    session.prefill_from_video_clip(video, "pick up the tissue")
    assert session.run_agent_speech_once() == ""


def test_agent_speech_once_then_noop_on_refresh():
    model = create_phi0_action_only_smoke(device="cpu", text_dim=2048)
    session = ActionInferenceSession(model)
    session.enable_agent_speech_for_eval(True)
    video = torch.zeros(1, 3, 1, 64, 64)
    session.prefill_from_video_clip(video, "pick up the tissue")
    first = session.run_agent_speech_once()
    assert first == "smoke_vlm_reply_0"
    assert session.run_agent_speech_once() == first
    session.refresh_video_context_from_clip(video + 1.0, prompt="other task")
    assert session.run_agent_speech_once() == first


def test_predict_unaffected_by_agent_flag():
    model = create_phi0_action_only_smoke(device="cpu", text_dim=2048)
    session = ActionInferenceSession(model)
    session.enable_agent_speech_for_eval(True)
    video = torch.zeros(1, 3, 1, 64, 64)
    session.prefill_from_video_clip(video, "pick up the tissue")
    session.seed_proprio_from_normalized(torch.zeros(model.action_expert.raw_action_dim))
    pred = session.predict(4)
    assert pred.shape[0] == 4
    session.run_agent_speech_once()
    pred2 = session.predict(4)
    assert pred2.shape[0] == 4


def test_generate_text_config_defaults():
    cfg = GenerateTextConfig()
    assert cfg.max_new_tokens == 256
    assert cfg.do_sample is False
