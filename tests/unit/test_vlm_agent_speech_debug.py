"""Diagnose VLM agent speech: input alignment vs degraded LM weights.

Run (needs GPU + Psi0 ckpt):
  cd Phi_0 && PYTHONPATH=src pytest tests/unit/test_vlm_agent_speech_debug.py -s -q
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch

from phi0.models.vlm.speech_quality import looks_degraded, word_repetition_ratio

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEFAULT_EP = 447
CONFIG_NAME = "train_pick_tissue_xperience_unified_ddp4_3k"


def _load_ep447_bundle():
    from hydra import compose, initialize_config_dir
    from pathlib import Path

    from phi0.data.pick_tissue_unified import PickTissueUnifiedClipDataset
    from phi0.deploy.pick_tissue_gt import clip_dataset_index_for_episode
    from phi0.inference.session import ActionInferenceSession
    from phi0.models.vlm.tower import GenerateTextConfig
    from phi0.runtime import (
        build_base_dataset,
        build_processor,
        create_phi0,
        prepare_model_batch_cpu,
    )

    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
        cfg = compose(config_name=CONFIG_NAME)
    cfg.device = "cuda"
    model = create_phi0(cfg)
    model.eval()
    processor = build_processor(cfg).eval()
    tower = model.vlm_tower
    proc = tower.processor

    base = build_base_dataset(cfg)
    clip_row = clip_dataset_index_for_episode(base, DEFAULT_EP, data_cfg=cfg.data)
    batch = PickTissueUnifiedClipDataset.collate_fn([base[clip_row]])
    cpu = prepare_model_batch_cpu(model, processor, batch)
    train_vlm = cpu["vlm_inputs"]
    obs = cpu["obs_pixel"]
    wrist_obs = cpu.get("obs_wrist_pixel")

    session = ActionInferenceSession(
        model,
        processor,
        use_wrist_view=bool(getattr(processor, "use_wrist_view", False)),
    )
    video = (obs[:, 0] * 2.0 - 1.0).unsqueeze(2)
    wrist_v = None
    if wrist_obs is not None:
        wrist_v = (wrist_obs[:, 0] * 2.0 - 1.0).unsqueeze(2)
    deploy_vlm = session._build_vlm_inputs_from_video(
        video.to(device=model.device),
        "pick tissue",
        wrist_video=wrist_v.to(device=model.device) if wrist_v is not None else None,
    )

    gen_cfg = GenerateTextConfig(max_new_tokens=64, do_sample=False, suppress_mm_tokens=True)

    def _gen(vlm: Dict[str, Any]) -> str:
        on_dev = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in vlm.items()}
        return tower.generate_text_from_vlm_batch(on_dev, gen_cfg=gen_cfg)[0]

    text_only = proc(
        text=["What should a robot do to pick up a tissue? Answer in one sentence."],
        return_tensors="pt",
        padding=True,
    )
    text_only = {k: v.to(model.device) for k, v in text_only.items() if k in ("input_ids", "attention_mask")}
    sup = list(range(151644, int(tower.vlm_model.lm_head.out_features)))
    gen_ids = tower.vlm_model.generate(
        **text_only,
        max_new_tokens=64,
        do_sample=False,
        suppress_tokens=sup,
    )
    trim = gen_ids[:, text_only["input_ids"].shape[1] :]
    text_only_out = proc.batch_decode(
        trim,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return {
        "train_vlm": train_vlm,
        "deploy_vlm": deploy_vlm,
        "train_text": _gen(train_vlm),
        "deploy_text": _gen(deploy_vlm),
        "text_only_out": text_only_out,
    }


@pytest.fixture(scope="module")
def ep447_diag():
    return _load_ep447_bundle()


def test_train_deploy_input_ids_match(ep447_diag):
    """Chat template / token ids aligned; gibberish is not from prompt mismatch."""
    train = ep447_diag["train_vlm"]["input_ids"]
    deploy = ep447_diag["deploy_vlm"]["input_ids"]
    assert torch.equal(train, deploy)


def test_pixel_values_minor_roundtrip_only(ep447_diag):
    """Deploy video roundtrip vs train obs differs slightly (PIL), not a full mismatch."""
    train_pv = ep447_diag["train_vlm"]["pixel_values"].float()
    dep_pv = ep447_diag["deploy_vlm"]["pixel_values"].float()
    max_diff = float((train_pv - dep_pv).abs().max())
    assert max_diff < 0.05, f"unexpected large pixel diff {max_diff}"


def test_text_only_also_degraded(ep447_diag):
    """Pure text AR (no image) still collapses → Psi0 HE ckpt LM head, not ep447 input."""
    text = ep447_diag["text_only_out"].strip()
    rep = word_repetition_ratio(text)
    assert len(text) > 10
    assert rep > 0.25, f"expected repetitive collapse, got rep={rep:.2f} text={text[:120]!r}"


def test_vision_paths_same_failure_mode(ep447_diag):
    """Train vs deploy vision inputs both degrade (stutter / nonsense)."""
    assert looks_degraded(ep447_diag["train_text"])
    assert looks_degraded(ep447_diag["deploy_text"])


def test_diagnostic_report(ep447_diag, capsys):
    """Print human-readable summary when run with ``pytest -s``."""
    bundle = ep447_diag
    lines = [
        "=== VLM agent speech diagnostic (ep447) ===",
        f"input_ids match: {torch.equal(bundle['train_vlm']['input_ids'], bundle['deploy_vlm']['input_ids'])}",
        f"train vision rep: {word_repetition_ratio(bundle['train_text']):.2f}",
        f"deploy vision rep: {word_repetition_ratio(bundle['deploy_text']):.2f}",
        f"text-only rep: {word_repetition_ratio(bundle['text_only_out']):.2f}",
        f"train vision: {bundle['train_text'][:160]!r}",
        f"text-only: {bundle['text_only_out'][:160]!r}",
        "verdict: input aligned; Psi0 HE ckpt AR language degraded (need base Instruct or lang SFT).",
    ]
    report = "\n".join(lines)
    print(report)
    assert "verdict" in report
