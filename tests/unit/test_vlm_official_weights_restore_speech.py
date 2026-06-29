"""Prove Psi0 HE ckpt LM collapse vs official Qwen3-VL on same ep447 frames.

Requires GPU + network (first run downloads official weights):
  PHI0_OFFICIAL_VLM_MODEL=Qwen/Qwen3-VL-2B-Instruct \\
  pytest tests/unit/test_vlm_official_weights_restore_speech.py -s -q
Skip: PHI0_SKIP_OFFICIAL_VLM_TEST=1
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest
import torch

from phi0.models.vlm.speech_quality import looks_coherent, looks_degraded
from phi0.models.vlm.tower import OFFICIAL_QWEN3VL_INSTRUCT, GenerateTextConfig, load_agent_speech_tower

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEFAULT_EP = 447
CONFIG_NAME = "train_pick_tissue_xperience_unified_ddp4_3k"
SKIP = os.environ.get("PHI0_SKIP_OFFICIAL_VLM_TEST", "").strip().lower() in {"1", "true", "yes"}
OFFICIAL_PATH = os.environ.get("PHI0_OFFICIAL_VLM_MODEL", OFFICIAL_QWEN3VL_INSTRUCT).strip()


def _ep447_pils_and_gen_cfg():
    from hydra import compose, initialize_config_dir
    from pathlib import Path

    from phi0.data.pick_tissue_unified import PickTissueUnifiedClipDataset
    from phi0.deploy.pick_tissue_gt import clip_dataset_index_for_episode
    from phi0.models.vlm.preprocess import build_qwenvl_inputs_single, tensor_frame_to_pil
    from phi0.runtime import build_base_dataset, build_processor, create_phi0, prepare_model_batch_cpu

    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
        cfg = compose(config_name=CONFIG_NAME)
    cfg.device = "cuda"
    model = create_phi0(cfg)
    model.eval()
    processor = build_processor(cfg).eval()
    psi0_tower = model.vlm_tower

    base = build_base_dataset(cfg)
    clip_row = clip_dataset_index_for_episode(base, DEFAULT_EP, data_cfg=cfg.data)
    batch = PickTissueUnifiedClipDataset.collate_fn([base[clip_row]])
    cpu = prepare_model_batch_cpu(model, processor, batch)
    obs = cpu["obs_pixel"]
    wrist_obs = cpu.get("obs_wrist_pixel")

    pils: List[Any] = [tensor_frame_to_pil(obs[0, 0])]
    if wrist_obs is not None:
        pils.append(tensor_frame_to_pil(wrist_obs[0, 0]))

    gen_cfg = GenerateTextConfig(max_new_tokens=96, do_sample=False, suppress_mm_tokens=True)
    instruction = "pick tissue"

    def _gen(tower, proc) -> str:
        vlm = build_qwenvl_inputs_single(proc, pils, instruction)
        on_dev = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in vlm.items()}
        return tower.generate_text_from_vlm_batch(on_dev, gen_cfg=gen_cfg)[0]

    psi0_text = _gen(psi0_tower, psi0_tower.processor)
    video = (obs[:, 0] * 2.0 - 1.0).unsqueeze(2)
    wrist_v = None
    if wrist_obs is not None:
        wrist_v = (wrist_obs[:, 0] * 2.0 - 1.0).unsqueeze(2)
    return {
        "model": model,
        "processor": processor,
        "pils": pils,
        "instruction": instruction,
        "gen_cfg": gen_cfg,
        "psi0_tower": psi0_tower,
        "psi0_text": psi0_text,
        "video": video,
        "wrist_video": wrist_v,
    }


@pytest.fixture(scope="module")
def ep447_speech_bundle():
    if SKIP:
        pytest.skip("PHI0_SKIP_OFFICIAL_VLM_TEST set")
    return _ep447_pils_and_gen_cfg()


@pytest.fixture(scope="module")
def official_tower(ep447_speech_bundle):
    if SKIP:
        pytest.skip("PHI0_SKIP_OFFICIAL_VLM_TEST set")
    model = ep447_speech_bundle["model"]
    try:
        tower = load_agent_speech_tower(
            OFFICIAL_PATH,
            device=str(model.device),
            torch_dtype=model.torch_dtype,
            attn_implementation=str(getattr(model.vlm_tower, "attn_implementation", "sdpa")),
            local_files_only=False,
        )
    except OSError as exc:
        pytest.skip(f"official VLM unavailable at {OFFICIAL_PATH!r}: {exc}")
    yield tower
    del tower
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_psi0_weights_degraded_on_same_ep447_frames(ep447_speech_bundle):
    text = ep447_speech_bundle["psi0_text"]
    assert looks_degraded(text), f"expected Psi0 collapse, got: {text[:200]!r}"


def test_official_weights_restore_speech(ep447_speech_bundle, official_tower):
    from phi0.models.vlm.preprocess import build_qwenvl_inputs_single

    bundle = ep447_speech_bundle
    vlm: Dict[str, Any] = build_qwenvl_inputs_single(
        official_tower.processor,
        bundle["pils"],
        bundle["instruction"],
    )
    on_dev = {
        k: v.to(bundle["model"].device) if torch.is_tensor(v) else v
        for k, v in vlm.items()
    }
    official_text = official_tower.generate_text_from_vlm_batch(
        on_dev,
        gen_cfg=bundle["gen_cfg"],
    )[0]
    print("\n--- Psi0 (action ckpt) ---")
    print(bundle["psi0_text"])
    print("\n--- Official Instruct ---")
    print(official_text)
    assert looks_degraded(bundle["psi0_text"])
    assert looks_coherent(official_text), f"official still bad: {official_text[:240]!r}"


def test_session_agent_speech_model_path_switch(ep447_speech_bundle, official_tower):
    """Session: action tower stays Psi0; agent AR uses official when configured."""
    from phi0.inference.session import ActionInferenceSession

    bundle = ep447_speech_bundle
    model = bundle["model"]
    processor = bundle["processor"]
    session = ActionInferenceSession(
        model,
        processor,
        use_wrist_view=bool(getattr(processor, "use_wrist_view", False)),
        agent_speech_model_path=OFFICIAL_PATH,
    )
    session._agent_speech_tower = official_tower  # ponytail: reuse fixture, skip 2nd load
    session.enable_agent_speech_for_eval(True)
    video = bundle["video"].to(model.device)
    wrist_v = bundle["wrist_video"]
    if wrist_v is not None:
        wrist_v = wrist_v.to(model.device)
    session.prefill_from_video_clip(video, bundle["instruction"], wrist_video=wrist_v)
    assert session.model.vlm_tower is bundle["psi0_tower"]
    agent_text = session.run_agent_speech_once(gen_cfg=bundle["gen_cfg"])
    print("\n--- session official agent ---")
    print(agent_text)
    assert looks_coherent(agent_text), f"session official path bad: {agent_text[:240]!r}"
