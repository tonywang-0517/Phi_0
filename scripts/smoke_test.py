#!/usr/bin/env python3
"""End-to-end smoke tests for Phi_0 (256-d D_raw, 52×3 keypoints pose)."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE = Path(os.environ.get("PHI0_WORKSPACE", "/home/user"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(_WORKSPACE / "FastWAM/src"))

from phi0.data.egodex import EgoDexDataset
from phi0.data.processor import Phi0Processor, build_overfit_datasets
from phi0.data.xperience import XperienceDataset, resolve_xperience_video_path
from phi0.schema.action_schema import get_action_schema
from phi0.schema.draw_schema import D_RAW, zero_unsupervised_action_dims


def test_mono_camera_resolution():
    path = resolve_xperience_video_path()
    if path is not None:
        assert path.name == "stereo_left.mp4", f"Expected mono left eye, got {path.name}"
    print("OK mono camera path")


def test_real_video_frames():
    xp = XperienceDataset(max_frames=2)
    eg = EgoDexDataset(max_frames=2)
    assert xp[0].get("uses_real_video", False), "Xperience should load stereo_left.mp4 frames"
    assert eg[0].get("uses_real_video", False), "EgoDex should load 0.mp4 frames"
    f0 = xp[0]["images"]["ego_view"][0]
    f1 = xp[1]["images"]["ego_view"][0]
    assert not torch.allclose(f0, f1), "Consecutive video frames should differ"
    print("OK real mono video frames")


def test_language_prompts():
    xp = XperienceDataset(max_frames=1)
    eg = EgoDexDataset(max_frames=1)
    assert xp[0]["task"] and len(str(xp[0]["task"])) > 3
    assert eg[0]["task"] and len(str(eg[0]["task"])) > 3
    assert xp[0]["task"] != eg[0]["task"], "Datasets should expose distinct task prompts"
    mixed = build_overfit_datasets(xperience_max_frames=2, egodex_max_frames=2)
    batch = mixed.collate_fn([mixed[0], mixed[len(xp)]])
    assert "task" in batch and len(batch["task"]) == 2
    print(f"OK language prompts: xperience='{xp[0]['task'][:60]}' egodex='{eg[0]['task']}'")


def test_data_loaders():
    pose_end = get_action_schema().pose_dim_end
    xp = XperienceDataset(max_frames=4)
    eg = EgoDexDataset(max_frames=4)
    assert len(xp) == 4
    assert len(eg) == 4
    x0 = xp[0]
    e0 = eg[0]
    assert x0["action"].shape[-1] == D_RAW
    assert e0["action"].shape[-1] == D_RAW
    assert x0["images"]["ego_view"].ndim == 4  # [T,C,H,W] mono
    assert (~x0["action_dim_is_pad"]).sum() == pose_end
    eg_avail = (~e0["action_dim_is_pad"]).sum().item()
    assert 0 < eg_avail < pose_end, f"EgoDex sparse keypoints GT expected, got {eg_avail}/{pose_end}"
    print(f"OK data loaders (xperience={pose_end}/{D_RAW}, egodex sparse={eg_avail}/{D_RAW})")


def test_vision_language_batch():
    from hydra import compose, initialize_config_dir
    from phi0.runtime import create_phi0, prepare_model_batch, build_dataloader

    config_dir = str(ROOT / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train_overfit")
    cfg.device = "cpu"
    model = create_phi0(cfg, smoke=True)
    processor = Phi0Processor().eval()
    batch = next(iter(build_dataloader(cfg)))
    mb = prepare_model_batch(model, processor, batch)
    assert mb["video"].shape[0] >= 1
    assert mb["context"].abs().sum() > 0, "Text context should be non-zero (smoke hash encoder)"
    assert mb["context"].shape[0] == mb["video"].shape[0]
    print("OK vision+language batch")


def test_training_step():
    from hydra import compose, initialize_config_dir
    from phi0.runtime import run_training

    config_dir = str(ROOT / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train_overfit")
    cfg.max_steps = 1
    cfg.batch_size = 1
    cfg.device = "cpu"
    cfg.data.seq_len = 5
    with tempfile.TemporaryDirectory() as td:
        cfg.output_dir = td
        run_training(cfg)
        assert (Path(td) / "phi0_smoke.pt").exists()
    print("OK training step")


def test_inference_smoke():
    from hydra import compose, initialize_config_dir
    from phi0.inference.session import ActionInferenceSession
    from phi0.runtime import create_phi0, prepare_model_batch, build_dataloader
    from phi0.data.processor import Phi0Processor

    config_dir = str(ROOT / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train_overfit")
    cfg.device = "cpu"
    model = create_phi0(cfg, smoke=True)
    model.eval()
    processor = Phi0Processor().eval()
    batch = next(iter(build_dataloader(cfg)))
    mb = prepare_model_batch(model, processor, batch)
    mb = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in mb.items()}
    prompt = batch["task"][0]
    session = ActionInferenceSession(model, processor=processor)
    with torch.no_grad():
        session.prefill_from_image(mb["video"][:, :, 0], prompt)
        pred = session.predict(int(mb["action"].shape[1]))
    assert pred.shape == (mb["action"].shape[1], D_RAW)
    print("OK inference smoke (FM chunk predict)")


def test_fm_predict_deterministic():
    from hydra import compose, initialize_config_dir
    from phi0.inference.session import ActionInferenceSession
    from phi0.runtime import create_phi0, prepare_model_batch, build_dataloader
    from phi0.data.processor import Phi0Processor

    config_dir = str(ROOT / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train_overfit")
    cfg.device = "cpu"
    model = create_phi0(cfg, smoke=True)
    model.eval()
    processor = Phi0Processor().eval()
    batch = next(iter(build_dataloader(cfg)))
    mb = prepare_model_batch(model, processor, batch)
    mb = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in mb.items()}
    num_frames = int(mb["action"].shape[1])

    ctx = torch.zeros(1, 16, model.text_dim, device=model.device, dtype=model.torch_dtype)
    ctx_mask = torch.ones(1, 16, device=model.device, dtype=torch.bool)
    with torch.no_grad():
        a = model.predict_action_fm(ctx, ctx_mask, num_frames)
        b = model.predict_action_fm(ctx, ctx_mask, num_frames)
    assert torch.allclose(a, b, atol=1e-5), "FM predict should be deterministic in eval mode"
    pose_end = get_action_schema().pose_dim_end
    assert float(a[..., pose_end:].abs().max()) == 0.0, "deploy tail must be zero"
    print("OK FM predict deterministic")


def main():
    test_mono_camera_resolution()
    test_real_video_frames()
    test_language_prompts()
    test_data_loaders()
    test_vision_language_batch()
    test_training_step()
    test_inference_smoke()
    test_fm_predict_deterministic()
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
