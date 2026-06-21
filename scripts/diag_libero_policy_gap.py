#!/usr/bin/env python3
"""Diagnose train vs deploy gap for LIBERO delta policy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ckpt = ROOT / "experiments/libero_spatial_vlm_only_35k_ddp4/libero_spatial_vlm_only_35k_ddp4_latest.pt"
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="train_libero_spatial_vlm_only_35k_ddp4")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    from phi0.checkpoint_utils import merge_saved_cfg

    if payload.get("cfg"):
        cfg = merge_saved_cfg(cfg, payload["cfg"])

    from phi0.benchmark.policy import Phi0VLAPolicy
    from phi0.data.sequence import sequence_dataset_from_cfg
    from phi0.inference.session import ActionInferenceSession
    from phi0.runtime import (
        apply_processor_stats_from_checkpoint,
        build_base_dataset,
        build_processor,
        create_phi0,
        prepare_model_batch,
        sync_model_action_norm,
    )

    device = torch.device("cuda:0")
    model = create_phi0(cfg)
    model.load_checkpoint(str(ckpt))
    model.eval().to(device)
    processor = build_processor(cfg).eval()
    apply_processor_stats_from_checkpoint(processor, payload, cfg)
    sync_model_action_norm(model, processor)
    seq = sequence_dataset_from_cfg(build_base_dataset(cfg), cfg.data)

    train_errs, train_grip = [], []
    deploy_errs, deploy_grip = [], []
    with torch.no_grad():
        for i in range(200):
            sample = seq[i % len(seq)]
            batch = seq.collate_fn([sample])
            mb = prepare_model_batch(model, processor, batch)
            mb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in mb.items()}
            inputs = model.build_inputs(mb)
            action_ctx, action_ctx_mask = model._resolve_action_context(inputs=inputs)
            proprio, future, _, _ = model._action_proprio_future(
                inputs["action"], inputs.get("action_is_pad")
            )
            pred_train = model.predict_action(
                action_ctx,
                action_ctx_mask,
                future.shape[1],
                batch_size=1,
                proprio_tokens=proprio,
            )
            pred_train_d = processor.denormalize_robot7d_future(pred_train.float()).cpu().numpy()[0]
            gt_d = processor.denormalize_robot7d_future(future.float()).cpu().numpy()[0]
            train_errs.append(np.linalg.norm(pred_train_d[:, :6] - gt_d[:, :6], axis=1))
            train_grip.append(np.abs(pred_train_d[:, 6] - gt_d[:, 6]))

            # Deploy path: session prefill from last video frame + sim-style proprio from GT clip
            session = ActionInferenceSession(model, processor, deploy_seq_len=int(cfg.data.seq_len))
            images = batch["images"]["ego_view"].float()
            if images.ndim == 5:
                frame = images[0, -1]
            elif images.ndim == 4:
                frame = images[-1]
            else:
                frame = images
            clip = frame.unsqueeze(0).unsqueeze(2) * 2.0 - 1.0
            prompt = str(sample["task"]).strip().lower()
            session.prefill_from_video_clip(clip, prompt)
            proprio_gt = batch["robot_proprio_7d"].float().to(device)
            steps = [
                proprio_gt[0, j].reshape(-1).to(device=device, dtype=model.torch_dtype)
                for j in range(proprio_gt.shape[1])
            ]
            session._proprio_history.clear()
            for s in steps:
                session._proprio_history.append(s)
            session._proprio_hold = steps[-1]
            pred_deploy = session.predict(future.shape[1])
            pred_deploy_d = processor.denormalize_robot7d_future(pred_deploy.float()).cpu().numpy()
            deploy_errs.append(np.linalg.norm(pred_deploy_d[:, :6] - gt_d[:, :6], axis=1))
            deploy_grip.append(np.abs(pred_deploy_d[:, 6] - gt_d[:, 6]))

    train_errs = np.concatenate(train_errs)
    deploy_errs = np.concatenate(deploy_errs)
    print("=== Offline teacher-forcing (training batch path) ===")
    print(
        f"6D L2/step: mean={train_errs.mean():.4f} "
        f"p50={np.percentile(train_errs, 50):.4f} p90={np.percentile(train_errs, 90):.4f}"
    )
    print(f"gripper |err|: mean={np.mean(train_grip):.4f}")
    print("=== Deploy session path (same GT proprio/image) ===")
    print(
        f"6D L2/step: mean={deploy_errs.mean():.4f} "
        f"p50={np.percentile(deploy_errs, 50):.4f} p90={np.percentile(deploy_errs, 90):.4f}"
    )
    print(f"gripper |err|: mean={np.mean(deploy_grip):.4f}")
    print(f"train vs deploy mean gap: {(deploy_errs.mean()-train_errs.mean()):+.4f}")

    policy = Phi0VLAPolicy.from_paths(
        checkpoint=str(ckpt),
        config_dir=str(ROOT / "configs"),
        config_name="train_libero_spatial_vlm_only_35k_ddp4",
        device="cuda:0",
        min_free_gb=8,
        num_open_loop_steps=8,
        action_mode="robot7d",
    )
    print("=== Policy flags ===")
    print("libero_delta_eef", policy._libero_delta_eef)
    print("libero_proprio_absolute", policy._libero_proprio_absolute)
    print("open_loop", policy.default_open_loop)


if __name__ == "__main__":
    main()
