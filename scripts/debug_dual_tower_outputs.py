#!/usr/bin/env python3
"""Dump Cosmos predict-video + VGGT register context on one LIBERO step for sanity checks."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = ROOT / "third_party" / "LIBERO"
if str(LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_ROOT))
sys.path.insert(0, str(ROOT / "src"))

# LIBERO init_states pickles (PyTorch 2.6+).
_torch_load = torch.load


def _torch_load_libero(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_libero  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug Cosmos + VGGT tower outputs on LIBERO")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config-dir", type=str, default=str(ROOT / "configs"))
    p.add_argument("--config-name", type=str, default="train_libero_spatial_act_300")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--min-free-gb", type=float, default=18.0)
    p.add_argument("--libero-suite", type=str, default="libero_spatial")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--libero-env-img-res", type=int, default=256)
    p.add_argument("--libero-steps-wait", type=int, default=10)
    p.add_argument("--control-step", type=int, default=10, help="Step index after wait to dump towers")
    p.add_argument("--output-dir", type=str, default="experiments/tower_debug")
    p.add_argument("--cosmos-video-steps", type=int, default=36, help="Diffusion steps for predict_video")
    p.add_argument("--cosmos-frames-out", type=int, default=16, help="Generated pixel frames")
    p.add_argument("--skip-cosmos-generate", action="store_true", help="Only dump hook/VGGT (faster)")
    p.add_argument(
        "--use-yaml-data-cfg",
        action="store_true",
        help="Ignore checkpoint data.cosmos_video_size (use current yaml, e.g. 704x1280)",
    )
    return p.parse_args()


def _clip_to_mp4(video_bcthw: torch.Tensor, path: Path, fps: int = 10) -> None:
    """Save ``[1,3,T,H,W]`` in [-1,1] or [0,1] as mp4."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x = video_bcthw.detach().float().cpu()
    if x.min() < -0.01:
        x = (x.clamp(-1, 1) + 1.0) * 0.5
    x = (x.clamp(0, 1) * 255.0).byte()
    frames = x[0].permute(1, 2, 3, 0).numpy()
    writer = imageio.get_writer(str(path), fps=fps)
    for t in range(frames.shape[0]):
        writer.append_data(frames[t])
    writer.close()


def _tensor_stats(name: str, t: torch.Tensor) -> dict:
    x = t.detach().float()
    return {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from phi0.benchmark.policy import Phi0VLAPolicy
    from phi0.benchmark.tower_viz import (
        save_cosmos_cond_vs_pred_frame,
        save_register_temporal_curve,
        save_vggt_input_frame_grid,
        save_vggt_register_heatmap,
        save_vggt_register_pca_rgb,
    )
    from phi0.inference.deploy_align import deploy_past_subsampled_video_control_indices
    from phi0.inference.session import _cosmos_video_input
    from phi0.models.vggt.tower import VGGT_NUM_REGISTERS

    policy = Phi0VLAPolicy.from_paths(
        checkpoint=args.checkpoint,
        config_dir=args.config_dir,
        config_name=args.config_name,
        device=args.device,
        min_free_gb=float(args.min_free_gb),
        action_mode="robot7d",
    )
    if args.use_yaml_data_cfg:
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(version_base="1.3", config_dir=args.config_dir):
            yaml_cfg = compose(config_name=args.config_name)
        from phi0.data.cosmos_video_size import cosmos_video_size_from_cfg
        from phi0.runtime import build_processor, sync_model_action_norm

        hw = cosmos_video_size_from_cfg(yaml_cfg.data)
        policy._cosmos_hw = (int(hw[0]), int(hw[1]))
        policy.processor = build_processor(yaml_cfg).eval()
        sync_model_action_norm(policy.model, policy.processor)
        policy.session.processor = policy.processor
        logger.info("Forced data cfg from yaml: cosmos_video_size=%s", policy._cosmos_hw)

    model = policy.model
    suite = benchmark.get_benchmark_dict()[args.libero_suite]()
    task = suite.get_task(int(args.task_id))
    task_desc = task.language
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=int(args.libero_env_img_res),
        camera_widths=int(args.libero_env_img_res),
    )
    env.seed(7)
    policy.reset()
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(int(args.task_id))[0])
    policy.observe(obs, benchmark="libero", step=0)

    steps = 0
    target = int(args.control_step)
    while steps < target + 1:
        if steps < int(args.libero_steps_wait):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            steps += 1
            policy.observe(obs, benchmark="libero", step=steps)
            continue
        if steps < target:
            # Roll forward with zero motion to reach target control step.
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            steps += 1
            policy.observe(obs, benchmark="libero", step=steps)
            continue
        break

    clip = policy._training_aligned_video_clip(steps)
    _clip_to_mp4(clip, out_dir / "vggt_input_clip.mp4")

    ctrl_indices = deploy_past_subsampled_video_control_indices(
        steps,
        control_fps=policy._control_fps,
        video_history_seconds=policy._video_history_seconds,
        action_video_freq_ratio=policy._action_video_freq_ratio,
    )
    unique_ctrl = sorted(set(ctrl_indices))

    save_vggt_input_frame_grid(
        clip,
        out_dir / "vggt_input_frames.png",
        control_indices=ctrl_indices,
        max_control_t=steps,
        image_resolution=int(getattr(model.vggt_tower, "image_resolution", 512)),
    )

    from phi0.benchmark.cosmos_prompts import resolve_libero_cosmos_prompt

    prompt_style = str(getattr(policy.processor, "cosmos_prompt_style", "raw")).lower()
    prompt = resolve_libero_cosmos_prompt(task_desc, prompt_style)
    pred_norm = policy.predict_phi0_chunk(obs, task_desc, benchmark="libero", step=steps)

    embeds, _ = policy.prompt_cache.get(model, prompt)
    from phi0.models.cosmos.video_input import (
        assert_tower_video_aligned,
        cosmos_hook_video_bcthw,
        num_latent_cond_frames_from_tower,
        v2w_cond_pixel_frame_count,
        vggt_tower_video_bcthw,
    )

    n_lcf = num_latent_cond_frames_from_tower(model.video_tower)
    cosmos_in = cosmos_hook_video_bcthw(clip, num_latent_conditional_frames=n_lcf)
    vggt_in = vggt_tower_video_bcthw(
        clip,
        num_latent_conditional_frames=n_lcf,
        vggt_use_full_video=bool(getattr(model, "vggt_use_full_video", True)),
    )
    assert_tower_video_aligned(cosmos_in, vggt_in, context="tower_debug")

    video_cosmos = _cosmos_video_input(clip, model=model)
    tower = model.video_tower
    _, action_ctx, action_ctx_mask = tower.forward_joint_step(
        video_cosmos, embeds, compute_video_loss=False
    )
    vggt_ctx, vggt_ctx_mask = model._resolve_vggt_context(clip, inputs={"vggt_video": clip})

    save_vggt_register_heatmap(
        vggt_ctx,
        out_dir / "vggt_register_heatmap.png",
        num_registers_per_frame=VGGT_NUM_REGISTERS,
    )
    save_vggt_register_pca_rgb(
        vggt_ctx,
        out_dir / "vggt_register_pca_rgb.png",
        num_registers_per_frame=VGGT_NUM_REGISTERS,
    )
    save_register_temporal_curve(
        vggt_ctx,
        out_dir / "vggt_register_temporal_curve.png",
        num_registers_per_frame=VGGT_NUM_REGISTERS,
    )

    ckpt_hw = None
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("cfg"), dict):
        ckpt_hw = (payload.get("cfg") or {}).get("data", {}).get("cosmos_video_size")

    report: dict = {
        "task": task_desc,
        "control_step": steps,
        "cosmos_video_size_yaml": list(policy._cosmos_hw),
        "cosmos_video_size_checkpoint": list(ckpt_hw) if ckpt_hw else None,
        "resolution_train_infer_match": (
            list(ckpt_hw) == list(policy._cosmos_hw) if ckpt_hw else None
        ),
        "video_control_indices": ctrl_indices,
        "unique_observed_control_frames": unique_ctrl,
        "video_history_seconds": policy._video_history_seconds,
        "control_fps": policy._control_fps,
        "v2w_cond_pixel_frames": v2w_cond_pixel_frame_count(
            int(getattr(model.video_tower.video_fm, "num_latent_conditional_frames", 2))
        ),
        "cosmos_vggt_input_frames": int(clip.shape[2]),
        "deploy_seq_len": policy._deploy_seq_len,
        "action_video_freq_ratio": policy._action_video_freq_ratio,
        "vggt_image_resolution": int(getattr(model.vggt_tower, "image_resolution", 512)),
        "action_output_dim_vla_adapter": 7,
        "action_chunk_open_loop": int(policy.default_open_loop),
        "vggt_visualizations": [
            "vggt_input_frames.png",
            "vggt_register_temporal_curve.png",
            "vggt_register_heatmap.png",
            "vggt_register_pca_rgb.png",
        ],
        "tensors": [
            _tensor_stats("input_clip", clip),
            _tensor_stats("cosmos_hook_video", video_cosmos),
            _tensor_stats("action_ctx", action_ctx),
            _tensor_stats("vggt_ctx", vggt_ctx),
        ],
    }

    with torch.no_grad():
        if model.uses_robot7d_action():
            pred_7d = (
                policy.processor.denormalize_robot7d_future(pred_norm.unsqueeze(0))
                .squeeze(0)
                .cpu()
                .numpy()
            )
        else:
            d_raw = policy.processor.postprocess(pred_norm.unsqueeze(0)).squeeze(0).float()
            pred_7d = d_raw.numpy()[:, :7]
    report["predicted_action_7d_first_step"] = pred_7d[0].tolist()
    report["predicted_action_chunk_shape"] = list(pred_7d.shape)
    report["action_decode"] = "denormalize_robot7d" if model.uses_robot7d_action() else "slice_first_7"

    if not args.skip_cosmos_generate:
        logger.info("Running Cosmos predict_video (%d steps, official I2W input)...", int(args.cosmos_video_steps))
        cond_chw = clip[0, :, -1].detach().float().cpu()
        from PIL import Image

        pil_cond = Image.fromarray(
            (
                ((cond_chw.clamp(-1, 1) + 1.0) * 0.5)
                .permute(1, 2, 0)
                .numpy()
                * 255.0
            )
            .round()
            .astype(np.uint8)
        )
        # Prefer raw LIBERO frame + official pipeline preprocess (cosmos-predict2.5 diffusers).
        img_raw = np.asarray(obs["agentview_image"], dtype=np.uint8)[::-1, ::-1].copy()
        pil_raw = Image.fromarray(img_raw)
        pil_raw.save(out_dir / "cosmos_cond_libero_raw256.png")

        gen = torch.Generator(device=model.device).manual_seed(42)
        cosmos_h, cosmos_w = policy._cosmos_hw
        pred_video, _ = model.predict_video(
            clip,
            embeds,
            prompt=prompt,
            num_inference_steps=int(args.cosmos_video_steps),
            num_pixel_frames_out=int(args.cosmos_frames_out),
            height=int(cosmos_h),
            width=int(cosmos_w),
            generator=gen,
            prefer_video2world=True,
        )
        _clip_to_mp4(pred_video.permute(0, 2, 1, 3, 4), out_dir / "cosmos_predict_video.mp4")
        cond_mse = save_cosmos_cond_vs_pred_frame(
            cond_chw,
            pred_video[0].detach().float().cpu(),
            out_dir / "cosmos_cond_vs_pred0.png",
        )
        report["cosmos_i2w_cond_frame_mse"] = cond_mse
        report["cosmos_action_hook_uses"] = "last_clip_frame (same as I2W cond)"
        report["tensors"].append(_tensor_stats("cosmos_predict_video", pred_video))

    (out_dir / "tower_debug_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    env.close()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote debug artifacts to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
