from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from phi0.benchmark.action_projection import KeypointToArmActionProjector, ProjectionConfig
from phi0.benchmark.adapters import (
    calvin_eval_gripper_flip,
    calvin_obs_to_vla,
    libero_obs_to_eef_7d,
    libero_obs_to_vla,
    make_vla_prompt,
    process_vla_action,
)
from phi0.benchmark.bridge_head import bridge_logits_to_action, load_bridge_checkpoint
from phi0.benchmark.libero_deploy import (
    LiberoDeployFlags,
    normalize_libero_proprio_eef_7d,
    postprocess_libero_robot7d_chunk,
    resolve_libero_deploy_flags,
)
from phi0.benchmark.vla_types import VLAObservation
from phi0.checkpoint_utils import merge_saved_cfg
from phi0.inference.session import ActionInferenceSession, PromptEmbedCache, resolve_deploy_action_chunk_size
from phi0.models.vlm.preprocess import normalize_vlm_instruction
from phi0.runtime import (
    activate_cuda_device,
    apply_processor_stats_from_checkpoint,
    build_processor,
    create_phi0,
    resolve_inference_device,
    sync_model_action_norm,
)

# Backward-compatible re-exports for tests and scripts.
__all__ = [
    "Phi0VLAPolicy",
    "Phi0VLAPolicyConfig",
    "LiberoDeployFlags",
    "resolve_libero_deploy_flags",
]


@dataclass
class Phi0VLAPolicyConfig:
    checkpoint: str
    config_dir: str
    config_name: str = "train_full"
    device: str = "cuda"
    min_free_gb: float = 18.0
    image_size: int = 224
    center_crop: bool = True
    num_open_loop_steps: Optional[int] = None
    invert_openvla_gripper: bool = False
    libero_absolute_eef: Optional[bool] = None
    libero_proprio_absolute: Optional[bool] = None
    libero_delta_eef: Optional[bool] = None
    action_mode: str = "heuristic"  # heuristic | bridge | robot7d
    bridge_checkpoint: Optional[str] = None
    bridge_input_mode: str = "keypoints_chunk"  # keypoints_chunk | latent_norm
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)


class Phi0VLAPolicy:
    """Phi_0 policy wrapper exposing VLA-Adapter-like reset/step API."""

    def __init__(self, cfg: Phi0VLAPolicyConfig) -> None:
        self.cfg = cfg
        with initialize_config_dir(version_base="1.3", config_dir=cfg.config_dir):
            train_cfg = compose(config_name=cfg.config_name)
        yaml_data_keys = (
            "control_fps",
            "seq_len",
            "action_video_freq_ratio",
        )
        yaml_data_overrides = {
            k: train_cfg.data.get(k) for k in yaml_data_keys if train_cfg.data.get(k) is not None
        }
        device = resolve_inference_device(cfg.device, min_free_gb=float(cfg.min_free_gb))
        activate_cuda_device(device)
        train_cfg.device = device

        payload = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
        saved_cfg = payload.get("cfg") if isinstance(payload, dict) else None
        if saved_cfg:
            train_cfg = merge_saved_cfg(train_cfg, saved_cfg)
        for key, value in yaml_data_overrides.items():
            train_cfg.data[key] = value

        model = create_phi0(train_cfg, smoke=bool(train_cfg.get("smoke_action_only", False)))
        if isinstance(payload, dict) and ("model" in payload or "action_expert" in payload):
            model.load_checkpoint(cfg.checkpoint)
        model.eval()

        processor = build_processor(train_cfg).eval()
        if isinstance(payload, dict):
            apply_processor_stats_from_checkpoint(processor, payload, train_cfg)
        sync_model_action_norm(model, processor)

        libero_flags = resolve_libero_deploy_flags(cfg, train_cfg)

        self.model = model
        self.processor = processor
        self.prompt_cache = PromptEmbedCache()
        self.session = ActionInferenceSession(
            model=model,
            processor=processor,
            deploy_seq_len=int(train_cfg.data.get("seq_len", 33)),
            action_video_freq_ratio=int(train_cfg.data.get("action_video_freq_ratio", 2)),
            use_gt_proprio=libero_flags.proprio_absolute,
        )
        self.projector = KeypointToArmActionProjector(cfg.projection)
        self.bridge_input_mode = str(cfg.bridge_input_mode).strip().lower()
        self.bridge_head = None
        if cfg.bridge_checkpoint:
            bridge_model, bridge_payload = load_bridge_checkpoint(
                cfg.bridge_checkpoint,
                map_location=device,
            )
            bridge_model = bridge_model.to(device=self.model.device).eval()
            self.bridge_head = bridge_model
            ckpt_mode = str(bridge_payload.get("input_mode", self.bridge_input_mode)).strip().lower()
            if ckpt_mode:
                self.bridge_input_mode = ckpt_mode
        self.action_mode = str(cfg.action_mode).strip().lower()
        if self.action_mode == "bridge" and self.bridge_head is None:
            raise ValueError("action_mode=bridge requires bridge_checkpoint")
        self.default_open_loop = cfg.num_open_loop_steps or resolve_deploy_action_chunk_size(
            model, seq_len=int(train_cfg.data.get("seq_len", 33))
        )
        self._deploy_seq_len = int(train_cfg.data.get("seq_len", 9))
        self._action_video_freq_ratio = int(train_cfg.data.get("action_video_freq_ratio", 2))
        self._control_fps = float(train_cfg.data.get("control_fps", 20.0))
        self._past_window = int(getattr(model, "past_action_window_size", 1))
        self._frame_buffer: dict[int, torch.Tensor] = {}
        self._libero_flags = libero_flags
        self._libero_delta_eef = libero_flags.delta_eef
        self._libero_proprio_absolute = libero_flags.proprio_absolute
        self._libero_absolute_eef = libero_flags.absolute_eef

    @classmethod
    def from_paths(
        cls,
        *,
        checkpoint: str | Path,
        config_dir: str | Path,
        config_name: str = "train_full",
        device: str = "cuda",
        min_free_gb: float = 18.0,
        num_open_loop_steps: Optional[int] = None,
        action_mode: str = "heuristic",
        bridge_checkpoint: Optional[str] = None,
        bridge_input_mode: str = "keypoints_chunk",
    ) -> "Phi0VLAPolicy":
        return cls(
            Phi0VLAPolicyConfig(
                checkpoint=str(checkpoint),
                config_dir=str(config_dir),
                config_name=config_name,
                device=device,
                min_free_gb=float(min_free_gb),
                num_open_loop_steps=num_open_loop_steps,
                action_mode=action_mode,
                bridge_checkpoint=bridge_checkpoint,
                bridge_input_mode=bridge_input_mode,
            )
        )

    def reset(self) -> None:
        self.session.reset()
        self.projector.reset()
        self._frame_buffer.clear()

    def _ensure_proprio_seeded(self) -> None:
        w = int(getattr(self.model, "past_action_window_size", 0) or 0)
        if w <= 0:
            return
        if self.session._proprio_hold is not None or self.session._proprio_history:
            return
        if self._libero_flags.proprio_absolute:
            return
        mean = self.processor.mean.to(device=self.model.device, dtype=self.model.torch_dtype)
        self.session.seed_proprio_from_normalized(mean)

    def _eef_7d_to_normalized_proprio(self, eef_7d: np.ndarray) -> torch.Tensor:
        return normalize_libero_proprio_eef_7d(
            self.processor, self.model, eef_7d, self._libero_flags
        )

    def _append_proprio_from_obs(self, obs: dict, *, benchmark: str) -> None:
        if benchmark.lower() != "libero" or not self._libero_flags.proprio_absolute:
            return
        w = int(getattr(self.model, "past_action_window_size", 0) or 0)
        if w <= 0:
            return
        step_vec = self._eef_7d_to_normalized_proprio(libero_obs_to_eef_7d(obs))
        self.session._proprio_history.append(step_vec)
        self.session._proprio_hold = step_vec

    def _obs_to_vla(self, obs: dict, benchmark: str) -> VLAObservation:
        name = benchmark.lower()
        if name in {"calvin", "cavin"}:
            return calvin_obs_to_vla(
                obs, image_size=self.cfg.image_size, center_crop=self.cfg.center_crop
            )
        if name == "libero":
            return libero_obs_to_vla(
                obs, image_size=self.cfg.image_size, center_crop=self.cfg.center_crop
            )
        raise ValueError(f"Unsupported benchmark: {benchmark}")

    def _store_control_frame(self, obs: dict, *, benchmark: str, step: int) -> None:
        name = benchmark.lower()
        if name == "libero":
            from phi0.benchmark.adapters import libero_obs_to_native_frame

            frame = libero_obs_to_native_frame(obs)
        else:
            vla_obs = self._obs_to_vla(obs, benchmark=benchmark)
            frame = torch.from_numpy(vla_obs.full_image).permute(2, 0, 1).float() / 255.0
        self._frame_buffer[int(step)] = frame.detach()

    def _read_buffered_frame(self, control_t: int) -> torch.Tensor:
        if control_t in self._frame_buffer:
            return self._frame_buffer[control_t]
        if not self._frame_buffer:
            raise RuntimeError("Frame buffer is empty; call _store_control_frame first.")
        prior = [k for k in self._frame_buffer if k <= control_t]
        if prior:
            return self._frame_buffer[max(prior)]
        return self._frame_buffer[min(self._frame_buffer)]

    def _current_frame_bcthw(self, step: int) -> torch.Tensor:
        frame = self._read_buffered_frame(int(step))
        clip = frame.unsqueeze(0).unsqueeze(2)
        return clip.to(device=self.model.device, dtype=self.model.torch_dtype) * 2.0 - 1.0

    def observe(self, obs: dict, *, benchmark: str, step: int) -> None:
        """Record a sim frame on the control timeline (e.g. LIBERO physics-settle steps)."""
        self._store_control_frame(obs, benchmark=benchmark, step=int(step))
        self._append_proprio_from_obs(obs, benchmark=benchmark)

    def predict_phi0_chunk(
        self,
        obs: dict,
        instruction: str,
        *,
        benchmark: str,
        step: int = 0,
    ) -> torch.Tensor:
        """Run Phi_0 and return normalized prediction chunk [T, D]."""
        self._ensure_proprio_seeded()
        bench = benchmark.lower()
        self._store_control_frame(obs, benchmark=benchmark, step=step)

        if bench == "libero":
            prompt = normalize_vlm_instruction(instruction)
            clip = self._current_frame_bcthw(step)
            vggt_clip = clip if self.model.uses_dual_vggt_cross_attn() else None
            if self.session.action_ctx is None:
                self.session.prefill_from_video_clip(
                    clip, prompt, prompt_cache=self.prompt_cache, vggt_video=vggt_clip
                )
            else:
                self.session.refresh_video_context_from_clip(
                    clip, prompt=prompt, vggt_video=vggt_clip
                )
            return self.session.predict(self.default_open_loop)

        vla_obs = self._obs_to_vla(obs, benchmark=benchmark)
        prompt = make_vla_prompt(instruction)
        image_t = torch.from_numpy(vla_obs.full_image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device, dtype=self.model.torch_dtype
        )
        image_t = image_t * (2.0 / 255.0) - 1.0

        if self.session.action_ctx is None:
            self.session.prefill_from_image(image_t, prompt, prompt_cache=self.prompt_cache)
        else:
            self.session.refresh_video_context(image_t, prompt=prompt)
        return self.session.predict(self.default_open_loop)

    def build_bridge_features(
        self,
        pred_norm_chunk: torch.Tensor,
        *,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        """Build bridge input features from Phi_0 output chunk."""
        input_mode = str(mode or self.bridge_input_mode).strip().lower()
        if input_mode == "latent_norm":
            return pred_norm_chunk.detach().cpu().float().numpy().astype(np.float32)
        if input_mode != "keypoints_chunk":
            raise ValueError(f"Unsupported bridge input mode: {input_mode}")
        with torch.no_grad():
            d_raw = (
                self.processor.postprocess(pred_norm_chunk.unsqueeze(0))
                .squeeze(0)
                .detach()
                .cpu()
                .float()
                .numpy()
                .astype(np.float32)
            )
        return d_raw[:, :156]

    def _predict_bridge_chunk(self, pred_norm_chunk: torch.Tensor) -> np.ndarray:
        if self.bridge_head is None:
            raise RuntimeError("Bridge head is not loaded.")
        feats = self.build_bridge_features(pred_norm_chunk)
        x = torch.from_numpy(feats).to(device=self.model.device, dtype=torch.float32)
        with torch.no_grad():
            logits = self.bridge_head(x)
            chunk = bridge_logits_to_action(logits).cpu().numpy().astype(np.float32)
        return chunk

    def _denormalize_robot7d_chunk(self, pred: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            if self.model.uses_robot7d_action():
                return (
                    self.processor.denormalize_robot7d_future(pred.float())
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                    .astype(np.float32)
                )
            d_raw = (
                self.processor.postprocess(pred.unsqueeze(0))
                .squeeze(0)
                .detach()
                .float()
            )
            return d_raw.numpy().astype(np.float32)[:, :7]

    def step(self, obs: dict, instruction: str, step: int, *, benchmark: str) -> list[np.ndarray]:
        """
        VLA-compatible step API.

        Returns a list of open-loop 7D actions.
        """
        pred = self.predict_phi0_chunk(obs, instruction, benchmark=benchmark, step=step)
        bench = benchmark.lower()
        invert_gripper = bool(self.cfg.invert_openvla_gripper or bench == "libero")

        if self.action_mode == "bridge":
            chunk = self._predict_bridge_chunk(pred)
        elif self.action_mode == "robot7d":
            d7 = self._denormalize_robot7d_chunk(pred)
            if bench == "libero":
                chunk = postprocess_libero_robot7d_chunk(
                    d7, self._libero_flags, invert_openvla_gripper=invert_gripper
                )
            else:
                chunk = np.clip(d7, -1.0, 1.0).astype(np.float32)
                chunk[:, 6] = np.clip(chunk[:, 6], 0.0, 1.0)
        else:
            chunk = self.projector.project_chunk(pred, self.processor)

        if bench in {"calvin", "cavin"}:
            chunk = calvin_eval_gripper_flip(chunk)
            chunk = process_vla_action(chunk, invert_openvla_gripper=True)
        elif self.action_mode != "robot7d" or bench != "libero":
            chunk = process_vla_action(chunk, invert_openvla_gripper=invert_gripper)

        return [chunk[i] for i in range(chunk.shape[0])]
