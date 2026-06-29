"""Phi0 downstream executor: per-skill checkpoint routing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from phi0.agent.checkpoints import (
    DEFAULT_SKILL_CHECKPOINTS,
    SkillCheckpointSpec,
    resolve_skill_checkpoint,
    skill_checkpoint_overrides,
)
from phi0.models.vlm.preprocess import normalize_vlm_instruction

logger = logging.getLogger(__name__)

SKILL_TO_PHI0_INSTRUCTION: dict[str, str] = {
    "pick_tissues": "pick tissue",
    "throw_rubbish": "throw rubbish",
}

ACTION_SKILLS = frozenset(SKILL_TO_PHI0_INSTRUCTION)


@dataclass
class Phi0SkillResult:
    skill: str
    phi0_instruction: str
    status: str
    checkpoint: str = ""
    used_fallback_checkpoint: bool = False
    action_shape: tuple[int, ...] | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "phi0_instruction": self.phi0_instruction,
            "status": self.status,
            "checkpoint": self.checkpoint,
            "used_fallback_checkpoint": self.used_fallback_checkpoint,
            "action_shape": list(self.action_shape) if self.action_shape else None,
            "message": self.message,
        }


@dataclass
class _SkillRuntime:
    checkpoint: Path
    used_fallback: bool
    model: Any
    processor: Any
    session: Any


@dataclass
class Phi0SkillRouter:
    skill_checkpoints: dict[str, SkillCheckpointSpec] = field(
        default_factory=lambda: dict(DEFAULT_SKILL_CHECKPOINTS)
    )
    config_dir: str | None = None
    device: str = "cuda"
    min_free_gb: float = 12.0
    num_action_frames: int = 8
    phi0_root: Path | None = None
    _runtimes: dict[str, _SkillRuntime] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_overrides(
        cls,
        *,
        pick_checkpoint: str | None = None,
        throw_checkpoint: str | None = None,
        config_name: str | None = None,
        device: str = "cuda",
        min_free_gb: float = 12.0,
        num_action_frames: int = 8,
    ) -> "Phi0SkillRouter":
        return cls(
            skill_checkpoints=skill_checkpoint_overrides(
                pick_tissues=pick_checkpoint,
                throw_rubbish=throw_checkpoint,
                config_name=config_name,
            ),
            device=device,
            min_free_gb=min_free_gb,
            num_action_frames=num_action_frames,
        )

    def checkpoint_for_skill(self, skill: str) -> Path:
        key = str(skill).strip()
        if key not in self.skill_checkpoints:
            raise KeyError(f"no checkpoint registered for skill {key!r}")
        ckpt, _ = resolve_skill_checkpoint(self.skill_checkpoints[key], root=self.phi0_root)
        return ckpt

    def _load_skill(self, skill: str) -> _SkillRuntime:
        if skill in self._runtimes:
            return self._runtimes[skill]

        from hydra import compose, initialize_config_dir

        from phi0.checkpoint_utils import merge_saved_cfg
        from phi0.inference.session import ActionInferenceSession
        from phi0.runtime import (
            activate_cuda_device,
            apply_processor_stats_from_checkpoint,
            build_processor,
            create_phi0,
            resolve_inference_device,
            sync_model_action_norm,
        )

        spec = self.skill_checkpoints[skill]
        root = self.phi0_root or Path(__file__).resolve().parents[3]
        ckpt_path, used_fallback = resolve_skill_checkpoint(spec, root=root)
        cfg_dir = self.config_dir or str(root / "configs")
        device = resolve_inference_device(self.device, min_free_gb=float(self.min_free_gb))
        activate_cuda_device(device)

        with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
            cfg = compose(config_name=spec.config_name)
        cfg.device = device

        payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        if isinstance(payload, dict) and payload.get("cfg"):
            cfg = merge_saved_cfg(cfg, payload["cfg"])

        model = create_phi0(cfg)
        model.load_checkpoint(str(ckpt_path))
        model.eval()
        processor = build_processor(cfg).eval()
        if isinstance(payload, dict):
            apply_processor_stats_from_checkpoint(processor, payload, cfg)
        sync_model_action_norm(model, processor)

        session = ActionInferenceSession(
            model,
            processor,
            use_wrist_view=bool(getattr(processor, "use_wrist_view", False)),
        )
        runtime = _SkillRuntime(
            checkpoint=ckpt_path,
            used_fallback=used_fallback,
            model=model,
            processor=processor,
            session=session,
        )
        self._runtimes[skill] = runtime
        logger.info(
            "Phi0SkillRouter loaded skill=%s ckpt=%s fallback=%s",
            skill,
            ckpt_path,
            used_fallback,
        )
        return runtime

    def _pil_to_bcthw(self, image: Image.Image, processor, model) -> torch.Tensor:
        import numpy as np

        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        target_h, target_w = processor.vlm_image_size
        if (h, w) != (target_h, target_w):
            rgb = np.asarray(
                image.convert("RGB").resize((target_w, target_h), Image.BILINEAR),
                dtype=np.uint8,
            )
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = (tensor * 2.0 - 1.0).unsqueeze(0).unsqueeze(2)
        return tensor.to(device=model.device, dtype=model.torch_dtype)

    def run_skill(
        self,
        skill: str,
        ego_image: Image.Image,
        *,
        wrist_image: Image.Image | None = None,
    ) -> Phi0SkillResult:
        key = str(skill).strip()
        if key not in ACTION_SKILLS:
            return Phi0SkillResult(
                skill=key,
                phi0_instruction="",
                status="error",
                message=f"unknown action skill {key!r}",
            )

        instruction = normalize_vlm_instruction(SKILL_TO_PHI0_INSTRUCTION[key])
        runtime = self._load_skill(key)
        video = self._pil_to_bcthw(ego_image, runtime.processor, runtime.model)
        wrist_video = (
            self._pil_to_bcthw(wrist_image, runtime.processor, runtime.model)
            if wrist_image is not None
            else None
        )
        runtime.session.reset()
        runtime.session.prefill_from_video_clip(video, instruction, wrist_video=wrist_video)
        dim = int(getattr(runtime.model.action_expert, "raw_action_dim", 512))
        runtime.session.seed_proprio_from_normalized(
            torch.zeros(dim, device=runtime.model.device)
        )
        action = runtime.session.predict(int(self.num_action_frames))
        shape = tuple(int(x) for x in action.shape)
        return Phi0SkillResult(
            skill=key,
            phi0_instruction=instruction,
            status="ok",
            checkpoint=str(runtime.checkpoint),
            used_fallback_checkpoint=runtime.used_fallback,
            action_shape=shape,
            message=f"Phi0 predict ok skill={key!r} instruction={instruction!r} shape={shape}",
        )


Phi0Executor = Phi0SkillRouter
