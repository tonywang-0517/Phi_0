"""Phi_0 world-action model: Cosmos-Predict2.5 video tower + DiT4DiT-style FM action head."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from phi0.models.action_act_dit import ActionACTDiT
from phi0.models.action_cross_attn import resolve_action_cross_attn_mode
from phi0.models.action_fm_dit import ActionFMDiT
from phi0.models.action_fm_scheduler import ActionFMConfig, ActionFlowMatching
from phi0.models.cosmos.video_tower import CosmosVideoTower, SmokeVideoTower
from phi0.models.cosmos.video_fm import VideoFMConfig
from phi0.schema.draw_schema import D_RAW
from phi0.losses.bone import (
    bone_direction_loss,
    bone_length_loss,
    hand_bone_length_loss,
    hand_keypoints_mse_loss,
)
from phi0.checkpoint_utils import load_action_expert_state_dict, load_model_state_dict
from phi0.data.video_pad import apply_video_pad_replacement
from phi0.models.action_history import (
    history_to_flow_source,
    split_history_future,
    split_history_future_dim_pad,
    split_history_future_pad,
)
from phi0.models.action_proprio import (
    split_proprio_future,
    split_proprio_future_dim_pad,
    split_proprio_future_pad,
)

import logging

logger = logging.getLogger(__name__)

ActionExpert = ActionFMDiT | ActionACTDiT


def build_action_expert(
    action_head: str,
    action_dit_config: dict[str, Any],
    *,
    raw_action_dim: int,
    device: str,
    torch_dtype: torch.dtype,
) -> ActionExpert:
    head = str(action_head).strip().lower()
    if head == "fm":
        return ActionFMDiT.from_action_dit_config(
            action_dit_config=action_dit_config,
            raw_action_dim=raw_action_dim,
            device=device,
            torch_dtype=torch_dtype,
        )
    if head == "act":
        return ActionACTDiT.from_action_dit_config(
            action_dit_config=action_dit_config,
            raw_action_dim=raw_action_dim,
            device=device,
            torch_dtype=torch_dtype,
        )
    raise ValueError(f"Unknown action_head={action_head!r}; expected 'fm' or 'act'.")


class Phi0(torch.nn.Module):
    """Cosmos pretrained VAE+DiT for video; ActionFMDiT with DiT4DiT hook cross-attn fusion."""

    def __init__(
        self,
        video_tower: Union[CosmosVideoTower, SmokeVideoTower],
        action_expert: ActionExpert,
        vggt_tower: Optional[nn.Module] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        loss_lambda_bone: float = 0.0,
        loss_lambda_bone_hand: float = 0.0,
        loss_lambda_bone_dir: float = 0.0,
        loss_lambda_hand_mse: float = 0.0,
        prompt_max_length: int = 512,
        action_head: str = "fm",
        action_fm_config: ActionFMConfig | None = None,
        past_action_window_size: int = 4,
        action_history_window: int | None = None,
        action_future_horizon: int | None = None,
        vggt_use_full_video: bool = True,
        infer_generate_video: bool = False,
    ):
        super().__init__()
        self.video_tower = video_tower
        self.action_expert = action_expert
        self.vggt_tower = vggt_tower
        self.action_cross_attn_mode = resolve_action_cross_attn_mode(
            getattr(action_expert, "action_cross_attn_mode", None),
            interleave_self_attention=getattr(action_expert, "interleave_self_attention", True),
        )
        self.action_head = str(action_head).strip().lower()
        if self.action_head not in {"fm", "act"}:
            raise ValueError(f"Unknown action_head={action_head!r}; expected 'fm' or 'act'.")
        self.text_dim = int(action_expert.text_dim)
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.loss_lambda_bone = float(loss_lambda_bone)
        self.loss_lambda_bone_hand = float(loss_lambda_bone_hand)
        self.loss_lambda_bone_dir = float(loss_lambda_bone_dir)
        self.loss_lambda_hand_mse = float(loss_lambda_hand_mse)
        self.prompt_max_length = int(prompt_max_length)
        self.past_action_window_size = int(past_action_window_size)
        self.action_history_window = (
            int(action_history_window) if action_history_window is not None else None
        )
        self.action_future_horizon = (
            int(action_future_horizon)
            if action_future_horizon is not None
            else (
                int(self.action_history_window)
                if self.action_history_window is not None
                else None
            )
        )
        self.vggt_use_full_video = bool(vggt_use_full_video)
        self.infer_generate_video = bool(infer_generate_video)
        self.repeated_action_steps = 1
        self.action_fm = (
            ActionFlowMatching(action_fm_config or ActionFMConfig())
            if self.action_head == "fm"
            else None
        )
        self.register_buffer("action_norm_mean", torch.zeros(D_RAW), persistent=False)
        self.register_buffer("action_norm_std", torch.ones(D_RAW), persistent=False)
        self.to(self.device)

    def uses_history_action_input(self) -> bool:
        """Ablation path: symmetric history window + dit4dit_prefix_query encoder."""
        encoder = getattr(self.action_expert, "action_token_encoder", "linear")
        return str(encoder).strip().lower() == "dit4dit_prefix_query"

    def set_action_norm_stats(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        self.action_norm_mean.copy_(mean.to(dtype=torch.float32).view(-1))
        self.action_norm_std.copy_(std.to(dtype=torch.float32).view(-1).clamp(min=1e-6))

    @classmethod
    def from_cosmos_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        base_model: str | None = None,
        revision: str = "diffusers/base/post-trained",
        checkpoints_dir: str | None = None,
        load_text_encoder: bool = True,
        load_transformer: bool = True,
        local_files_only: bool = True,
        action_dit_config: dict[str, Any] | None = None,
        action_head: str = "fm",
        action_fm_config: dict[str, Any] | None = None,
        extract_layer: int = 17,
        num_context_tokens: int = 64,
        raw_action_dim: int = D_RAW,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        loss_lambda_bone: float = 0.0,
        loss_lambda_bone_hand: float = 0.0,
        loss_lambda_bone_dir: float = 0.0,
        loss_lambda_hand_mse: float = 0.0,
        freeze_text_encoder: bool = True,
        freeze_vae: bool = True,
        freeze_transformer: bool = False,
        freeze_video_tower: bool | None = None,
        detach_action_context: bool = True,
        action_context_mode: str = "first_frame",
        capture_stochastic: bool = False,
        vae_sample: bool = False,
        conditional_frame_timestep: float = 0.0001,
        enable_cosmos_gradient_checkpointing: bool = False,
        cosmos_hook_early_exit: bool = True,
        prompt_max_length: int = 512,
        past_action_window_size: int = 4,
        action_history_window: int | None = None,
        action_future_horizon: int | None = None,
        vggt_use_full_video: bool = True,
        vggt_tower: Optional[nn.Module] = None,
        infer_generate_video: bool = False,
        video_fm_config: dict[str, Any] | None = None,
    ):
        del num_context_tokens
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required.")
        video_fm = VideoFMConfig(**dict(video_fm_config or {}))
        video_tower = CosmosVideoTower.from_pretrained(
            device=device,
            torch_dtype=torch_dtype,
            extract_layer=int(extract_layer),
            base_model=base_model,
            revision=revision,
            checkpoints_dir=checkpoints_dir,
            load_text_encoder=load_text_encoder,
            load_transformer=load_transformer,
            local_files_only=local_files_only,
            freeze_text_encoder=freeze_text_encoder,
            freeze_vae=freeze_vae,
            freeze_transformer=freeze_transformer,
            freeze=freeze_video_tower,
            detach_action_context=detach_action_context,
            action_context_mode=action_context_mode,
            capture_stochastic=capture_stochastic,
            vae_sample=vae_sample,
            conditional_frame_timestep=conditional_frame_timestep,
            enable_gradient_checkpointing=enable_cosmos_gradient_checkpointing,
            hook_early_exit=bool(cosmos_hook_early_exit),
            video_fm_config=video_fm,
        )
        action_cfg = dict(action_dit_config)
        action_cfg["text_dim"] = int(video_tower.action_context_dim)
        resolved_past_window = int(past_action_window_size)
        encoder = str(action_cfg.get("action_token_encoder", "linear")).strip().lower()
        if encoder == "dit4dit_prefix_query":
            resolved_history = (
                int(action_history_window)
                if action_history_window is not None
                else resolved_past_window
            )
            resolved_future_horizon = (
                int(action_future_horizon)
                if action_future_horizon is not None
                else resolved_history
            )
            action_cfg.setdefault("action_future_horizon", resolved_future_horizon)
        else:
            action_cfg["proprio_window"] = resolved_past_window
            resolved_history = None
            resolved_future_horizon = None
        if video_tower.transformer is not None:
            tcfg = video_tower.transformer.config
            action_cfg.setdefault("num_layers", 16)
            action_cfg.setdefault("num_heads", int(getattr(tcfg, "num_attention_heads", 16)))
            action_cfg.setdefault("attn_head_dim", int(getattr(tcfg, "attention_head_dim", 128)))

        action_expert = build_action_expert(
            action_head,
            action_cfg,
            raw_action_dim=raw_action_dim,
            device=device,
            torch_dtype=torch_dtype,
        )
        fm_cfg = ActionFMConfig(**dict(action_fm_config or {})) if action_head == "fm" else None
        return cls(
            video_tower=video_tower,
            action_expert=action_expert,
            vggt_tower=vggt_tower,
            device=device,
            torch_dtype=torch_dtype,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            loss_lambda_bone=loss_lambda_bone,
            loss_lambda_bone_hand=loss_lambda_bone_hand,
            loss_lambda_bone_dir=loss_lambda_bone_dir,
            loss_lambda_hand_mse=loss_lambda_hand_mse,
            prompt_max_length=prompt_max_length,
            action_head=action_head,
            action_fm_config=fm_cfg,
            past_action_window_size=resolved_past_window,
            action_history_window=resolved_history,
            action_future_horizon=resolved_future_horizon,
            vggt_use_full_video=bool(vggt_use_full_video),
            infer_generate_video=bool(infer_generate_video),
        )

    @staticmethod
    def _repeat_batch_dim0(tensor: Optional[torch.Tensor], repeats: int) -> Optional[torch.Tensor]:
        if tensor is None or repeats <= 1:
            return tensor
        return tensor.repeat(repeats, *([1] * (tensor.ndim - 1)))

    def _repeat_action_batch(
        self,
        repeats: int,
        action_ctx: torch.Tensor,
        action_ctx_mask: torch.Tensor,
        action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        action_dim_is_pad: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if repeats <= 1:
            return action_ctx, action_ctx_mask, action, action_is_pad, action_dim_is_pad
        return (
            self._repeat_batch_dim0(action_ctx, repeats),
            self._repeat_batch_dim0(action_ctx_mask, repeats),
            self._repeat_batch_dim0(action, repeats),
            self._repeat_batch_dim0(action_is_pad, repeats),
            self._repeat_batch_dim0(action_dim_is_pad, repeats),
        )

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.video_tower.text_encoder is None:
            raise ValueError("Prompt encoding requires Cosmos text encoder.")
        return self.video_tower.encode_prompt(prompt, max_sequence_length=self.prompt_max_length)

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor: torch.Tensor) -> torch.Tensor:
        return self.video_tower.encode_video(video_tensor)

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor) -> torch.Tensor:
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        return self.video_tower.encode_frame(input_image)

    def build_inputs(self, sample, tiled: bool = False):
        del tiled
        video = sample["video"]
        context = sample["context"]
        context_mask = sample["context_mask"]
        if video.ndim != 5:
            raise ValueError(f"`video` must be [B,3,T,H,W], got {tuple(video.shape)}")
        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"Video H,W must be multiples of 16, got {height}x{width}")
        if num_frames < 1:
            raise ValueError(f"Video T must be >= 1, got {num_frames}")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`action` must be [B,T,D], got {tuple(action.shape)}")

        cached_latents = sample.get("input_latents")
        video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        video = apply_video_pad_replacement(video, sample.get("image_is_pad"))

        if cached_latents is not None:
            input_latents = cached_latents.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        elif self.loss_lambda_video > 0:
            input_latents = self._encode_video_latents(video)
        elif self.loss_lambda_action > 0:
            # Action-only: Cosmos hook encodes via forward_joint_step; avoid full-clip VAE here.
            b, _, t, h, w = video.shape
            lh, lw = h // self.video_tower.vae_scale_factor_spatial, w // self.video_tower.vae_scale_factor_spatial
            lt = max(1, (t - 1) // self.video_tower.vae_scale_factor_temporal + 1)
            input_latents = torch.zeros(
                (b, self.video_tower.latent_channels, lt, lh, lw),
                device=self.device,
                dtype=self.torch_dtype,
            )
        else:
            b, _, t, h, w = video.shape
            lh, lw = h // self.video_tower.vae_scale_factor_spatial, w // self.video_tower.vae_scale_factor_spatial
            lt = max(1, (t - 1) // self.video_tower.vae_scale_factor_temporal + 1)
            input_latents = torch.randn(
                (b, self.video_tower.latent_channels, lt, lh, lw),
                device=self.device,
                dtype=self.torch_dtype,
            )

        context = context.to(device=self.device, dtype=self.torch_dtype)
        context_mask = context_mask.to(device=self.device)
        action = action.to(device=self.device, dtype=self.torch_dtype)

        vggt_video = sample.get("vggt_video")
        if vggt_video is not None:
            vggt_video = vggt_video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            vggt_video = apply_video_pad_replacement(vggt_video, sample.get("image_is_pad"))

        return {
            "input_latents": input_latents,
            "video": video,
            "vggt_video": vggt_video,
            "context": context,
            "context_mask": context_mask,
            "action": action,
            "action_is_pad": sample.get("action_is_pad"),
            "action_dim_is_pad": sample.get("action_dim_is_pad"),
            "image_is_pad": sample.get("image_is_pad"),
            **{
                key: sample[key]
                for key in ("action_ctx", "action_ctx_mask", "vggt_ctx", "vggt_ctx_mask")
                if key in sample
            },
        }

    def set_frozen_towers_eval(self) -> None:
        """Keep frozen Cosmos/VGGT in eval while action_expert trains."""
        vt = self.video_tower
        for mod in (getattr(vt, "vae", None), getattr(vt, "text_encoder", None)):
            if mod is not None:
                mod.eval()
        if getattr(vt, "transformer", None) is not None and getattr(vt, "freeze_transformer", False):
            vt.transformer.eval()
        if self.vggt_tower is not None:
            self.vggt_tower.eval()
        self.action_expert.train()

    def uses_dual_vggt_cross_attn(self) -> bool:
        return self.action_cross_attn_mode == "dual_cosmos_vggt"

    def _embed_action_contexts(
        self,
        action_ctx: torch.Tensor,
        vggt_ctx: Optional[torch.Tensor] = None,
        *,
        context_emb: Optional[torch.Tensor] = None,
        vggt_context_emb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Project Cosmos hook / VGGT registers once (train + cached eval)."""
        if context_emb is None:
            context_emb = self.action_expert.text_embedding(action_ctx)
        if (
            vggt_ctx is not None
            and vggt_context_emb is None
            and getattr(self.action_expert, "vggt_embedding", None) is not None
        ):
            vggt_context_emb = self.action_expert.vggt_embedding(
                vggt_ctx.to(device=self.device, dtype=self.torch_dtype)
            )
        return context_emb, vggt_context_emb

    @torch.no_grad()
    def _resolve_vggt_context(
        self,
        video: torch.Tensor,
        *,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.uses_dual_vggt_cross_attn():
            return None, None
        if inputs is not None and "vggt_ctx" in inputs and "vggt_ctx_mask" in inputs:
            # Scene registers are inference-only; never backprop into VGGT or external cache.
            return inputs["vggt_ctx"].detach(), inputs["vggt_ctx_mask"]
        if self.vggt_tower is None:
            raise RuntimeError(
                "dual_cosmos_vggt requires vggt_tower or precomputed vggt_ctx in inputs."
            )
        vggt_video = video
        if inputs is not None and inputs.get("vggt_video") is not None:
            vggt_video = inputs["vggt_video"]
        vggt_video = vggt_video.to(device=self.device, dtype=self.torch_dtype)
        if vggt_video.ndim != 5:
            raise ValueError(f"vggt_video must be [B,3,T,H,W], got {tuple(vggt_video.shape)}")
        if not self.vggt_use_full_video:
            vggt_video = vggt_video[:, :, -1:, :, :]
        return self.vggt_tower.extract_register_context(vggt_video)

    def _resolve_action_context(
        self,
        input_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs is not None and "action_ctx" in inputs and "action_ctx_mask" in inputs:
            return inputs["action_ctx"], inputs["action_ctx_mask"]
        video = inputs.get("video") if inputs is not None else None
        if getattr(self.video_tower, "transformer", None) is not None:
            if video is None:
                raise ValueError("Cosmos action context requires inputs['video'] (DiT4DiT frame-0 cond).")
            if video.ndim != 5:
                raise ValueError(f"inputs['video'] must be [B,3,T,H,W], got {tuple(video.shape)}")
            # Use current frame only for Cosmos action-context extraction.
            video_current = video[:, :, -1:, :, :]
            _, action_ctx, action_ctx_mask = self.video_tower.forward_joint_step(
                video_current,
                prompt_embeds,
                compute_video_loss=False,
            )
            return action_ctx, action_ctx_mask
        if video is None:
            raise ValueError("extract_action_context requires video tensor.")
        if video.ndim != 5:
            raise ValueError(f"inputs['video'] must be [B,3,T,H,W], got {tuple(video.shape)}")
        return self.video_tower.extract_action_context(video[:, :, -1:, :, :], prompt_embeds)

    @staticmethod
    def _dim_valid_mask(
        action_dim_is_pad: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if action_dim_is_pad is None:
            return None
        pad = action_dim_is_pad.to(device=device)
        if pad.ndim == 1:
            return (~pad).to(dtype=dtype).view(1, 1, -1)
        if pad.ndim == 2:
            return (~pad).to(dtype=dtype).unsqueeze(0)
        return (~pad).to(dtype=dtype)

    def _compute_action_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        action_dim_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        loss = F.mse_loss(pred.float(), target.float(), reduction="none")
        dim_valid = self._dim_valid_mask(action_dim_is_pad, loss.device, loss.dtype)
        if dim_valid is not None:
            loss = loss * dim_valid
        if action_is_pad is not None:
            token_valid = (~action_is_pad).to(device=loss.device, dtype=loss.dtype).unsqueeze(-1)
            loss = loss * token_valid
        if dim_valid is not None and action_is_pad is not None:
            token_valid = (~action_is_pad).float().unsqueeze(-1)
            denom = (dim_valid * token_valid).sum().clamp(min=1.0)
            return loss.sum() / denom
        if dim_valid is not None:
            denom = dim_valid.sum().clamp(min=1.0)
            return loss.sum() / denom
        if action_is_pad is not None:
            token_valid = (~action_is_pad).float().unsqueeze(-1)
            denom = token_valid.sum().clamp(min=1.0)
            return loss.sum() / denom
        return loss.mean()

    def _action_proprio_future(
        self,
        action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        action_dim_is_pad: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        w = self.past_action_window_size
        proprio, future = split_proprio_future(action, w)
        future_pad = split_proprio_future_pad(action_is_pad, w)
        future_dim_pad = split_proprio_future_dim_pad(action_dim_is_pad, w)
        return proprio, future, future_pad, future_dim_pad

    def _action_history_future(
        self,
        action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        action_dim_is_pad: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.action_history_window is None:
            raise RuntimeError("action_history_window is required for history-mode training.")
        w = self.action_history_window
        future_horizon = self.action_future_horizon
        history, future = split_history_future(action, w, future_horizon=future_horizon)
        future_pad = split_history_future_pad(action_is_pad, w)
        future_dim_pad = split_history_future_dim_pad(action_dim_is_pad, w)
        return history, future, future_pad, future_dim_pad

    def _predict_velocity(
        self,
        noisy_action: torch.Tensor,
        timestep_disc: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        *,
        context_emb: Optional[torch.Tensor] = None,
        vggt_context: Optional[torch.Tensor] = None,
        vggt_context_mask: Optional[torch.Tensor] = None,
        vggt_context_emb: Optional[torch.Tensor] = None,
        proprio_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.action_head != "fm":
            raise RuntimeError("_predict_velocity requires action_head='fm'.")
        return self.action_expert(
            noisy_action,
            timestep_disc,
            action_context,
            action_context_mask,
            context_emb=context_emb,
            vggt_context=vggt_context,
            vggt_context_mask=vggt_context_mask,
            vggt_context_emb=vggt_context_emb,
            proprio_tokens=proprio_tokens,
        )

    def _predict_action_chunk(
        self,
        action_tokens: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        *,
        context_emb: Optional[torch.Tensor] = None,
        vggt_context: Optional[torch.Tensor] = None,
        vggt_context_mask: Optional[torch.Tensor] = None,
        vggt_context_emb: Optional[torch.Tensor] = None,
        proprio_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.action_head == "act":
            return self.action_expert(
                action_tokens,
                action_context,
                action_context_mask,
                context_emb=context_emb,
                vggt_context=vggt_context,
                vggt_context_mask=vggt_context_mask,
                vggt_context_emb=vggt_context_emb,
                proprio_tokens=proprio_tokens,
            )
        raise RuntimeError("_predict_action_chunk for ACT requires action_head='act'.")

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        context = inputs["context"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        action_dim_is_pad = inputs["action_dim_is_pad"]
        loss_dict: Dict[str, float] = {}

        loss_video = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        prompt_embeds = context
        if getattr(self.video_tower, "transformer", None) is not None:
            compute_video = self.loss_lambda_video > 0
            if compute_video or self.loss_lambda_action > 0:
                if compute_video:
                    loss_video, action_ctx, action_ctx_mask = self.video_tower.forward_joint_step(
                        inputs["video"],
                        prompt_embeds,
                        compute_video_loss=True,
                    )
                    loss_dict["loss_video"] = float(self.loss_lambda_video * loss_video.detach().item())
                else:
                    with torch.inference_mode():
                        action_ctx, action_ctx_mask = self._resolve_action_context(
                            input_latents, prompt_embeds, inputs=inputs
                        )
                    # Clone so action_expert cross-attn embed can backprop (inference tensors cannot).
                    action_ctx = action_ctx.detach().clone()
                    action_ctx_mask = action_ctx_mask.detach().clone()
            else:
                action_ctx = torch.zeros(
                    (input_latents.shape[0], 4, self.text_dim),
                    device=self.device,
                    dtype=self.torch_dtype,
                )
                action_ctx_mask = torch.ones(
                    (input_latents.shape[0], 4), device=self.device, dtype=torch.bool
                )
        else:
            action_ctx, action_ctx_mask = self.video_tower.extract_action_context(
                inputs["video"], prompt_embeds
            )

        vggt_ctx, vggt_ctx_mask = self._resolve_vggt_context(inputs["video"], inputs=inputs)

        if "action_ctx" in inputs and (
            not self.uses_dual_vggt_cross_attn() or inputs.get("vggt_ctx") is not None
        ):
            inputs.pop("video", None)
            inputs.pop("vggt_video", None)

        repeats = max(1, int(getattr(self, "repeated_action_steps", 1)))
        action_ctx, action_ctx_mask, action, action_is_pad, action_dim_is_pad = (
            self._repeat_action_batch(
                repeats, action_ctx, action_ctx_mask, action, action_is_pad, action_dim_is_pad
            )
        )
        if vggt_ctx is not None:
            vggt_ctx = self._repeat_batch_dim0(vggt_ctx, repeats)
            vggt_ctx_mask = self._repeat_batch_dim0(vggt_ctx_mask, repeats)

        context_emb, vggt_context_emb = self._embed_action_contexts(action_ctx, vggt_ctx)

        use_fp32_action = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=torch.float32, enabled=use_fp32_action):
            if self.uses_history_action_input():
                history, future_action, future_pad, future_dim_pad = self._action_history_future(
                    action, action_is_pad, action_dim_is_pad
                )
                if self.action_head == "act":
                    pred_action = self._predict_action_chunk(
                        history,
                        action_ctx,
                        action_ctx_mask,
                        context_emb=context_emb,
                        vggt_context=vggt_ctx,
                        vggt_context_mask=vggt_ctx_mask,
                        vggt_context_emb=vggt_context_emb,
                    )
                    loss_action = self._compute_action_loss(
                        pred_action, future_action, future_pad, future_dim_pad
                    )
                    action_est = pred_action
                else:
                    batch_size = future_action.shape[0]
                    flow_source = history_to_flow_source(history, future_action.shape[1])
                    t_cont = self.action_fm.sample_training_t(
                        batch_size, future_action.device, future_action.dtype
                    )
                    t_view = t_cont.view(-1, *([1] * (future_action.ndim - 1)))
                    noisy_action = self.action_fm.corrupt(future_action, flow_source, t_cont)
                    target_velocity = self.action_fm.training_target(future_action, flow_source)
                    t_disc = self.action_fm.discretize_t(t_cont)
                    pred_velocity = self._predict_velocity(
                        noisy_action,
                        t_disc,
                        action_ctx,
                        action_ctx_mask,
                        context_emb=context_emb,
                        vggt_context=vggt_ctx,
                        vggt_context_mask=vggt_ctx_mask,
                        vggt_context_emb=vggt_context_emb,
                    )
                    loss_action = self._compute_action_loss(
                        pred_velocity, target_velocity, future_pad, future_dim_pad
                    )
                    action_est = noisy_action - t_view * pred_velocity
            else:
                proprio, future_action, future_pad, future_dim_pad = self._action_proprio_future(
                    action, action_is_pad, action_dim_is_pad
                )
                if self.action_head == "act":
                    placeholder = torch.zeros_like(future_action)
                    pred_action = self._predict_action_chunk(
                        placeholder,
                        action_ctx,
                        action_ctx_mask,
                        context_emb=context_emb,
                        vggt_context=vggt_ctx,
                        vggt_context_mask=vggt_ctx_mask,
                        vggt_context_emb=vggt_context_emb,
                        proprio_tokens=proprio,
                    )
                    loss_action = self._compute_action_loss(
                        pred_action, future_action, future_pad, future_dim_pad
                    )
                    action_est = pred_action
                else:
                    batch_size = future_action.shape[0]
                    noise = torch.randn_like(future_action)
                    t_cont = self.action_fm.sample_training_t(
                        batch_size, future_action.device, future_action.dtype
                    )
                    t_view = t_cont.view(-1, *([1] * (future_action.ndim - 1)))
                    noisy_action = self.action_fm.corrupt(future_action, noise, t_cont)
                    target_velocity = self.action_fm.training_target(future_action, noise)
                    t_disc = self.action_fm.discretize_t(t_cont)
                    pred_velocity = self._predict_velocity(
                        noisy_action,
                        t_disc,
                        action_ctx,
                        action_ctx_mask,
                        context_emb=context_emb,
                        vggt_context=vggt_ctx,
                        vggt_context_mask=vggt_ctx_mask,
                        vggt_context_emb=vggt_context_emb,
                        proprio_tokens=proprio,
                    )
                    loss_action = self._compute_action_loss(
                        pred_velocity, target_velocity, future_pad, future_dim_pad
                    )
                    action_est = noisy_action - t_view * pred_velocity
            loss_bone = bone_length_loss(
                action_est,
                future_action,
                action_is_pad=future_pad,
                action_dim_is_pad=future_dim_pad,
                norm_mean=self.action_norm_mean,
                norm_std=self.action_norm_std,
            )
            loss_bone_dir = bone_direction_loss(
                action_est,
                future_action,
                action_is_pad=future_pad,
                action_dim_is_pad=future_dim_pad,
                norm_mean=self.action_norm_mean,
                norm_std=self.action_norm_std,
            )
            loss_bone_hand = hand_bone_length_loss(
                action_est,
                future_action,
                action_is_pad=future_pad,
                action_dim_is_pad=future_dim_pad,
                norm_mean=self.action_norm_mean,
                norm_std=self.action_norm_std,
            )
            loss_hand_mse = hand_keypoints_mse_loss(
                action_est,
                future_action,
                action_is_pad=future_pad,
                action_dim_is_pad=future_dim_pad,
            )
        with torch.no_grad():
            loss_parts: list[tuple[str, torch.Tensor]] = [
                ("loss_action", self.loss_lambda_action * loss_action),
            ]
            if self.loss_lambda_bone > 0:
                loss_parts.append(("loss_bone", self.loss_lambda_bone * loss_bone))
            if self.loss_lambda_bone_hand > 0:
                loss_parts.append(("loss_bone_hand", self.loss_lambda_bone_hand * loss_bone_hand))
            if self.loss_lambda_bone_dir > 0:
                loss_parts.append(("loss_bone_dir", self.loss_lambda_bone_dir * loss_bone_dir))
            if self.loss_lambda_hand_mse > 0:
                loss_parts.append(("loss_hand_mse", self.loss_lambda_hand_mse * loss_hand_mse))
            scaled = torch.stack([part.reshape(()) for _, part in loss_parts])
            for (name, _), value in zip(loss_parts, scaled.cpu().tolist()):
                loss_dict[name] = float(value)
        loss_total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action * loss_action
            + self.loss_lambda_bone * loss_bone
            + self.loss_lambda_bone_hand * loss_bone_hand
            + self.loss_lambda_bone_dir * loss_bone_dir
            + self.loss_lambda_hand_mse * loss_hand_mse
        )
        return loss_total, loss_dict

    @torch.no_grad()
    def predict_action_fm(
        self,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        num_frames: int,
        *,
        batch_size: int = 1,
        context_emb: Optional[torch.Tensor] = None,
        vggt_context: Optional[torch.Tensor] = None,
        vggt_context_mask: Optional[torch.Tensor] = None,
        vggt_context_emb: Optional[torch.Tensor] = None,
        proprio_tokens: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Euler FM denoise: returns normalized actions [B, T, D]."""
        if self.action_head != "fm" or self.action_fm is None:
            raise RuntimeError("predict_action_fm requires action_head='fm'.")
        if context_emb is None:
            context_emb = self.action_expert.text_embedding(action_context)

        def _predict_velocity(noisy: torch.Tensor, t_disc: torch.Tensor) -> torch.Tensor:
            return self._predict_velocity(
                noisy,
                t_disc,
                action_context,
                action_context_mask,
                context_emb=context_emb,
                vggt_context=vggt_context,
                vggt_context_mask=vggt_context_mask,
                vggt_context_emb=vggt_context_emb,
                proprio_tokens=proprio_tokens,
            )

        initial_state: Optional[torch.Tensor] = None
        if self.uses_history_action_input():
            if history is None:
                raise ValueError("history is required for FM predict when uses_history_action_input=True")
            initial_state = history_to_flow_source(history, int(num_frames))

        return self.action_fm.denoise_euler(
            _predict_velocity,
            initial_state=initial_state,
            batch_size=int(batch_size),
            seq_len=int(num_frames),
            action_dim=self.action_expert.raw_action_dim,
            device=self.device,
            dtype=self.torch_dtype,
        )

    @torch.no_grad()
    def predict_action_act(
        self,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        num_frames: int,
        *,
        batch_size: int = 1,
        context_emb: Optional[torch.Tensor] = None,
        vggt_context: Optional[torch.Tensor] = None,
        vggt_context_mask: Optional[torch.Tensor] = None,
        vggt_context_emb: Optional[torch.Tensor] = None,
        proprio_tokens: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Direct regression: returns normalized actions [B, T, D]."""
        if self.uses_history_action_input():
            if history is None:
                raise ValueError("history is required when uses_history_action_input=True")
            pred = self._predict_action_chunk(
                history,
                action_context,
                action_context_mask,
                context_emb=context_emb,
                vggt_context=vggt_context,
                vggt_context_mask=vggt_context_mask,
                vggt_context_emb=vggt_context_emb,
            )
            return pred[:, : int(num_frames)]
        placeholder = torch.zeros(
            batch_size,
            int(num_frames),
            self.action_expert.raw_action_dim,
            device=self.device,
            dtype=self.torch_dtype,
        )
        return self._predict_action_chunk(
            placeholder,
            action_context,
            action_context_mask,
            context_emb=context_emb,
            vggt_context=vggt_context,
            vggt_context_mask=vggt_context_mask,
            vggt_context_emb=vggt_context_emb,
            proprio_tokens=proprio_tokens,
        )

    @torch.no_grad()
    def predict_action(
        self,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        num_frames: int,
        *,
        batch_size: int = 1,
        context_emb: Optional[torch.Tensor] = None,
        vggt_context: Optional[torch.Tensor] = None,
        vggt_context_mask: Optional[torch.Tensor] = None,
        vggt_context_emb: Optional[torch.Tensor] = None,
        proprio_tokens: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.action_head == "act":
            return self.predict_action_act(
                action_context,
                action_context_mask,
                num_frames,
                batch_size=batch_size,
                context_emb=context_emb,
                vggt_context=vggt_context,
                vggt_context_mask=vggt_context_mask,
                vggt_context_emb=vggt_context_emb,
                proprio_tokens=proprio_tokens,
                history=history,
            )
        return self.predict_action_fm(
            action_context,
            action_context_mask,
            num_frames,
            batch_size=batch_size,
            context_emb=context_emb,
            vggt_context=vggt_context,
            vggt_context_mask=vggt_context_mask,
            vggt_context_emb=vggt_context_emb,
            proprio_tokens=proprio_tokens,
            history=history,
        )

    @torch.no_grad()
    def predict_video(
        self,
        video: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        num_inference_steps: int | None = None,
        num_pixel_frames_out: int | None = None,
        num_frames_in: int | None = None,
        seq_len: int | None = None,
        action_video_freq_ratio: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Image2World via official Cosmos2.5 pipeline. Returns (video [B,T,3,H,W] in [0,1], latents=None)."""
        tower = self.video_tower
        if not isinstance(tower, CosmosVideoTower):
            raise RuntimeError("predict_video requires CosmosVideoTower.")
        return tower.generate_video(
            video,
            prompt_embeds,
            num_inference_steps=num_inference_steps,
            num_pixel_frames_out=num_pixel_frames_out,
            num_frames_in=num_frames_in,
            seq_len=seq_len,
            action_video_freq_ratio=action_video_freq_ratio,
            generator=generator,
        )

    @torch.no_grad()
    def predict_action_chunk(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Eval helper: predict future horizon (FM denoise or ACT regression)."""
        action_ctx, action_ctx_mask = self._resolve_action_context(
            inputs["input_latents"], inputs["context"], inputs=inputs
        )
        vggt_ctx, vggt_ctx_mask = self._resolve_vggt_context(
            inputs.get("video"), inputs=inputs
        ) if "video" in inputs else (None, None)
        action = inputs["action"]
        batch_size = int(action_ctx.shape[0])
        context_emb = inputs.get("context_emb")
        vggt_context_emb = inputs.get("vggt_context_emb")
        if context_emb is None or (vggt_ctx is not None and vggt_context_emb is None):
            context_emb, vggt_context_emb = self._embed_action_contexts(
                action_ctx,
                vggt_ctx,
                context_emb=context_emb,
                vggt_context_emb=vggt_context_emb,
            )
        if self.uses_history_action_input():
            history, future, _, _ = self._action_history_future(
                action, inputs.get("action_is_pad"), inputs.get("action_dim_is_pad")
            )
            return self.predict_action(
                action_ctx,
                action_ctx_mask,
                future.shape[1],
                batch_size=batch_size,
                context_emb=context_emb,
                vggt_context=vggt_ctx,
                vggt_context_mask=vggt_ctx_mask,
                vggt_context_emb=vggt_context_emb,
                history=history,
            )
        proprio, future, _, _ = self._action_proprio_future(
            action, inputs.get("action_is_pad"), inputs.get("action_dim_is_pad")
        )
        return self.predict_action(
            action_ctx,
            action_ctx_mask,
            future.shape[1],
            batch_size=batch_size,
            context_emb=context_emb,
            vggt_context=vggt_ctx,
            vggt_context_mask=vggt_ctx_mask,
            vggt_context_emb=vggt_context_emb,
            proprio_tokens=proprio,
        )

    @torch.no_grad()
    def predict_action_fm_chunk(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Eval helper: full FM denoise on the clip future action horizon."""
        if self.action_head != "fm":
            raise RuntimeError("predict_action_fm_chunk requires action_head='fm'.")
        return self.predict_action_chunk(inputs)

    @torch.no_grad()
    def infer_action_sequence(
        self,
        input_image: torch.Tensor,
        prompt: str,
        num_frames: int,
        *,
        prompt_cache: Optional[Any] = None,
        processor: Optional[Any] = None,
        denormalize: bool = False,
    ) -> torch.Tensor:
        """Deploy FM chunk prediction (see ``ActionInferenceSession``)."""
        from phi0.inference.session import ActionInferenceSession

        session = ActionInferenceSession(self, processor=processor)
        session.prefill_from_image(input_image, prompt, prompt_cache=prompt_cache)
        return session.predict(num_frames, denormalize=denormalize)

    def load_checkpoint(self, path: str, optimizer=None, strict_mot: bool = False) -> dict:
        del strict_mot
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "action_expert" in payload:
            load_action_expert_state_dict(self, payload["action_expert"], source=path)
            logger.info("Loaded action_expert-only checkpoint from %s", path)
        elif isinstance(payload, dict) and "model" in payload:
            load_model_state_dict(self, payload["model"], strict=False, source=path)
            logger.info("Loaded Phi_0 training checkpoint from %s", path)
        elif isinstance(payload, dict) and "mot" in payload:
            logger.warning(
                "FastWAM MoT checkpoint detected; loading action expert weights only (video tower unchanged)."
            )
            action_sd = {}
            for k, v in payload["mot"].items():
                if k.startswith("action."):
                    action_sd[k.replace("action.", "", 1)] = v
            if action_sd:
                self.action_expert.load_state_dict(action_sd, strict=False)
        elif isinstance(payload, dict) and "dit" in payload:
            logger.warning("Legacy Wan `dit` checkpoint ignored (Cosmos video tower).")
        else:
            raise ValueError(f"Checkpoint missing `model`, `action_expert`, or compatible keys: {path}")
        if optimizer is not None and isinstance(payload, dict) and "optimizer" in payload:
            try:
                optimizer.load_state_dict(payload["optimizer"])
            except ValueError as exc:
                logger.warning(
                    "Skipping optimizer state from %s (%s); using freshly initialized optimizer.",
                    path,
                    exc,
                )
        return payload

    def save_checkpoint(self, path: str, optimizer=None, step=None) -> None:
        payload = {
            "model": self.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
