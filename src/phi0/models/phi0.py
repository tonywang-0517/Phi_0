"""Phi_0 world-action model: Qwen3-VL tower + ACT/FM action head with optional VGGT cross-attn."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from phi0.models.action_act_dit import ActionACTDiT
from phi0.models.action_cross_attn import resolve_action_cross_attn_mode
from phi0.models.action_fm_dit import ActionFMDiT
from phi0.models.action_fm_scheduler import ActionFMConfig, ActionFlowMatching
from phi0.models.vlm.tower import Qwen3VLTower, SmokeVLMTower
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
from phi0.models.action_placeholder import make_future_action_placeholder

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
    """Qwen3-VL VLM tower for observation encoding; ACT/FM head with optional VGGT fusion."""

    def __init__(
        self,
        action_expert: ActionExpert,
        vlm_tower: Optional[Union[Qwen3VLTower, SmokeVLMTower]] = None,
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
        past_action_window_size: int = 1,
        action_history_window: int | None = None,
        action_future_horizon: int | None = None,
        vggt_use_full_video: bool = False,
    ):
        super().__init__()
        self.vlm_tower = vlm_tower
        # Backward-compatible alias for legacy scripts.
        self.video_tower = vlm_tower
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
        self.repeated_action_steps = 1
        self.action_fm = (
            ActionFlowMatching(action_fm_config or ActionFMConfig())
            if self.action_head == "fm"
            else None
        )
        self.register_buffer("action_norm_mean", torch.zeros(D_RAW), persistent=False)
        self.register_buffer("action_norm_std", torch.ones(D_RAW), persistent=False)
        self.register_buffer("action_norm_q01", torch.zeros(D_RAW), persistent=False)
        self.register_buffer("action_norm_q99", torch.ones(D_RAW), persistent=False)
        self.action_norm_mode: str = "z-score"
        self.action_normalize_gripper: bool = True
        self.robot_action_loss_type: str = "mse"
        self.to(self.device)

    def uses_vlm_tower(self) -> bool:
        return self.vlm_tower is not None

    def uses_cross_attn_context(self) -> bool:
        return self.action_cross_attn_mode != "self_only"

    def uses_history_action_input(self) -> bool:
        """Ablation path: symmetric history window + dit4dit_prefix_query encoder."""
        encoder = getattr(self.action_expert, "action_token_encoder", "linear")
        return str(encoder).strip().lower() == "dit4dit_prefix_query"

    def set_action_norm_stats(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        q01: Optional[torch.Tensor] = None,
        q99: Optional[torch.Tensor] = None,
        norm_mode: str = "z-score",
        normalize_gripper: bool = True,
    ) -> None:
        self.action_norm_mean.copy_(mean.to(dtype=torch.float32).view(-1))
        self.action_norm_std.copy_(std.to(dtype=torch.float32).view(-1).clamp(min=1e-6))
        if q01 is not None:
            self.action_norm_q01.copy_(q01.to(dtype=torch.float32).view(-1))
        else:
            self.action_norm_q01.copy_(self.action_norm_mean)
        if q99 is not None:
            self.action_norm_q99.copy_(q99.to(dtype=torch.float32).view(-1))
        else:
            self.action_norm_q99.copy_(self.action_norm_mean)
        self.action_norm_mode = str(norm_mode).strip().lower()
        self.action_normalize_gripper = bool(normalize_gripper)

    def _robot_action_norm_stats(self) -> dict[str, Any]:
        return {
            "norm_mode": self.action_norm_mode,
            "mean": self.action_norm_mean.detach().cpu().tolist(),
            "std": self.action_norm_std.detach().cpu().tolist(),
            "q01": self.action_norm_q01.detach().cpu().tolist(),
            "q99": self.action_norm_q99.detach().cpu().tolist(),
        }

    @classmethod
    def from_vlm_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        vlm_model_path: str | None = None,
        checkpoints_dir: str | None = None,
        local_files_only: bool = False,
        freeze_vlm: bool = True,
        attn_implementation: str = "flash_attention_2",
        action_dit_config: dict[str, Any] | None = None,
        action_head: str = "fm",
        action_fm_config: dict[str, Any] | None = None,
        raw_action_dim: int = D_RAW,
        loss_lambda_video: float = 0.0,
        loss_lambda_action: float = 1.0,
        loss_lambda_bone: float = 0.0,
        loss_lambda_bone_hand: float = 0.0,
        loss_lambda_bone_dir: float = 0.0,
        loss_lambda_hand_mse: float = 0.0,
        prompt_max_length: int = 512,
        past_action_window_size: int = 1,
        action_history_window: int | None = None,
        action_future_horizon: int | None = None,
        vggt_use_full_video: bool = False,
        vggt_tower: Optional[nn.Module] = None,
        **kwargs,
    ):
        del kwargs, loss_lambda_video
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required.")
        vlm_tower = Qwen3VLTower.from_pretrained(
            model_path=vlm_model_path,
            checkpoints_dir=checkpoints_dir,
            device=device,
            torch_dtype=torch_dtype,
            freeze=freeze_vlm,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
        )
        action_cfg = dict(action_dit_config)
        action_cfg["text_dim"] = int(vlm_tower.action_context_dim)
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

        action_expert = build_action_expert(
            action_head,
            action_cfg,
            raw_action_dim=raw_action_dim,
            device=device,
            torch_dtype=torch_dtype,
        )
        fm_cfg = ActionFMConfig(**dict(action_fm_config or {})) if action_head == "fm" else None
        return cls(
            action_expert=action_expert,
            vlm_tower=vlm_tower,
            vggt_tower=vggt_tower,
            device=device,
            torch_dtype=torch_dtype,
            loss_lambda_video=0.0,
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
        )

    @classmethod
    def from_action_only(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        action_dit_config: dict[str, Any] | None = None,
        action_head: str = "fm",
        action_fm_config: dict[str, Any] | None = None,
        raw_action_dim: int = D_RAW,
        loss_lambda_action: float = 1.0,
        loss_lambda_bone: float = 0.0,
        loss_lambda_bone_hand: float = 0.0,
        loss_lambda_bone_dir: float = 0.0,
        loss_lambda_hand_mse: float = 0.0,
        prompt_max_length: int = 512,
        past_action_window_size: int = 1,
        action_history_window: int | None = None,
        action_future_horizon: int | None = None,
        vggt_use_full_video: bool = False,
        vggt_tower: Optional[nn.Module] = None,
    ):
        """Build Phi0 with action head only (no VLM tower loaded)."""
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required.")
        action_cfg = dict(action_dit_config)
        if "text_dim" not in action_cfg:
            raise ValueError("action_dit_config.text_dim is required when VLM is disabled.")
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

        action_expert = build_action_expert(
            action_head,
            action_cfg,
            raw_action_dim=raw_action_dim,
            device=device,
            torch_dtype=torch_dtype,
        )
        fm_cfg = ActionFMConfig(**dict(action_fm_config or {})) if action_head == "fm" else None
        return cls(
            action_expert=action_expert,
            vlm_tower=None,
            vggt_tower=vggt_tower,
            device=device,
            torch_dtype=torch_dtype,
            loss_lambda_video=0.0,
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
        )

    @staticmethod
    def _dummy_action_context(
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        text_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = torch.zeros(batch_size, 1, text_dim, device=device, dtype=dtype)
        mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        return ctx, mask

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

    def build_inputs(self, sample, tiled: bool = False):
        del tiled
        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`action` must be [B,T,D], got {tuple(action.shape)}")

        action = action.to(device=self.device, dtype=self.torch_dtype)

        vggt_video = sample.get("vggt_video")
        if vggt_video is not None:
            vggt_video = vggt_video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            image_is_pad = sample.get("image_is_pad")
            if image_is_pad is not None and image_is_pad.shape[-1] != vggt_video.shape[2]:
                # Multi-frame pad flags from the clip; VGGT path uses current frame only.
                image_is_pad = image_is_pad[..., -vggt_video.shape[2] :]
            vggt_video = apply_video_pad_replacement(vggt_video, image_is_pad)

        vlm_keys = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
        vlm_inputs: Dict[str, Any] = {}
        if self.uses_vlm_tower():
            vlm_inputs = {
                key: sample[key].to(device=self.device, non_blocking=True)
                for key in vlm_keys
                if key in sample
            }
            if len(vlm_inputs) != len(vlm_keys):
                raise ValueError(f"Sample missing VLM keys; expected {vlm_keys}")

        return {
            **vlm_inputs,
            "vggt_video": vggt_video,
            "action": action,
            "action_is_pad": sample.get("action_is_pad"),
            "action_dim_is_pad": sample.get("action_dim_is_pad"),
            "image_is_pad": sample.get("image_is_pad"),
            **{
                key: sample[key]
                for key in (
                    "action_ctx",
                    "action_ctx_mask",
                    "vggt_ctx",
                    "vggt_ctx_mask",
                    "robot_action_7d",
                    "robot_future_delta_7d",
                )
                if key in sample
            },
        }

    def set_frozen_towers_eval(self) -> None:
        """Keep frozen VLM/VGGT in eval while action_expert trains."""
        if self.vlm_tower is not None and getattr(self.vlm_tower, "freeze", True):
            self.vlm_tower.eval()
        if self.vggt_tower is not None:
            self.vggt_tower.eval()
        self.action_expert.train()

    def uses_dual_vggt_cross_attn(self) -> bool:
        return self.action_cross_attn_mode == "dual_vlm_vggt"

    def uses_robot7d_action(self) -> bool:
        """LIBERO / robot arm: ACT head outputs normalized 7D (VLA-Adapter ACTION_DIM)."""
        from phi0.data.robot_action_norm import ROBOT_DIM

        return int(getattr(self.action_expert, "raw_action_dim", D_RAW)) == ROBOT_DIM

    def _embed_action_contexts(
        self,
        action_ctx: torch.Tensor,
        vggt_ctx: Optional[torch.Tensor] = None,
        *,
        context_emb: Optional[torch.Tensor] = None,
        vggt_context_emb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Project VLM hidden states / VGGT registers once (train + cached eval)."""
        # Clone tower outputs so gradient checkpointing can save cross-attn keys
        # (tensors from torch.inference_mode() precompute are not backward-safe).
        if context_emb is None:
            embed = getattr(self.action_expert, "text_embedding", None)
            ctx = action_ctx.detach().clone()
            context_emb = ctx if embed is None else embed(ctx)
        if (
            vggt_ctx is not None
            and vggt_context_emb is None
        ):
            vggt_in = vggt_ctx.detach().clone().to(device=self.device, dtype=self.torch_dtype)
            embed = getattr(self.action_expert, "vggt_embedding", None)
            if embed is None:
                vggt_context_emb = vggt_in
            else:
                vggt_context_emb = embed(vggt_in)
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
            return inputs["vggt_ctx"].detach().clone(), inputs["vggt_ctx_mask"]
        if self.vggt_tower is None:
            raise RuntimeError(
                "dual_vlm_vggt requires vggt_tower or precomputed vggt_ctx in inputs."
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
        *,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs is not None and "action_ctx" in inputs and "action_ctx_mask" in inputs:
            return inputs["action_ctx"].detach().clone(), inputs["action_ctx_mask"]
        if not self.uses_vlm_tower():
            batch_size = int(inputs["action"].shape[0]) if inputs is not None else 1
            return self._dummy_action_context(
                batch_size,
                device=self.device,
                dtype=self.torch_dtype,
                text_dim=self.text_dim,
            )
        required = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
        if inputs is None or any(k not in inputs for k in required):
            raise ValueError(f"VLM action context requires inputs {required}")
        return self.vlm_tower.extract_action_context(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["pixel_values"],
            inputs["image_grid_thw"],
        )

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

    def _compute_robot_action_decoder_loss(
        self,
        pred_norm_7d: torch.Tensor,
        target_norm_7d: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """L1/MSE on normalized 7D controls (VLA-Adapter ``L1Loss(pred, gt_norm)``)."""
        pred = pred_norm_7d
        target = target_norm_7d.to(device=pred.device, dtype=pred.dtype)
        loss_type = str(getattr(self, "robot_action_loss_type", "mse")).strip().lower()
        if loss_type == "l1":
            loss_fn = F.l1_loss
        else:
            loss_fn = F.mse_loss
        loss = loss_fn(pred, target, reduction="none")
        if action_is_pad is not None:
            token_valid = (~action_is_pad).to(device=loss.device, dtype=loss.dtype).unsqueeze(-1)
            loss = loss * token_valid
            denom = token_valid.sum().clamp(min=1.0) * float(pred.shape[-1])
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
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        action_dim_is_pad = inputs["action_dim_is_pad"]
        loss_dict: Dict[str, float] = {}

        if "action_ctx" in inputs and "action_ctx_mask" in inputs:
            action_ctx = inputs["action_ctx"]
            action_ctx_mask = inputs["action_ctx_mask"]
        elif self.uses_vlm_tower():
            with torch.inference_mode():
                action_ctx, action_ctx_mask = self._resolve_action_context(inputs=inputs)
            action_ctx = action_ctx.detach().clone()
            action_ctx_mask = action_ctx_mask.detach().clone()
        else:
            action_ctx, action_ctx_mask = self._dummy_action_context(
                action.shape[0],
                device=action.device,
                dtype=action.dtype,
                text_dim=self.text_dim,
            )

        vggt_video = inputs.get("vggt_video")
        vggt_ctx, vggt_ctx_mask = (
            self._resolve_vggt_context(vggt_video, inputs=inputs)
            if vggt_video is not None and self.uses_cross_attn_context()
            else (None, None)
        )

        if "action_ctx" in inputs and (
            not self.uses_dual_vggt_cross_attn() or inputs.get("vggt_ctx") is not None
        ):
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

        context_emb, vggt_context_emb = (
            self._embed_action_contexts(action_ctx, vggt_ctx)
            if self.uses_cross_attn_context()
            else (None, None)
        )

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

                placeholder = make_future_action_placeholder(

                    future_action.shape[0],

                    future_action.shape[1],

                    future_action.shape[2],

                    device=future_action.device,

                    dtype=future_action.dtype,

                )

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

                if (

                    inputs.get("robot_action_7d") is not None

                    or inputs.get("robot_future_delta_7d") is not None

                ) and self.uses_robot7d_action():

                    loss_action = self._compute_robot_action_decoder_loss(

                        pred_action,

                        future_action,

                        future_pad,

                    )

                else:

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

        need_bone = (

            self.loss_lambda_bone > 0

            or self.loss_lambda_bone_hand > 0

            or self.loss_lambda_bone_dir > 0

            or self.loss_lambda_hand_mse > 0

        )

        if need_bone:

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

        else:

            loss_bone = loss_bone_dir = loss_bone_hand = loss_hand_mse = loss_action.new_zeros(())
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
            self.loss_lambda_action * loss_action
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
        placeholder = make_future_action_placeholder(
            int(batch_size),
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
    def predict_video(self, *args, **kwargs):
        raise NotImplementedError("Video generation removed; Phi_0 uses Qwen3-VL observation tower only.")

    @torch.no_grad()
    def predict_action_chunk(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Eval helper: predict future horizon (FM denoise or ACT regression)."""
        action_ctx, action_ctx_mask = self._resolve_action_context(inputs=inputs)
        vggt_ctx, vggt_ctx_mask = (
            self._resolve_vggt_context(inputs.get("vggt_video"), inputs=inputs)
            if inputs.get("vggt_video") is not None
            else (None, None)
        )
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
            logger.warning("Legacy Wan `dit` checkpoint ignored (VLM tower unchanged).")
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
