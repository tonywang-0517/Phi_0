"""Cosmos VAE + DiT4DiT-style action context from transformer hidden states.

Action cross-attention uses hooked Cosmos DiT block outputs (B, S, D), NOT raw
100352-d prompt embeds. Prompt embeds only feed the video transformer internally.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from phi0.models.cosmos.loader import CosmosComponents, load_cosmos_predict25_2b

logger = logging.getLogger(__name__)


def _retrieve_latents(encoder_output: torch.Tensor, *, vae_sample: bool = False) -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist"):
        dist = encoder_output.latent_dist
        if vae_sample:
            return dist.sample()
        if hasattr(dist, "mode"):
            return dist.mode()
        if hasattr(dist, "mean"):
            return dist.mean
        return dist.sample()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents from VAE encode output.")


def cosmos_transformer_token_dim(transformer: nn.Module) -> int:
    """DiT4DiT: action cross_attention_dim matches Cosmos DiT token channels."""
    cfg = getattr(transformer, "config", None)
    if cfg is None:
        return 2048
    heads = int(getattr(cfg, "num_attention_heads", 16))
    head_dim = int(getattr(cfg, "attention_head_dim", 128))
    return heads * head_dim


def hidden_to_action_tokens(hidden: torch.Tensor) -> torch.Tensor:
    """Normalize hook output to (B, S, D) — same as DiT4DiT ``_hidden_to_bsd``."""
    if hidden.ndim == 3:
        return hidden
    if hidden.ndim == 5:
        b, c, t, h, w = hidden.shape
        return hidden.permute(0, 2, 3, 4, 1).contiguous().view(b, t * h * w, c)
    raise ValueError(f"Unsupported hidden shape {tuple(hidden.shape)}; expected 3D or 5D.")


class CosmosVideoTower(nn.Module):
    """Cosmos-Predict2.5 VAE + DiT; DiT4DiT joint training with hooked action context."""

    def __init__(
        self,
        components: CosmosComponents,
        extract_layer: int = 17,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        freeze_text_encoder: bool = True,
        freeze_vae: bool = True,
        freeze_transformer: bool = False,
        freeze: bool | None = None,
        detach_action_context: bool = True,
        action_context_mode: str = "first_frame",
        capture_stochastic: bool = False,
        vae_sample: bool = False,
        conditional_frame_timestep: float = 0.0001,
        enable_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.vae = components.vae
        self.transformer = components.transformer
        self.text_encoder = components.text_encoder
        self.tokenizer = components.tokenizer
        if components.latents_mean is not None:
            self.register_buffer("latents_mean", components.latents_mean, persistent=False)
        else:
            self.latents_mean = None
        if components.latents_std is not None:
            self.register_buffer("latents_std", components.latents_std, persistent=False)
        else:
            self.latents_std = None
        self.vae_scale_factor_spatial = int(components.vae_scale_factor_spatial)
        self.vae_scale_factor_temporal = int(components.vae_scale_factor_temporal)
        self.text_embed_dim = int(components.text_embed_dim)
        self.latent_channels = int(components.transformer_in_channels) - 1
        self.extract_layer = int(extract_layer)
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.detach_action_context = bool(detach_action_context)
        mode = str(action_context_mode).strip().lower()
        if mode not in {"first_frame", "full_clip"}:
            raise ValueError(f"action_context_mode must be first_frame or full_clip, got {action_context_mode!r}")
        self.action_context_mode = mode
        self.capture_stochastic = bool(capture_stochastic)
        self.vae_sample = bool(vae_sample)
        self.conditional_frame_timestep = float(conditional_frame_timestep)
        self.action_context_dim = (
            cosmos_transformer_token_dim(self.transformer) if self.transformer is not None else 1024
        )
        self._hook_handle = None
        self._cached_hidden: List[torch.Tensor] = []
        self._capture_hidden_enabled = True
        if self.transformer is not None:
            self._register_transformer_hook()
            if enable_gradient_checkpointing and hasattr(self.transformer, "enable_gradient_checkpointing"):
                self.transformer.enable_gradient_checkpointing()
                logger.info("Cosmos transformer gradient checkpointing enabled")

        if freeze is not None:
            freeze_text_encoder = freeze_vae = freeze_transformer = bool(freeze)
        self._set_requires_grad(self.text_encoder, not freeze_text_encoder)
        self._set_requires_grad(self.vae, not freeze_vae)
        self._set_requires_grad(self.transformer, not freeze_transformer)

    @staticmethod
    def _set_requires_grad(module: Optional[nn.Module], trainable: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = bool(trainable)

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        extract_layer: int = 17,
        base_model: str | None = None,
        revision: str = "diffusers/base/post-trained",
        checkpoints_dir: str | None = None,
        load_text_encoder: bool = True,
        load_transformer: bool = True,
        local_files_only: bool = True,
        freeze_text_encoder: bool = True,
        freeze_vae: bool = True,
        freeze_transformer: bool = False,
        freeze: bool | None = None,
        detach_action_context: bool = True,
        action_context_mode: str = "first_frame",
        capture_stochastic: bool = False,
        vae_sample: bool = False,
        conditional_frame_timestep: float = 0.0001,
        enable_gradient_checkpointing: bool = False,
        **kwargs,
    ) -> "CosmosVideoTower":
        del kwargs  # legacy: context_hidden_dim, num_context_tokens (SmokeVideoTower only)
        components = load_cosmos_predict25_2b(
            device=device,
            torch_dtype=torch_dtype,
            base_model=base_model,
            revision=revision,
            checkpoints_dir=checkpoints_dir,
            load_text_encoder=load_text_encoder,
            load_transformer=load_transformer,
            local_files_only=local_files_only,
        )
        return cls(
            components=components,
            extract_layer=extract_layer,
            device=device,
            torch_dtype=torch_dtype,
            freeze_text_encoder=freeze_text_encoder,
            freeze_vae=freeze_vae,
            freeze_transformer=freeze_transformer,
            freeze=freeze,
            detach_action_context=detach_action_context,
            action_context_mode=action_context_mode,
            capture_stochastic=capture_stochastic,
            vae_sample=vae_sample,
            conditional_frame_timestep=conditional_frame_timestep,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
        )

    def _register_transformer_hook(self) -> None:
        if self.transformer is None or not hasattr(self.transformer, "transformer_blocks"):
            return
        blocks = self.transformer.transformer_blocks
        if self.extract_layer < 0 or self.extract_layer >= len(blocks):
            raise ValueError(
                f"extract_layer={self.extract_layer} out of bounds for {len(blocks)} blocks"
            )
        target = blocks[self.extract_layer]

        def hook_fn(_module, _inp, out):
            if not self._capture_hidden_enabled:
                return
            if torch.is_tensor(out):
                self._cached_hidden.append(out)
            elif isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
                self._cached_hidden.append(out[0])

        self._hook_handle = target.register_forward_hook(hook_fn)

    def _normalize_latents(self, z: torch.Tensor) -> torch.Tensor:
        if self.latents_mean is None or self.latents_std is None:
            return z
        return (z - self.latents_mean) / self.latents_std

    def _action_context_from_hook(self, detach: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._cached_hidden:
            raise RuntimeError(
                "No Cosmos transformer hidden captured. Check transformer_blocks / extract_layer."
            )
        use_detach = self.detach_action_context if detach is None else bool(detach)
        hidden = self._cached_hidden[-1].to(dtype=self.torch_dtype)
        if use_detach:
            hidden = hidden.detach()
        if self.action_context_mode == "first_frame" and hidden.ndim == 5:
            hidden = hidden[:, :, 0:1, :, :]
        tokens = hidden_to_action_tokens(hidden)
        mask = torch.ones((tokens.shape[0], tokens.shape[1]), device=tokens.device, dtype=torch.bool)
        return tokens, mask

    def _run_cosmos_transformer_with_hook(
        self,
        hidden_states: torch.Tensor,
        condition_mask: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        padding_mask: torch.Tensor,
        *,
        enable_hook: bool = True,
    ) -> torch.Tensor:
        """Single DiT forward; optionally capture hook hidden at ``extract_layer``."""
        transformer_dtype = self.transformer.dtype
        if enable_hook:
            self._cached_hidden.clear()
            self._capture_hidden_enabled = True
        out = self.transformer(
            hidden_states=hidden_states.to(dtype=transformer_dtype),
            condition_mask=condition_mask.to(dtype=transformer_dtype),
            timestep=timestep.to(dtype=transformer_dtype),
            encoder_hidden_states=prompt_embeds.to(dtype=transformer_dtype),
            padding_mask=padding_mask.to(dtype=transformer_dtype),
            return_dict=False,
        )[0]
        if enable_hook:
            self._capture_hidden_enabled = False
        return out

    def _forward_cosmos_joint_unified(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        compute_video_loss: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One DiT forward: hook capture for action (+ optional FM video loss on same pass)."""
        if self.transformer is None:
            zero = torch.tensor(0.0, device=latents.device, dtype=torch.float32)
            empty = torch.zeros((latents.shape[0], 4, self.action_context_dim), device=latents.device, dtype=self.torch_dtype)
            mask = torch.ones((latents.shape[0], 4), device=latents.device, dtype=torch.bool)
            return zero, empty, mask

        zero = torch.tensor(0.0, device=latents.device, dtype=torch.float32)
        if not compute_video_loss:
            hook_latents = self._latents_for_action_context(latents)
            grad_enabled = not self.detach_action_context
            capture_ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
            with capture_ctx:
                b, _, t, h, w = hook_latents.shape
                cond_latents, cond_mask, cond_indicator, _ = self._build_cond_tensors(hook_latents)
                if self.capture_stochastic and self.training:
                    noise = torch.randn_like(hook_latents)
                    sigma = torch.rand(b, device=hook_latents.device, dtype=hook_latents.dtype).view(
                        b, 1, 1, 1, 1
                    )
                else:
                    noise = torch.zeros_like(hook_latents)
                    sigma = torch.zeros(b, device=hook_latents.device, dtype=hook_latents.dtype).view(
                        b, 1, 1, 1, 1
                    )
                noisy = (1.0 - sigma) * hook_latents + sigma * noise
                noisy = cond_mask * hook_latents + (1.0 - cond_mask) * noisy
                in_timestep = cond_indicator * self.conditional_frame_timestep + (1.0 - cond_indicator) * sigma
                padding_mask = self._cosmos_padding_mask(hook_latents)
                self._run_cosmos_transformer_with_hook(
                    noisy,
                    cond_mask,
                    in_timestep,
                    prompt_embeds,
                    padding_mask,
                    enable_hook=True,
                )
            return zero, *self._action_context_from_hook()

        # FM + hook in one forward on full clip latents.
        b, _, t, h, w = latents.shape
        if t <= 1:
            return self._forward_cosmos_joint_unified(latents, prompt_embeds, compute_video_loss=False)

        compute_dtype = self.transformer.dtype
        grad_enabled = not self.detach_action_context or self.training
        fm_ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
        with fm_ctx:
            cond_latents, cond_mask, cond_indicator, cond_count = self._build_cond_tensors(latents)
            cond_mask_t = cond_mask.to(dtype=compute_dtype)
            cond_timestep = torch.ones_like(cond_indicator, dtype=compute_dtype) * self.conditional_frame_timestep
            padding_mask = self._cosmos_padding_mask(latents)

            x0_future = latents[:, :, cond_count:].to(dtype=compute_dtype)
            if x0_future.numel() == 0:
                return zero, *self._action_context_from_hook()

            t_noise = torch.rand((b,), device=latents.device, dtype=compute_dtype).view(b, 1, 1, 1, 1)
            z_future = torch.randn_like(x0_future)
            xt_future = (1.0 - t_noise) * x0_future + t_noise * z_future

            xt_full = torch.randn_like(latents, dtype=compute_dtype)
            t_sup = xt_future.shape[2]
            xt_full[:, :, cond_count : cond_count + t_sup] = xt_future

            t_b1t11 = torch.zeros_like(cond_indicator, dtype=compute_dtype)
            t_b1t11[:, :, cond_count : cond_count + t_sup] = t_noise

            in_latents = cond_mask_t * cond_latents.to(dtype=compute_dtype) + (1.0 - cond_mask_t) * xt_full
            in_timestep = cond_indicator.to(dtype=compute_dtype) * cond_timestep + (
                1.0 - cond_indicator.to(dtype=compute_dtype)
            ) * t_b1t11

            v_pred = self._run_cosmos_transformer_with_hook(
                in_latents,
                cond_mask_t,
                in_timestep,
                prompt_embeds,
                padding_mask,
                enable_hook=True,
            )

            v_tgt = (z_future - x0_future).to(dtype=v_pred.dtype)
            v_pred_future = v_pred[:, :, cond_count : cond_count + t_sup]
            loss_video = F.mse_loss(v_pred_future.float(), v_tgt.float())

        return loss_video, *self._action_context_from_hook()

    def _latents_for_action_context(self, latents: torch.Tensor) -> torch.Tensor:
        """Slice latents for hook capture (first_frame matches deploy single-frame prefill)."""
        if self.action_context_mode == "first_frame":
            return latents[:, :, 0:1, :, :]
        return latents

    def _build_cond_tensors(
        self, latents: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """DiT4DiT: first latent frame is conditional; returns cond_latents, cond_mask, cond_indicator, cond_count."""
        b, c, t, h, w = latents.shape
        cond_mask = torch.zeros((b, 1, t, h, w), device=latents.device, dtype=latents.dtype)
        cond_indicator = torch.zeros((b, 1, t, 1, 1), device=latents.device, dtype=latents.dtype)
        cond_count = 0
        if t > 0:
            cond_mask[:, :, 0:1] = 1.0
            cond_indicator[:, :, 0:1] = 1.0
            cond_count = 1
        cond_latents = latents * cond_mask
        return cond_latents, cond_mask, cond_indicator, cond_count

    def _cosmos_padding_mask(self, latents: torch.Tensor) -> torch.Tensor:
        """Cosmos DiT expects [1,1,H,W] pixel-space mask; it repeats along batch internally."""
        _, _, _, h, w = latents.shape
        return latents.new_zeros(
            1,
            1,
            h * self.vae_scale_factor_spatial,
            w * self.vae_scale_factor_spatial,
        )

    def _capture_action_context(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hook-only path (inference / legacy alias)."""
        _, action_ctx, action_mask = self._forward_cosmos_joint_unified(
            latents, prompt_embeds, compute_video_loss=False
        )
        return action_ctx, action_mask

    def _future_latent_flow_matching_loss(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Legacy alias; prefer ``forward_joint_step`` unified path."""
        loss_video, _, _ = self._forward_cosmos_joint_unified(
            latents, prompt_embeds, compute_video_loss=True
        )
        return loss_video

    def forward_joint_step(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        compute_video_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single DiT forward when video loss on; hook-only when off."""
        return self._forward_cosmos_joint_unified(
            latents, prompt_embeds, compute_video_loss=compute_video_loss
        )

    def _cosmos_transformer_step(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Legacy alias: joint forward returning (loss_video, _, action_ctx, action_mask)."""
        loss_video, action_ctx, action_mask = self.forward_joint_step(latents, prompt_embeds)
        return loss_video, loss_video.new_zeros(1), action_ctx, action_mask

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode [B,3,T,H,W] in [-1,1] to normalized latents [B,C,T',H',W']."""
        if video.ndim != 5:
            raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video.shape)}")
        b, _, t, h, w = video.shape
        if h % 16 != 0 or w % 16 != 0:
            raise ValueError(f"Video H,W must be multiples of 16, got {h}x{w}")

        self.vae.eval()
        video = video.to(device=self.device, dtype=self.vae.dtype)
        z = _retrieve_latents(self.vae.encode(video), vae_sample=self.vae_sample)
        z = z.to(dtype=self.torch_dtype)
        return self._normalize_latents(z)

    @torch.no_grad()
    def encode_frame(self, image: torch.Tensor) -> torch.Tensor:
        """Encode single frame [B,3,H,W] or [3,H,W] -> [B,C,1,H',W']."""
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"image must be [B,3,H,W], got {tuple(image.shape)}")
        _, _, h, w = image.shape
        if h % 16 != 0 or w % 16 != 0:
            raise ValueError(f"Image H,W must be multiples of 16, got {h}x{w}")
        clip = image.unsqueeze(2).to(device=self.device, dtype=self.vae.dtype)
        z = _retrieve_latents(self.vae.encode(clip), vae_sample=self.vae_sample)
        z = z.to(dtype=self.torch_dtype)
        return self._normalize_latents(z)

    def forward_training_step(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """DiT4DiT joint: future flow loss + hook action context."""
        loss_video, action_ctx, action_mask = self.forward_joint_step(latents, prompt_embeds)
        return loss_video, loss_video.detach(), action_ctx, action_mask

    def extract_action_context(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DiT4DiT inference: cond-frame hook context (detached when configured)."""
        return self._capture_action_context(latents, prompt_embeds)

    def build_action_context(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for ``extract_action_context`` (prompt_embeds = 100352-d Cosmos prompt)."""
        del text_mask
        return self.extract_action_context(latents, prompt_embeds)

    @torch.no_grad()
    def encode_prompt(self, prompt, max_sequence_length: int = 512):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError("Prompt encoding requires Cosmos text encoder.")
        prompt_list = [prompt] if isinstance(prompt, str) else list(prompt)
        input_ids_batch = []
        for sample in prompt_list:
            conversations = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant who will provide prompts to an image generator.",
                        }
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": sample}]},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                add_vision_id=False,
                max_length=max_sequence_length,
                truncation=True,
                padding="max_length",
            )
            if not isinstance(input_ids, list):
                input_ids = input_ids["input_ids"] if "input_ids" in input_ids else input_ids
            input_ids_batch.append(torch.LongTensor(input_ids))
        input_ids_batch = torch.stack(input_ids_batch, dim=0).to(self.device)
        outputs = self.text_encoder(input_ids_batch, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        normalized = []
        for layer_idx in range(1, len(hidden_states)):
            hs = hidden_states[layer_idx]
            normalized.append((hs - hs.mean(dim=-1, keepdim=True)) / (hs.std(dim=-1, keepdim=True) + 1e-8))
        embeds = torch.cat(normalized, dim=-1).to(dtype=self.torch_dtype, device=self.device)
        mask = torch.ones((embeds.shape[0], embeds.shape[1]), device=self.device, dtype=torch.bool)
        return embeds, mask

    def training_loss(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del padding_mask
        if self.transformer is None:
            return torch.tensor(0.0, device=latents.device, dtype=torch.float32)
        loss_video, _, _ = self.forward_joint_step(latents, prompt_embeds)
        return loss_video


class SmokeVideoTower(nn.Module):
    """CPU smoke stub: random DiT-like context tokens (no HF download)."""

    def __init__(
        self,
        latent_channels: int = 16,
        action_context_dim: int = 1024,
        num_context_tokens: int = 32,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        del kwargs
        super().__init__()
        self.latent_channels = latent_channels
        self.action_context_dim = action_context_dim
        self.num_context_tokens = int(num_context_tokens)
        self.text_embed_dim = action_context_dim
        self.vae_scale_factor_spatial = 8
        self.vae_scale_factor_temporal = 4
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        from phi0.models.cosmos.smoke_text import SmokeTextEncoder

        self.text_encoder = SmokeTextEncoder(embed_dim=512)
        self.tokenizer = None
        self.transformer = None
        self.vae = None

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        b, _, t, h, w = video.shape
        lh, lw = h // self.vae_scale_factor_spatial, w // self.vae_scale_factor_spatial
        lt = max(1, (t - 1) // self.vae_scale_factor_temporal + 1)
        return torch.randn(
            (b, self.latent_channels, lt, lh, lw),
            device=video.device,
            dtype=self.torch_dtype,
        )

    def encode_frame(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        b, _, h, w = image.shape
        lh, lw = h // self.vae_scale_factor_spatial, w // self.vae_scale_factor_spatial
        return torch.randn((b, self.latent_channels, 1, lh, lw), device=image.device, dtype=self.torch_dtype)

    def _smoke_action_context(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx = torch.randn(
            batch_size,
            self.num_context_tokens,
            self.action_context_dim,
            device=device,
            dtype=self.torch_dtype,
        )
        mask = torch.ones((batch_size, self.num_context_tokens), device=device, dtype=torch.bool)
        return ctx, mask

    def extract_action_context(self, latents, prompt_embeds):
        b, device = latents.shape[0], latents.device
        del prompt_embeds
        return self._smoke_action_context(b, device)

    def build_action_context(self, latents, prompt_embeds, text_mask=None):
        del text_mask
        return self.extract_action_context(latents, prompt_embeds)

    @torch.no_grad()
    def encode_prompt(self, prompt, max_sequence_length: int = 512):
        from phi0.models.cosmos.smoke_text import encode_smoke_prompt

        return encode_smoke_prompt(
            prompt,
            embed_dim=512,
            max_sequence_length=max_sequence_length,
            device=self.device,
            dtype=self.torch_dtype,
        )

    def training_loss(self, latents, prompt_embeds, padding_mask=None):
        del latents, prompt_embeds, padding_mask
        return torch.tensor(0.0, device=self.device, dtype=torch.float32)
