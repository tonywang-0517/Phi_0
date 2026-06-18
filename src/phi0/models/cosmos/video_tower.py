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
from diffusers.video_processor import VideoProcessor

from phi0.data.temporal_align import (
    build_video2world_prepare_clip,
    dit4dit_train_num_frames_out,
)
from phi0.models.cosmos.loader import CosmosComponents, _DefaultDummySafetyChecker, load_cosmos_predict25_2b
from phi0.models.cosmos.hook_forward import forward_transformer_to_hook_layer
from phi0.models.cosmos.video_fm import VideoFMConfig, sample_flow_matching_t

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
        hook_early_exit: bool = True,
        video_fm_config: VideoFMConfig | None = None,
    ):
        super().__init__()
        self.vae = components.vae
        self.transformer = components.transformer
        self.text_encoder = components.text_encoder
        self.tokenizer = components.tokenizer
        self.scheduler = components.scheduler
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
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
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
        self.hook_early_exit = bool(hook_early_exit)
        self.video_fm = video_fm_config or VideoFMConfig()
        self.action_context_dim = (
            cosmos_transformer_token_dim(self.transformer) if self.transformer is not None else 1024
        )
        self._hook_handle = None
        self._cached_hidden: List[torch.Tensor] = []
        self._capture_hidden_enabled = True
        self._predict_pipeline = None
        if self.transformer is not None:
            self._register_transformer_hook()
            if enable_gradient_checkpointing and hasattr(self.transformer, "enable_gradient_checkpointing"):
                self.transformer.enable_gradient_checkpointing()
                logger.info("Cosmos transformer gradient checkpointing enabled")

        if freeze is not None:
            freeze_text_encoder = freeze_vae = freeze_transformer = bool(freeze)
        self.freeze_transformer = bool(freeze_transformer)
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
        hook_early_exit: bool = True,
        video_fm_config: VideoFMConfig | None = None,
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
            hook_early_exit=hook_early_exit,
            video_fm_config=video_fm_config,
        )

    def _should_hook_early_exit(self, *, compute_video_loss: bool) -> bool:
        """Skip Cosmos blocks after ``extract_layer`` when only the hook is needed."""
        if not self.hook_early_exit or compute_video_loss:
            return False
        if self.transformer is None or not hasattr(self.transformer, "transformer_blocks"):
            return False
        return True

    def set_hook_early_exit(self, enabled: bool) -> None:
        self.hook_early_exit = bool(enabled)

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

    def _denormalize_latents(self, z: torch.Tensor) -> torch.Tensor:
        if self.latents_mean is None or self.latents_std is None:
            return z
        mean = self.latents_mean.to(device=z.device, dtype=z.dtype)
        std = self.latents_std.to(device=z.device, dtype=z.dtype)
        if mean.ndim == 5 and mean.shape[2] >= z.shape[2]:
            mean = mean[:, :, : z.shape[2]]
        if std.ndim == 5 and std.shape[2] >= z.shape[2]:
            std = std[:, :, : z.shape[2]]
        return z * std + mean

    @torch.no_grad()
    def decode_latents_to_video(self, latents_norm: torch.Tensor) -> torch.Tensor:
        """Decode normalized latents to pixel video in [0, 1], shape (B, T, 3, H, W)."""
        if self.vae is None:
            raise RuntimeError("VAE is required to decode latents.")
        latents_raw = self._denormalize_latents(latents_norm).to(dtype=self.vae.dtype)
        out = self.vae.decode(latents_raw, return_dict=False)[0]
        x = out
        if x.ndim == 5 and x.shape[1] == 3:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        elif x.ndim == 4 and x.shape[1] == 3:
            x = x.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected decoded tensor shape {tuple(x.shape)}")
        x = x.float()
        if x.min().item() < -0.5:
            x = (x + 1.0) / 2.0
        return x.clamp(0.0, 1.0)

    def _match_pixel_num_frames(self, video: torch.Tensor, target_num_frames: int) -> torch.Tensor:
        """Pad or truncate decoded pixels; do not upsample along time (VAE already full resolution)."""
        if target_num_frames <= 0 or video.shape[1] == target_num_frames:
            return video
        if video.shape[1] < target_num_frames:
            pad = video[:, -1:].repeat(1, target_num_frames - video.shape[1], 1, 1, 1)
            return torch.cat([video, pad], dim=1)
        return video[:, :target_num_frames]

    @staticmethod
    def _coerce_video_bcthw(video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"video must be 5D, got {tuple(video.shape)}")
        if video.shape[1] == 3:
            return video
        if video.shape[2] == 3:
            return video.permute(0, 2, 1, 3, 4).contiguous()
        raise ValueError(f"video must be (B,3,T,H,W) or (B,T,3,H,W), got {tuple(video.shape)}")

    def _preprocess_pixels_for_vae(self, video_bcthw: torch.Tensor) -> torch.Tensor:
        """Cosmos Predict pipeline: VideoProcessor expects (B,T,C,H,W), returns (B,C,T,H,W)."""
        video_bcthw = self._coerce_video_bcthw(video_bcthw)
        _, _, _, height, width = video_bcthw.shape
        # VideoProcessor expects [0,1] tensors and internally normalizes to [-1,1].
        video_01 = ((video_bcthw.clamp(-1.0, 1.0) + 1.0) * 0.5).to(dtype=torch.float32)
        video_btchw = video_01.permute(0, 2, 1, 3, 4).contiguous()
        processed = self.video_processor.preprocess_video(
            video_btchw,
            height=int(height),
            width=int(width),
        )
        if processed.shape[1] != 3:
            raise ValueError(f"video_processor must return (B,3,T,H,W), got {tuple(processed.shape)}")
        return processed.to(device=video_bcthw.device, dtype=video_bcthw.dtype)

    @torch.no_grad()
    def prepare_latents(
        self,
        *,
        video: torch.Tensor,
        num_frames_in: int,
        num_frames_out: int,
        generator: torch.Generator | None = None,
        noise_latents: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """DiT4DiT-style latent prep: pixel pad, VAE encode, multi-frame conditioning mask."""
        from diffusers.utils.torch_utils import randn_tensor

        video_bcthw = self._coerce_video_bcthw(video)
        video_bcthw = self._preprocess_pixels_for_vae(video_bcthw)
        b = int(video_bcthw.shape[0])
        c = int(self.latent_channels)
        device = video_bcthw.device
        latent_dtype = torch.float32  # Cosmos Predict pipeline uses fp32 latents in the denoising loop
        temporal_factor = max(1, int(self.vae_scale_factor_temporal))
        t_out_lat = (int(num_frames_out) - 1) // temporal_factor + 1

        t_in_pix = int(video_bcthw.shape[2])
        if t_in_pix < num_frames_out:
            n_pad = int(num_frames_out) - t_in_pix
            zero_pad = torch.zeros(
                (b, 3, n_pad, video_bcthw.shape[3], video_bcthw.shape[4]),
                device=video_bcthw.device,
                dtype=video_bcthw.dtype,
            )
            video_bcthw_padded = torch.cat([video_bcthw, zero_pad], dim=2)
        else:
            video_bcthw_padded = video_bcthw[:, :, : int(num_frames_out)]

        cond_latents = self._encode_video_for_prepare(video_bcthw_padded, generator=generator)
        h_lat = int(cond_latents.shape[-2])
        w_lat = int(cond_latents.shape[-1])
        shape_out = (b, c, t_out_lat, h_lat, w_lat)

        if int(cond_latents.shape[2]) != t_out_lat:
            cond_adjusted = cond_latents.new_zeros(shape_out)
            t_copy = min(int(cond_latents.shape[2]), t_out_lat)
            cond_adjusted[:, :, :t_copy] = cond_latents[:, :, :t_copy]
            cond_latents = cond_adjusted

        if noise_latents is None:
            if generator is not None:
                single_noise = randn_tensor(
                    (1, c, t_out_lat, h_lat, w_lat),
                    generator=generator,
                    device=device,
                    dtype=latent_dtype,
                )
                noise_latents = single_noise.repeat(b, 1, 1, 1, 1)
            else:
                noise_latents = randn_tensor(shape_out, generator=generator, device=device, dtype=latent_dtype)
        else:
            noise_latents = noise_latents.to(device=device, dtype=latent_dtype)
            if tuple(noise_latents.shape) != shape_out:
                raise ValueError(
                    f"noise_latents shape {tuple(noise_latents.shape)} != expected {shape_out}"
                )

        num_cond_latent_frames = min(t_out_lat, (int(num_frames_in) - 1) // temporal_factor + 1)
        cond_indicator = noise_latents.new_zeros(b, 1, t_out_lat, 1, 1)
        cond_indicator[:, :, :num_cond_latent_frames] = 1.0
        ones_padding = noise_latents.new_ones((b, 1, t_out_lat, h_lat, w_lat))
        zeros_padding = noise_latents.new_zeros((b, 1, t_out_lat, h_lat, w_lat))
        cond_mask = cond_indicator * ones_padding + (1.0 - cond_indicator) * zeros_padding
        return noise_latents, cond_latents, cond_mask, cond_indicator

    def resolve_num_pixel_frames_out(
        self,
        *,
        num_frames_in: int,
        num_pixel_frames_out: int | None = None,
        seq_len: int | None = None,
        action_video_freq_ratio: int | None = None,
    ) -> int:
        """Resolve output pixel length (DiT4DiT ``train_num_frames_out`` when configured)."""
        temporal_factor = max(1, int(self.vae_scale_factor_temporal))
        min_frames_out = 1 + temporal_factor
        if num_pixel_frames_out is None:
            num_pixel_frames_out = self.video_fm.num_pixel_frames_out
        if num_pixel_frames_out is None and seq_len is not None and action_video_freq_ratio is not None:
            num_pixel_frames_out = dit4dit_train_num_frames_out(seq_len, action_video_freq_ratio)
        if num_pixel_frames_out is None:
            num_pixel_frames_out = int(num_frames_in)
        return max(int(num_pixel_frames_out), min_frames_out)

    def _get_predict_pipeline(self):
        """Lazy official Cosmos2.5 Predict pipeline sharing tower weights."""
        if self._predict_pipeline is not None:
            return self._predict_pipeline
        if self.transformer is None or self.vae is None:
            raise RuntimeError("Cosmos transformer and VAE are required for video generation.")
        from diffusers import Cosmos2_5_PredictBasePipeline

        pipe = Cosmos2_5_PredictBasePipeline(
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            vae=self.vae,
            scheduler=self.scheduler,
            safety_checker=_DefaultDummySafetyChecker(),
        )
        self._predict_pipeline = pipe.to(self.device)
        return self._predict_pipeline

    @staticmethod
    def _frame_chw_to_pil(frame_chw: torch.Tensor):
        import numpy as np
        from PIL import Image

        arr = (
            frame_chw.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
        ).round().astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def _pipeline_frames_to_btchw(frames) -> torch.Tensor:
        """Convert official pipeline output to (B, T, 3, H, W) float in [0, 1]."""
        if torch.is_tensor(frames):
            video = frames.detach().float()
            if video.ndim == 5 and video.shape[1] == 3:
                video = video.permute(0, 2, 1, 3, 4).contiguous()
            elif video.ndim == 4 and video.shape[1] == 3:
                video = video.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
            elif video.ndim == 5 and video.shape[2] == 3:
                pass
            else:
                raise ValueError(f"Unexpected pipeline tensor shape {tuple(video.shape)}")
            if video.min().item() < -0.5:
                video = (video + 1.0) / 2.0
            return video.clamp(0.0, 1.0)

        import numpy as np

        batch = []
        for sample in frames:
            if torch.is_tensor(sample):
                arr = sample.detach().float().cpu()
                if arr.ndim == 4 and arr.shape[0] == 3:
                    arr = arr.permute(1, 2, 3, 0)
                elif arr.ndim == 3 and arr.shape[-1] != 3:
                    arr = arr.permute(1, 2, 0)
            else:
                arr = np.asarray(sample.convert("RGB"), dtype=np.float32) / 255.0
                batch.append(torch.from_numpy(arr).permute(2, 0, 1))
                continue
            if arr.min().item() < -0.5:
                arr = (arr + 1.0) / 2.0
            batch.append(arr.permute(2, 0, 1) if arr.shape[-1] == 3 else arr)
        if not batch:
            raise ValueError("Pipeline returned no frames.")
        stacked = torch.stack(batch, dim=0).permute(0, 2, 1, 3, 4)
        return stacked.clamp(0.0, 1.0)

    @torch.no_grad()
    def generate_video(
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
    ) -> Tuple[torch.Tensor, None]:
        """Image2World inference via official ``Cosmos2_5_PredictBasePipeline``."""
        del num_frames_in  # I2W always conditions on a single frame
        if self.transformer is None:
            raise RuntimeError("Cosmos transformer is required for video generation.")

        video_bcthw = self._coerce_video_bcthw(video)
        b, _, full_t, height, width = video_bcthw.shape
        frames_out = self.resolve_num_pixel_frames_out(
            num_frames_in=1,
            num_pixel_frames_out=num_pixel_frames_out or (full_t if full_t > 1 else None),
            seq_len=seq_len,
            action_video_freq_ratio=action_video_freq_ratio,
        )
        steps = max(1, int(num_inference_steps or self.video_fm.preview_inference_timesteps))
        guidance_scale = float(self.video_fm.inference_guidance_scale)
        conditional_frame_timestep = float(self.video_fm.inference_conditional_frame_timestep)
        frame_01 = ((video_bcthw.clamp(-1.0, 1.0) + 1.0) * 0.5).to(dtype=torch.float32)

        negative_prompt_embeds = None
        if guidance_scale > 1.0 and self.text_encoder is not None:
            from diffusers.pipelines.cosmos.pipeline_cosmos2_5_predict import DEFAULT_NEGATIVE_PROMPT

            negative_prompt_embeds, _ = self.encode_prompt(DEFAULT_NEGATIVE_PROMPT)

        pipe = self._get_predict_pipeline()
        pred_batches = []
        for bi in range(b):
            pipe_kwargs = dict(
                image=self._frame_chw_to_pil(frame_01[bi, :, 0]),
                video=None,
                prompt_embeds=prompt_embeds[bi : bi + 1],
                negative_prompt_embeds=(
                    negative_prompt_embeds[bi : bi + 1] if negative_prompt_embeds is not None else None
                ),
                height=int(height),
                width=int(width),
                num_frames=int(frames_out),
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pt",
                conditional_frame_timestep=conditional_frame_timestep,
            )
            out = pipe(**pipe_kwargs)
            pred_batches.append(self._pipeline_frames_to_btchw(out.frames))
        pred_video = torch.cat(pred_batches, dim=0).to(device=video_bcthw.device)
        pred_video = self._match_pixel_num_frames(pred_video, frames_out)
        return pred_video, None

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
        compute_video_loss: bool = False,
    ) -> torch.Tensor:
        """Single DiT forward; optionally capture hook hidden at ``extract_layer``."""
        transformer_dtype = self.transformer.dtype
        if enable_hook:
            self._cached_hidden.clear()
            self._capture_hidden_enabled = True

        if self._should_hook_early_exit(compute_video_loss=compute_video_loss):
            forward_transformer_to_hook_layer(
                self.transformer,
                self.extract_layer,
                hidden_states=hidden_states.to(dtype=transformer_dtype),
                timestep=timestep.to(dtype=transformer_dtype),
                encoder_hidden_states=prompt_embeds.to(dtype=transformer_dtype),
                condition_mask=condition_mask.to(dtype=transformer_dtype),
                padding_mask=padding_mask.to(dtype=transformer_dtype),
            )
            if enable_hook:
                self._capture_hidden_enabled = False
            return hidden_states.to(dtype=transformer_dtype)

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
        video_bcthw: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        compute_video_loss: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """DiT4DiT joint forward: frame-0 cond via prepare_latents, full latent T for DiT."""
        if self.transformer is None:
            zero = torch.tensor(0.0, device=video_bcthw.device, dtype=torch.float32)
            empty = torch.zeros(
                (video_bcthw.shape[0], 4, self.action_context_dim),
                device=video_bcthw.device,
                dtype=self.torch_dtype,
            )
            mask = torch.ones((video_bcthw.shape[0], 4), device=video_bcthw.device, dtype=torch.bool)
            return zero, empty, mask

        video_bcthw = self._coerce_video_bcthw(video_bcthw)
        num_frames_out = int(video_bcthw.shape[2])
        n_lcf = int(getattr(self.video_fm, "num_latent_conditional_frames", 2))
        video_for_prepare, num_frames_in = build_video2world_prepare_clip(
            video_bcthw,
            num_frames_out=num_frames_out,
            num_latent_conditional_frames=n_lcf,
        )

        noise_latents, cond_latents, cond_mask, cond_indicator = self.prepare_latents(
            video=video_for_prepare,
            num_frames_in=num_frames_in,
            num_frames_out=num_frames_out,
        )
        cond_count = int(cond_indicator[0, 0, :, 0, 0].sum().item())
        compute_dtype = self.transformer.dtype
        cond_mask_t = cond_mask.to(dtype=compute_dtype)
        cond_timestep = torch.ones_like(cond_indicator, dtype=compute_dtype) * self.conditional_frame_timestep
        padding_mask = self._cosmos_padding_mask(cond_latents)
        zero = torch.tensor(0.0, device=video_bcthw.device, dtype=torch.float32)

        if not compute_video_loss:
            grad_enabled = not self.detach_action_context
            capture_ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
            with capture_ctx:
                in_latents = cond_mask_t * cond_latents.to(dtype=compute_dtype) + (
                    1.0 - cond_mask_t
                ) * noise_latents.to(dtype=compute_dtype)
                in_timestep = cond_indicator.to(dtype=compute_dtype) * cond_timestep
                self._run_cosmos_transformer_with_hook(
                    in_latents,
                    cond_mask_t,
                    in_timestep,
                    prompt_embeds,
                    padding_mask,
                    enable_hook=True,
                    compute_video_loss=False,
                )
            return zero, *self._action_context_from_hook()

        b = int(video_bcthw.shape[0])
        t_lat = int(noise_latents.shape[2])
        if t_lat <= cond_count:
            return zero, *self._action_context_from_hook()

        grad_enabled = not self.detach_action_context and not self.freeze_transformer
        fm_ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
        with fm_ctx:
            with torch.no_grad():
                gt_latents_norm = self._encode_video_to_latents_norm(video_bcthw)

            x0_future = gt_latents_norm[:, :, cond_count:].to(dtype=compute_dtype)
            if x0_future.numel() == 0:
                return zero, *self._action_context_from_hook()

            t_sup = int(x0_future.shape[2])
            fm_cfg = self.video_fm
            t_noise = sample_flow_matching_t(
                b,
                video_bcthw.device,
                torch.float32,
                time_distribution=fm_cfg.flow_time_distribution,
                high_sigma_ratio=fm_cfg.flow_high_sigma_ratio,
                high_sigma_min=fm_cfg.flow_high_sigma_min,
            ).view(b, 1, 1, 1, 1).to(dtype=compute_dtype)

            z_future = torch.randn_like(x0_future)
            xt_future = (1.0 - t_noise) * x0_future + t_noise * z_future

            xt_full = torch.randn_like(noise_latents, dtype=compute_dtype)
            xt_full[:, :, cond_count : cond_count + t_sup] = xt_future

            t_b1t11 = noise_latents.new_zeros(cond_indicator.shape, dtype=compute_dtype)
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
                compute_video_loss=True,
            )

            v_tgt = (z_future - x0_future).to(dtype=v_pred.dtype)
            v_pred_future = v_pred[:, :, cond_count : cond_count + t_sup]
            loss_video = F.mse_loss(v_pred_future.float(), v_tgt.float())

        return loss_video, *self._action_context_from_hook()

    def _cosmos_padding_mask(
        self,
        latents: torch.Tensor,
        *,
        pixel_height: int | None = None,
        pixel_width: int | None = None,
    ) -> torch.Tensor:
        """Cosmos DiT expects [1,1,H,W] pixel-space mask (official pipeline)."""
        _, _, _, h, w = latents.shape
        ph = int(pixel_height) if pixel_height is not None else h * self.vae_scale_factor_spatial
        pw = int(pixel_width) if pixel_width is not None else w * self.vae_scale_factor_spatial
        return latents.new_zeros(1, 1, ph, pw)

    def _capture_action_context(
        self,
        video_bcthw: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hook-only path (inference)."""
        _, action_ctx, action_mask = self._forward_cosmos_joint_unified(
            video_bcthw, prompt_embeds, compute_video_loss=False
        )
        return action_ctx, action_mask

    def _future_latent_flow_matching_loss(
        self,
        video_bcthw: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Legacy alias; prefer ``forward_joint_step`` unified path."""
        loss_video, _, _ = self._forward_cosmos_joint_unified(
            video_bcthw, prompt_embeds, compute_video_loss=True
        )
        return loss_video

    def forward_joint_step(
        self,
        video_bcthw: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        compute_video_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single DiT forward on DiT4DiT-style frame-0 cond + full latent timeline."""
        return self._forward_cosmos_joint_unified(
            video_bcthw, prompt_embeds, compute_video_loss=compute_video_loss
        )

    def _cosmos_transformer_step(
        self,
        video_bcthw: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Legacy alias: joint forward returning (loss_video, _, action_ctx, action_mask)."""
        loss_video, action_ctx, action_mask = self.forward_joint_step(video_bcthw, prompt_embeds)
        return loss_video, loss_video.new_zeros(1), action_ctx, action_mask

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode [B,3,T,H,W] in [-1,1] to normalized latents [B,C,T',H',W'] (VAE mode/mean)."""
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
    def _encode_video_for_prepare(
        self,
        video_bcthw: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """DiT4DiT prepare_latents: VAE encode with sample(generator)."""
        if self.vae is None:
            raise RuntimeError("VAE is required.")
        vb = video_bcthw.to(device=self.device, dtype=self.vae.dtype)
        enc = self.vae.encode(vb)
        if hasattr(enc, "latent_dist"):
            z = enc.latent_dist.sample(generator=generator)
        else:
            z = enc
        return self._normalize_latents(z.to(dtype=torch.float32))

    @torch.no_grad()
    def _encode_video_to_latents_norm(self, video_bcthw: torch.Tensor) -> torch.Tensor:
        """DiT4DiT GT / flow-matching target encode (deterministic mean)."""
        if self.vae is None:
            raise RuntimeError("VAE is required.")
        if self.latents_mean is None or self.latents_std is None:
            raise ValueError("VAE must define latents_mean and latents_std.")
        vb = video_bcthw.to(device=self.device, dtype=self.vae.dtype)
        enc = self.vae.encode(vb)
        if hasattr(enc, "latent_dist"):
            z = enc.latent_dist.mode() if hasattr(enc.latent_dist, "mode") else enc.latent_dist.mean
        else:
            z = enc
        z = z.to(dtype=torch.float32)
        mean = self.latents_mean.to(device=z.device, dtype=z.dtype)
        std = self.latents_std.to(device=z.device, dtype=z.dtype)
        if mean.ndim == 5 and mean.shape[2] >= z.shape[2]:
            mean = mean[:, :, : z.shape[2]]
        if std.ndim == 5 and std.shape[2] >= z.shape[2]:
            std = std[:, :, : z.shape[2]]
        return (z - mean) / std

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
        video_bcthw: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """DiT4DiT joint: future flow loss + hook action context."""
        loss_video, action_ctx, action_mask = self.forward_joint_step(video_bcthw, prompt_embeds)
        return loss_video, loss_video.detach(), action_ctx, action_mask

    def extract_action_context(
        self,
        video_bcthw: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DiT4DiT inference: frame-0 cond hook context (detached when configured)."""
        return self._capture_action_context(video_bcthw, prompt_embeds)

    def build_action_context(
        self,
        video_bcthw: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for ``extract_action_context`` (prompt_embeds = 100352-d Cosmos prompt)."""
        del text_mask
        return self.extract_action_context(video_bcthw, prompt_embeds)

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
        video_bcthw: torch.Tensor,
        prompt_embeds: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del padding_mask
        if self.transformer is None:
            return torch.tensor(0.0, device=video_bcthw.device, dtype=torch.float32)
        loss_video, _, _ = self.forward_joint_step(video_bcthw, prompt_embeds)
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

    def extract_action_context(self, video, prompt_embeds):
        b, device = video.shape[0], video.device
        del prompt_embeds
        return self._smoke_action_context(b, device)

    def build_action_context(self, video, prompt_embeds, text_mask=None):
        del text_mask
        return self.extract_action_context(video, prompt_embeds)

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
