"""Frozen VGGT-Omega aggregator → scene register tokens for action cross-attn.

Inference-only: ``extract_register_context`` runs under ``@torch.no_grad()`` with
``freeze=True`` (aggregator weights ``requires_grad=False``, ``model.eval()``).
Gradients flow only through the trainable ``vggt_embedding`` linear projection in the action head.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from phi0.models.vggt.preprocess import video_to_vggt_input

logger = logging.getLogger(__name__)

# Cached aggregator output concat frame+inter tokens → 2 * embed_dim.
VGGT_REGISTER_DIM = 2048
VGGT_NUM_REGISTERS = 16


def registers_from_aggregated(
    camera_and_register_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten scene registers ``[B,S,17,D]`` → ctx ``[B,S*16,D]`` + mask."""
    if camera_and_register_tokens.ndim != 4:
        raise ValueError(
            "Expected camera_and_register_tokens [B,S,1+R,D], "
            f"got {tuple(camera_and_register_tokens.shape)}"
        )
    registers = camera_and_register_tokens[:, :, 1:, :]
    b, s, r, d = registers.shape
    ctx = registers.reshape(b, s * r, d)
    mask = torch.ones((b, ctx.shape[1]), dtype=torch.bool, device=ctx.device)
    return ctx, mask


class VGGTOmegaTower(nn.Module):
    """Frozen VGGT-Omega aggregator; exposes scene register tokens only."""

    register_dim: int = VGGT_REGISTER_DIM

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        image_resolution: int = 512,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.image_resolution = int(image_resolution)
        self.freeze = bool(freeze)

        from vggt_omega.models import VGGTOmega

        self.model = VGGTOmega(
            enable_camera=False,
            enable_depth=False,
            enable_alignment=False,
        ).to(device=self.device, dtype=self.torch_dtype)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected:
            logger.debug("VGGT load ignored unexpected keys: %d", len(unexpected))
        if missing:
            logger.warning("VGGT load missing keys: %d (heads disabled — expected)", len(missing))
        self.model.eval()
        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def extract_register_context(
        self,
        video: torch.Tensor,
        *,
        precomputed_registers: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return register ctx ``[B,S*16,2048]`` and bool mask from stereo clip."""
        if precomputed_registers is not None:
            if precomputed_registers.ndim == 4:
                return registers_from_aggregated(precomputed_registers)
            if precomputed_registers.ndim == 3:
                b, n, d = precomputed_registers.shape
                mask = torch.ones((b, n), dtype=torch.bool, device=precomputed_registers.device)
                return precomputed_registers, mask
            raise ValueError(
                f"precomputed_registers must be [B,S,R,D] or [B,N,D], got {tuple(precomputed_registers.shape)}"
            )

        images = video_to_vggt_input(video, image_resolution=self.image_resolution)
        if images.device != self.device:
            images = images.to(device=self.device, non_blocking=True)
        images = images.to(dtype=self.torch_dtype)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=self.device.type == "cuda"):
            aggregated_tokens_list, patch_token_start = self.model.aggregator(images)
        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise RuntimeError("VGGT aggregator did not return final cached layer.")
        camera_and_register = final_tokens[:, :, :patch_token_start].contiguous()
        return registers_from_aggregated(camera_and_register)


class VGGTSmokeTower(nn.Module):
    """Deterministic fake registers for unit tests (no checkpoint)."""

    register_dim: int = VGGT_REGISTER_DIM

    def __init__(
        self,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        num_frames: int = 8,
        image_resolution: int = 512,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.num_frames = int(num_frames)
        self.image_resolution = int(image_resolution)

    @torch.no_grad()
    def extract_register_context(
        self,
        video: torch.Tensor,
        *,
        precomputed_registers: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del precomputed_registers
        b = int(video.shape[0])
        t = int(video.shape[2]) if video.ndim == 5 else self.num_frames
        n = t * VGGT_NUM_REGISTERS
        ctx = torch.zeros(
            (b, n, self.register_dim),
            device=video.device,
            dtype=self.torch_dtype,
        )
        mask = torch.ones((b, n), dtype=torch.bool, device=video.device)
        return ctx, mask
