"""Partial Cosmos DiT forward: run blocks through ``extract_layer`` only (hook path)."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F


def _prepare_cosmos_transformer_inputs(
    transformer,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    fps: Optional[int] = None,
    condition_mask: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Mirror diffusers ``CosmosTransformer3DModel.forward`` steps 1–6."""
    batch_size, num_channels, num_frames, height, width = hidden_states.shape

    if condition_mask is not None:
        hidden_states = torch.cat([hidden_states, condition_mask], dim=1)

    if transformer.config.concat_padding_mask:
        if padding_mask is None:
            raise ValueError("padding_mask is required when concat_padding_mask=True")
        padding_mask_resized = padding_mask
        if tuple(padding_mask.shape[-2:]) != tuple(hidden_states.shape[-2:]):
            padding_mask_resized = F.interpolate(
                padding_mask.float(),
                size=list(hidden_states.shape[-2:]),
                mode="nearest",
            ).to(dtype=padding_mask.dtype)
        hidden_states = torch.cat(
            [
                hidden_states,
                padding_mask_resized.unsqueeze(2).repeat(batch_size, 1, num_frames, 1, 1),
            ],
            dim=1,
        )

    if attention_mask is not None:
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

    image_rotary_emb = transformer.rope(hidden_states, fps=fps)
    extra_pos_emb = (
        transformer.learnable_pos_embed(hidden_states)
        if transformer.config.extra_pos_embed_type
        else None
    )

    p_t, p_h, p_w = transformer.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    hidden_states = transformer.patch_embed(hidden_states)
    hidden_states = hidden_states.flatten(1, 3)

    if timestep.ndim == 1:
        temb, embedded_timestep = transformer.time_embed(hidden_states, timestep)
    elif timestep.ndim == 5:
        if timestep.shape != (batch_size, 1, num_frames, 1, 1):
            raise ValueError(
                f"Expected timestep [B,1,T,1,1], got {tuple(timestep.shape)}"
            )
        timestep_flat = timestep.flatten()
        temb, embedded_timestep = transformer.time_embed(hidden_states, timestep_flat)
        temb, embedded_timestep = (
            x.view(batch_size, post_patch_num_frames, 1, 1, -1)
            .expand(-1, -1, post_patch_height, post_patch_width, -1)
            .flatten(1, 3)
            for x in (temb, embedded_timestep)
        )
    else:
        raise ValueError(f"Unsupported timestep shape {tuple(timestep.shape)}")

    text_context, img_context = (
        encoder_hidden_states
        if isinstance(encoder_hidden_states, tuple)
        else (encoder_hidden_states, None)
    )
    if transformer.config.use_crossattn_projection:
        text_context = transformer.crossattn_proj(text_context)

    if img_context is not None and transformer.config.img_context_dim_in:
        img_context = transformer.img_context_proj(img_context)

    processed_encoder_hidden_states = (
        (text_context, img_context) if isinstance(encoder_hidden_states, tuple) else text_context
    )

    return {
        "hidden_states": hidden_states,
        "processed_encoder_hidden_states": processed_encoder_hidden_states,
        "embedded_timestep": embedded_timestep,
        "temb": temb,
        "image_rotary_emb": image_rotary_emb,
        "extra_pos_emb": extra_pos_emb,
        "attention_mask": attention_mask,
        "post_patch_num_frames": post_patch_num_frames,
        "post_patch_height": post_patch_height,
        "post_patch_width": post_patch_width,
        "p_t": p_t,
        "p_h": p_h,
        "p_w": p_w,
    }


def _run_cosmos_transformer_blocks(
    transformer,
    state: dict[str, Any],
    *,
    stop_layer: int | None = None,
    block_controlnet_hidden_states: Optional[list[torch.Tensor]] = None,
) -> torch.Tensor:
    blocks = transformer.transformer_blocks
    hidden = state["hidden_states"]
    processed_encoder_hidden_states = state["processed_encoder_hidden_states"]
    embedded_timestep = state["embedded_timestep"]
    temb = state["temb"]
    image_rotary_emb = state["image_rotary_emb"]
    extra_pos_emb = state["extra_pos_emb"]
    attention_mask = state["attention_mask"]

    controlnet_block_index_map: dict[int, torch.Tensor] = {}
    if block_controlnet_hidden_states is not None:
        n_blocks = len(blocks)
        controlnet_block_index_map = {
            block_idx: block_controlnet_hidden_states[idx]
            for idx, block_idx in enumerate(range(0, n_blocks, transformer.config.controlnet_block_every_n))
        }

    for block_idx, block in enumerate(blocks):
        controlnet_residual = controlnet_block_index_map.get(block_idx)
        if torch.is_grad_enabled() and getattr(transformer, "gradient_checkpointing", False):
            hidden = transformer._gradient_checkpointing_func(
                block,
                hidden,
                processed_encoder_hidden_states,
                embedded_timestep,
                temb,
                image_rotary_emb,
                extra_pos_emb,
                attention_mask,
                controlnet_residual,
            )
        else:
            hidden = block(
                hidden,
                processed_encoder_hidden_states,
                embedded_timestep,
                temb,
                image_rotary_emb,
                extra_pos_emb,
                attention_mask,
                controlnet_residual,
            )
        if stop_layer is not None and block_idx >= stop_layer:
            break
    return hidden


def forward_transformer_to_hook_layer(
    transformer,
    extract_layer: int,
    *,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    fps: Optional[int] = None,
    condition_mask: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    block_controlnet_hidden_states: Optional[list[torch.Tensor]] = None,
) -> torch.Tensor:
    """Run Cosmos DiT blocks ``0..extract_layer`` and skip output norm/proj."""
    if not hasattr(transformer, "transformer_blocks"):
        raise AttributeError("transformer has no transformer_blocks")
    blocks = transformer.transformer_blocks
    stop_layer = int(extract_layer)
    if stop_layer < 0 or stop_layer >= len(blocks):
        raise ValueError(f"extract_layer={stop_layer} out of bounds for {len(blocks)} blocks")

    state = _prepare_cosmos_transformer_inputs(
        transformer,
        hidden_states,
        timestep,
        encoder_hidden_states,
        attention_mask=attention_mask,
        fps=fps,
        condition_mask=condition_mask,
        padding_mask=padding_mask,
    )
    return _run_cosmos_transformer_blocks(
        transformer,
        state,
        stop_layer=stop_layer,
        block_controlnet_hidden_states=block_controlnet_hidden_states,
    )
