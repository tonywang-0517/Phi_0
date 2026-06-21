"""DiT4DiT-style flow-matching Action DiT (cross-attn to Cosmos hook + interleaved self-attn)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from fastwam.models.wan22.helpers.gradient import gradient_checkpoint_forward
from fastwam.models.wan22.wan_video_dit import (
    DiTBlock,
    modulate,
    precompute_freqs_cis,
    sinusoidal_embedding_1d,
)
from phi0.models.action_cross_attn import cross_attn_target, resolve_action_cross_attn_mode
from phi0.models.dit4dit_action_encoder import Dit4DiTActionEncoder
from phi0.models.vggt.tower import VGGT_REGISTER_DIM


class ActionFMDiT(nn.Module):
    """Predict FM velocity on an action chunk; conditions on Cosmos hook via cross-attn."""

    def __init__(
        self,
        hidden_dim: int,
        raw_action_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        max_seq_len: int = 1024,
        use_gradient_checkpointing: bool = False,
        interleave_self_attention: bool = True,
        action_cross_attn_mode: Optional[str] = None,
        vggt_dim: int = VGGT_REGISTER_DIM,
        add_pos_embed: bool = True,
        proprio_window: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.raw_action_dim = raw_action_dim
        self.action_dim = raw_action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.vggt_dim = int(vggt_dim)
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.max_seq_len = max_seq_len
        self.action_cross_attn_mode = resolve_action_cross_attn_mode(
            action_cross_attn_mode,
            interleave_self_attention=interleave_self_attention,
        )
        self.interleave_self_attention = self.action_cross_attn_mode == "interleave_vlm"
        self.add_pos_embed = bool(add_pos_embed)
        self.proprio_window = int(proprio_window)

        self.flow_action_encoder = Dit4DiTActionEncoder(raw_action_dim, hidden_dim)
        self.proprio_encoder = (
            nn.Linear(raw_action_dim, hidden_dim) if self.proprio_window > 0 else None
        )
        self.output_proj = nn.Linear(hidden_dim, raw_action_dim)
        self.text_embedding = (
            None if int(text_dim) == int(hidden_dim) else nn.Linear(text_dim, hidden_dim, bias=True)
        )
        if self.action_cross_attn_mode == "dual_vlm_vggt":
            self.vggt_embedding = (
                None
                if int(self.vggt_dim) == int(hidden_dim)
                else nn.Linear(self.vggt_dim, hidden_dim, bias=True)
            )
        else:
            self.vggt_embedding = None
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        if self.add_pos_embed:
            self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        else:
            self.position_embedding = None

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.freqs = precompute_freqs_cis(attn_head_dim, end=max_seq_len)
        self.use_gradient_checkpointing = use_gradient_checkpointing

    @classmethod
    def from_action_dit_config(
        cls,
        action_dit_config: dict[str, Any],
        raw_action_dim: int,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionFMDiT":
        cfg = dict(action_dit_config)
        cfg.pop("action_dim", None)
        for legacy_key in (
            "use_action_query",
            "num_action_queries",
            "num_query_adapter_layers",
            "num_query_refine_blocks",
            "query_adapter_heads",
        ):
            cfg.pop(legacy_key, None)
        cfg.setdefault("interleave_self_attention", True)
        cfg.setdefault("add_pos_embed", True)
        cfg.setdefault("proprio_window", 0)
        return cls(raw_action_dim=raw_action_dim, **cfg).to(device=device, dtype=torch_dtype)

    def _cross_attn_target(self, block_idx: int) -> Optional[str]:
        return cross_attn_target(self.action_cross_attn_mode, block_idx)

    def _embed_vggt_context(self, vggt_context: torch.Tensor) -> torch.Tensor:
        if self.action_cross_attn_mode != "dual_vlm_vggt":
            raise RuntimeError("_embed_vggt_context requires dual_vlm_vggt mode.")
        if self.vggt_embedding is None:
            return vggt_context
        return self.vggt_embedding(vggt_context)

    def _embed_vlm_context(self, context: torch.Tensor) -> torch.Tensor:
        if self.text_embedding is None:
            return context
        return self.text_embedding(context)

    def _encode_fm_action_tokens(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        proprio_tokens: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """DiT4DiT-style: flow-t on noisy actions; proprio prefix without pos embed."""
        batch_size, action_seq_len, _ = action_tokens.shape
        t_disc = timestep.long()
        action_emb = self.flow_action_encoder(action_tokens, t_disc)
        if self.add_pos_embed and self.position_embedding is not None:
            pos_ids = torch.arange(action_seq_len, device=action_emb.device, dtype=torch.long)
            action_emb = action_emb + self.position_embedding(pos_ids).unsqueeze(0)

        proprio_len = 0
        if proprio_tokens is not None and proprio_tokens.shape[1] > 0:
            if self.proprio_encoder is None:
                raise ValueError("proprio_tokens provided but proprio_window is 0")
            proprio_emb = self.proprio_encoder(proprio_tokens)
            tokens = torch.cat([proprio_emb, action_emb], dim=1)
            proprio_len = int(proprio_tokens.shape[1])
        else:
            tokens = action_emb

        meta = {
            "batch_size": batch_size,
            "action_seq_len": action_seq_len,
            "proprio_len": proprio_len,
            "total_seq_len": tokens.shape[1],
        }
        return tokens, meta

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        *,
        context_emb: Optional[torch.Tensor] = None,
        vggt_context: Optional[torch.Tensor] = None,
        vggt_context_mask: Optional[torch.Tensor] = None,
        vggt_context_emb: Optional[torch.Tensor] = None,
        proprio_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if action_tokens.ndim != 3:
            raise ValueError(f"`action_tokens` must be [B,T,D], got {tuple(action_tokens.shape)}")
        if action_tokens.shape[2] != self.raw_action_dim:
            raise ValueError(
                f"Expected raw_action_dim={self.raw_action_dim}, got {action_tokens.shape[2]}"
            )

        batch_size, seq_len, _ = action_tokens.shape
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be [B], got {tuple(timestep.shape)}")
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(f"`timestep` length must be 1 or batch_size={batch_size}")
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, FM timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)

        if context_mask is None and self.action_cross_attn_mode != "self_only":
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )
        if self.proprio_encoder is None:
            proprio_tokens = None
        elif proprio_tokens is None and self.proprio_window > 0:
            raise ValueError("proprio_tokens required when proprio_window > 0")

        tokens, meta = self._encode_fm_action_tokens(action_tokens, timestep, proprio_tokens)
        total_len = meta["total_seq_len"]

        if self.action_cross_attn_mode == "self_only":
            context_emb = None
            context_attn_mask = None
        else:
            context_attn_mask = context_mask.unsqueeze(1).expand(-1, total_len, -1)
            if context_emb is None:
                context_emb = self._embed_vlm_context(context)

        vggt_emb = vggt_context_emb
        vggt_attn_mask = None
        if self.action_cross_attn_mode == "dual_vlm_vggt":
            if vggt_context is None and vggt_emb is None:
                raise ValueError("dual_vlm_vggt requires vggt_context or vggt_context_emb.")
            if vggt_emb is None:
                vggt_emb = self._embed_vggt_context(vggt_context)
            if vggt_context_mask is None:
                vggt_context_mask = torch.ones(
                    (batch_size, vggt_emb.shape[1]), dtype=torch.bool, device=vggt_emb.device
                )
            vggt_attn_mask = vggt_context_mask.unsqueeze(1).expand(-1, total_len, -1)

        t_sin = sinusoidal_embedding_1d(self.freq_dim, timestep.float())
        t_emb = self.time_embedding(t_sin.to(dtype=tokens.dtype))
        t_mod = self.time_projection(t_emb).unflatten(1, (6, self.hidden_dim))
        freqs = self.freqs[:total_len].view(total_len, 1, -1).to(tokens.device)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "vggt_context": vggt_emb,
            "vggt_context_mask": vggt_attn_mask,
            "meta": meta,
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        action_seq_len = int(pre_state["meta"]["action_seq_len"])
        return self.output_proj(tokens[:, -action_seq_len:])

    def _apply_block(
        self,
        block_idx: int,
        block: DiTBlock,
        x: torch.Tensor,
        context: torch.Tensor,
        vggt_context: Optional[torch.Tensor],
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        vggt_context_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        ctx_mask = context_mask
        if ctx_mask is not None and ctx_mask.dim() == 3:
            ctx_mask = ctx_mask.unsqueeze(1)
        vggt_mask = vggt_context_mask
        if vggt_mask is not None and vggt_mask.dim() == 3:
            vggt_mask = vggt_mask.unsqueeze(1)

        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )

        input_x = modulate(block.norm1(x), shift_msa, scale_msa)
        attn_out = block.self_attn(input_x, freqs)
        x = block.gate(x, gate_msa, attn_out)

        target = self._cross_attn_target(block_idx)
        if target == "vlm":
            x = x + block.cross_attn(block.norm3(x), context, ctx_mask=ctx_mask)
        elif target == "vggt":
            if vggt_context is None:
                raise RuntimeError("VGGT cross-attn requested but vggt_context is missing.")
            x = x + block.cross_attn(block.norm3(x), vggt_context, ctx_mask=vggt_mask)

        input_x = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(input_x))
        return x

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        *,
        context_emb: Optional[torch.Tensor] = None,
        vggt_context: Optional[torch.Tensor] = None,
        vggt_context_mask: Optional[torch.Tensor] = None,
        vggt_context_emb: Optional[torch.Tensor] = None,
        proprio_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre = self.pre_dit(
            action_tokens,
            timestep,
            context,
            context_mask,
            context_emb=context_emb,
            vggt_context=vggt_context,
            vggt_context_mask=vggt_context_mask,
            vggt_context_emb=vggt_context_emb,
            proprio_tokens=proprio_tokens,
        )
        x = pre["tokens"]
        for block_idx, block in enumerate(self.blocks):
            if self.use_gradient_checkpointing:
                x = gradient_checkpoint_forward(
                    self._apply_block,
                    self.use_gradient_checkpointing,
                    block_idx,
                    block,
                    x,
                    pre["context"],
                    pre["vggt_context"],
                    pre["t_mod"],
                    pre["freqs"],
                    pre["context_mask"],
                    pre["vggt_context_mask"],
                )
            else:
                x = self._apply_block(
                    block_idx,
                    block,
                    x,
                    pre["context"],
                    pre["vggt_context"],
                    pre["t_mod"],
                    pre["freqs"],
                    pre["context_mask"],
                    pre["vggt_context_mask"],
                )
        return self.post_dit(x, pre)
