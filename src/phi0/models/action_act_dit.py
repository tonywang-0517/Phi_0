"""ACT-style action chunk head: direct regression conditioned on Cosmos hook cross-attn."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from fastwam.models.wan22.helpers.gradient import gradient_checkpoint_forward
from fastwam.models.wan22.wan_video_dit import (
    DiTBlock,
    modulate,
    precompute_freqs_cis,
)
from phi0.models.action_cross_attn import cross_attn_target, resolve_action_cross_attn_mode
from phi0.models.action_history import history_to_flow_source
from phi0.models.action_placeholder import FUTURE_PLACEHOLDER_NOISE_STD
from phi0.models.action_proprio import merge_proprio_action_embeddings
from phi0.models.dit4dit_action_encoder import Dit4DiTActionEncoder
from phi0.models.vggt.tower import VGGT_REGISTER_DIM


class ActionACTDiT(nn.Module):
    """Predict an action chunk [B,T,D] from vision/language context (no flow matching)."""

    def __init__(
        self,
        hidden_dim: int,
        raw_action_dim: int,
        ffn_dim: int,
        text_dim: int,
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
        action_token_encoder: str = "linear",
        action_future_horizon: int | None = None,
        future_placeholder_noise_std: float = FUTURE_PLACEHOLDER_NOISE_STD,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.raw_action_dim = raw_action_dim
        self.action_dim = raw_action_dim
        self.text_dim = text_dim
        self.vggt_dim = int(vggt_dim)
        self.max_seq_len = max_seq_len
        self.action_cross_attn_mode = resolve_action_cross_attn_mode(
            action_cross_attn_mode,
            interleave_self_attention=interleave_self_attention,
        )
        self.interleave_self_attention = self.action_cross_attn_mode == "interleave_vlm"
        self.add_pos_embed = bool(add_pos_embed)
        self.proprio_window = int(proprio_window)
        self.action_token_encoder = str(action_token_encoder).strip().lower()
        self.action_future_horizon = (
            int(action_future_horizon) if action_future_horizon is not None else None
        )
        self.future_placeholder_noise_std = float(future_placeholder_noise_std)
        # VLA-Adapter: fixed learnable perturbation on zero future slots (not resampled each step).
        pert_slots = self.action_future_horizon if self.action_future_horizon else max_seq_len
        if self.future_placeholder_noise_std > 0:
            self.future_placeholder_perturbation = nn.Parameter(
                torch.zeros(int(pert_slots), raw_action_dim)
            )
            nn.init.normal_(
                self.future_placeholder_perturbation,
                mean=0.0,
                std=self.future_placeholder_noise_std,
            )
        else:
            self.register_parameter("future_placeholder_perturbation", None)

        self.action_encoder = nn.Linear(raw_action_dim, hidden_dim)
        self.proprio_encoder = (
            nn.Linear(raw_action_dim, hidden_dim) if self.proprio_window > 0 else None
        )
        if self.action_token_encoder == "dit4dit_prefix_query":
            if self.action_future_horizon is None or self.action_future_horizon <= 0:
                raise ValueError(
                    "dit4dit_prefix_query requires positive action_future_horizon."
                )
            self.prefix_encoder = nn.Sequential(
                nn.Linear(raw_action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.query_encoder = Dit4DiTActionEncoder(raw_action_dim, hidden_dim)
        else:
            self.prefix_encoder = None
            self.query_encoder = None
        self.output_proj = nn.Linear(hidden_dim, raw_action_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        # Skip projection when tower latent dim already matches DiT hidden dim.
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
    ) -> "ActionACTDiT":
        cfg = dict(action_dit_config)
        cfg.pop("action_dim", None)
        cfg.pop("freq_dim", None)
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
        cfg.setdefault("action_token_encoder", "linear")
        cfg.setdefault("future_placeholder_noise_std", FUTURE_PLACEHOLDER_NOISE_STD)
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

    @staticmethod
    def _checkpoint_safe_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Clone to drop inference-mode flag; keeps grad_fn for trainable embed outputs."""
        if t is None:
            return None
        return t.clone()

    def _encode_prefix_query_tokens(self, action_tokens: torch.Tensor) -> tuple[torch.Tensor, Dict[str, Any]]:
        if self.action_token_encoder != "dit4dit_prefix_query":
            raise RuntimeError("_encode_prefix_query_tokens requires dit4dit_prefix_query mode.")
        if self.prefix_encoder is None or self.query_encoder is None:
            raise RuntimeError("prefix_encoder/query_encoder are not initialized.")

        batch_size, prefix_len, _ = action_tokens.shape
        future_horizon = int(self.action_future_horizon)
        prefix_emb = self.prefix_encoder(action_tokens)
        query_actions = history_to_flow_source(action_tokens, future_horizon)
        query_timesteps = torch.zeros(batch_size, device=action_tokens.device, dtype=torch.long)
        query_emb = self.query_encoder(query_actions, query_timesteps)
        tokens = torch.cat([prefix_emb, query_emb], dim=1)
        total_len = prefix_len + future_horizon

        if self.add_pos_embed and self.position_embedding is not None:
            pos_ids = torch.arange(total_len, device=tokens.device, dtype=torch.long)
            tokens = tokens + self.position_embedding(pos_ids).unsqueeze(0)

        meta = {
            "batch_size": batch_size,
            "action_seq_len": future_horizon,
            "prefix_seq_len": prefix_len,
            "total_seq_len": total_len,
        }
        return tokens, meta

    def _maybe_noise_future_placeholder(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """Zero future slots + fixed learnable perturbation (VLA-Adapter training path)."""
        if (
            self.future_placeholder_perturbation is None
            or not self.training
            or self.action_token_encoder == "dit4dit_prefix_query"
        ):
            return action_tokens
        seq_len = int(action_tokens.shape[1])
        if seq_len > self.future_placeholder_perturbation.shape[0]:
            raise ValueError(
                f"future seq_len={seq_len} exceeds perturbation buffer "
                f"{self.future_placeholder_perturbation.shape[0]}"
            )
        pert = self.future_placeholder_perturbation[:seq_len]
        return action_tokens + pert.unsqueeze(0).expand(action_tokens.shape[0], -1, -1)

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
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

        batch_size, _, _ = action_tokens.shape

        if self.action_token_encoder == "dit4dit_prefix_query":
            tokens, meta = self._encode_prefix_query_tokens(action_tokens)
        else:
            if self.proprio_encoder is None:
                proprio_tokens = None
            elif proprio_tokens is None and self.proprio_window > 0:
                raise ValueError("proprio_tokens required when proprio_window > 0")
            action_tokens = self._maybe_noise_future_placeholder(action_tokens)
            tokens, meta = merge_proprio_action_embeddings(
                self.proprio_encoder if self.proprio_encoder is not None else self.action_encoder,
                self.action_encoder,
                proprio_tokens,
                action_tokens,
                position_embedding=self.position_embedding,
            )
        total_len = meta["total_seq_len"]

        if self.action_cross_attn_mode == "self_only":
            context_emb = None
            context_attn_mask = None
        else:
            if context_mask is None:
                context_mask = torch.ones(
                    (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
                )
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

        # Zero AdaLN modulation (no diffusion timestep).
        t_mod = torch.zeros(
            batch_size, 6, self.hidden_dim, device=tokens.device, dtype=tokens.dtype
        )
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
            context,
            context_mask,
            context_emb=context_emb,
            vggt_context=vggt_context,
            vggt_context_mask=vggt_context_mask,
            vggt_context_emb=vggt_context_emb,
            proprio_tokens=proprio_tokens,
        )
        x = pre["tokens"]
        ctx = self._checkpoint_safe_tensor(pre["context"])
        vggt_ctx = self._checkpoint_safe_tensor(pre["vggt_context"])
        t_mod = self._checkpoint_safe_tensor(pre["t_mod"])
        freqs = self._checkpoint_safe_tensor(pre["freqs"])
        ctx_mask = self._checkpoint_safe_tensor(pre["context_mask"])
        vggt_mask = self._checkpoint_safe_tensor(pre["vggt_context_mask"])
        for block_idx, block in enumerate(self.blocks):
            if self.use_gradient_checkpointing:
                x = gradient_checkpoint_forward(
                    self._apply_block,
                    self.use_gradient_checkpointing,
                    block_idx,
                    block,
                    x,
                    ctx,
                    vggt_ctx,
                    t_mod,
                    freqs,
                    ctx_mask,
                    vggt_mask,
                )
            else:
                x = self._apply_block(
                    block_idx,
                    block,
                    x,
                    ctx,
                    vggt_ctx,
                    t_mod,
                    freqs,
                    ctx_mask,
                    vggt_mask,
                )
        return self.post_dit(x, pre)
