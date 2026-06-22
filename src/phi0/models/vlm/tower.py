"""Qwen3-VL tower: frozen VLM backbone → action cross-attention context."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

QWEN3VL_HIDDEN_DIM = 2048
DEFAULT_VLM_VARIANT = "Qwen/Qwen3-VL-2B-Instruct"


class Qwen3VLTower(nn.Module):
    """Psi0-style VLM tower: last-layer hidden states as action context."""

    action_context_dim: int = QWEN3VL_HIDDEN_DIM

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        freeze: bool = True,
        attn_implementation: str = "flash_attention_2",
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.model_path = str(model_path)
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.freeze = bool(freeze)
        self.attn_implementation = str(attn_implementation)

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        load_kwargs = {
            "torch_dtype": torch_dtype,
            "attn_implementation": attn_implementation,
            "local_files_only": local_files_only,
        }
        path = Path(self.model_path)
        if path.is_dir():
            self.vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                **load_kwargs,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                local_files_only=local_files_only,
            )
        else:
            self.vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                **load_kwargs,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                local_files_only=local_files_only,
            )

        self.vlm_model.to(device=self.device)
        if self.freeze:
            self.vlm_model.eval()
            for param in self.vlm_model.parameters():
                param.requires_grad = False

        total = sum(p.numel() for p in self.vlm_model.parameters())
        logger.info(
            "Loaded Qwen3-VL from %s (%d params, freeze=%s)",
            self.model_path,
            total,
            self.freeze,
        )

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_path: str | None = None,
        checkpoints_dir: str | None = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        freeze: bool = True,
        attn_implementation: str = "flash_attention_2",
        local_files_only: bool = False,
    ) -> "Qwen3VLTower":
        resolved = model_path
        if resolved is None:
            if checkpoints_dir is None:
                resolved = DEFAULT_VLM_VARIANT
            else:
                root = Path(checkpoints_dir)
                candidates = list(root.glob("**/config.json"))
                if candidates:
                    resolved = str(candidates[0].parent)
                else:
                    resolved = DEFAULT_VLM_VARIANT
        return cls(
            str(resolved),
            device=device,
            torch_dtype=torch_dtype,
            freeze=freeze,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
        )

    def set_trainable(self, tune: bool) -> None:
        """Toggle VLM fine-tuning (default frozen)."""
        self.freeze = not bool(tune)
        for param in self.vlm_model.parameters():
            param.requires_grad = bool(tune)
        self.vlm_model.train(bool(tune))

    @torch.no_grad()
    def extract_action_context(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return VLM hidden states ``[B,S,D]`` and bool mask ``[B,S]``."""
        input_ids = input_ids.to(device=self.device, non_blocking=True)
        attention_mask = attention_mask.to(device=self.device, non_blocking=True)
        pixel_values = pixel_values.to(device=self.device, non_blocking=True)
        image_grid_thw = image_grid_thw.to(device=self.device, non_blocking=True)
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.to(device=self.device, non_blocking=True)

        amp_enabled = self.device.type == "cuda" and self.torch_dtype in {
            torch.float16,
            torch.bfloat16,
        }
        model_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if mm_token_type_ids is not None:
            model_kwargs["mm_token_type_ids"] = mm_token_type_ids
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.torch_dtype,
            enabled=amp_enabled,
        ):
            output = self.vlm_model(**model_kwargs)
        hidden = output.hidden_states[-1]
        ctx_mask = attention_mask.to(dtype=torch.bool)
        return hidden, ctx_mask


class SmokeVLMTower(nn.Module):
    """CPU smoke tower without HuggingFace weights."""

    action_context_dim: int = QWEN3VL_HIDDEN_DIM

    def __init__(
        self,
        *,
        num_context_tokens: int = 80,
        action_context_dim: int = QWEN3VL_HIDDEN_DIM,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.num_context_tokens = int(num_context_tokens)
        self.action_context_dim = int(action_context_dim)
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.processor = None
        self.freeze = True

    @torch.no_grad()
    def extract_action_context(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = int(input_ids.shape[0])
        seq = int(attention_mask.shape[1]) if attention_mask.ndim == 2 else self.num_context_tokens
        bias = float(pixel_values.float().mean()) if torch.is_tensor(pixel_values) else 0.0
        del image_grid_thw, mm_token_type_ids
        hidden = torch.full(
            (batch, seq, self.action_context_dim),
            bias,
            device=self.device,
            dtype=self.torch_dtype,
        )
        mask = attention_mask.to(device=self.device, dtype=torch.bool)
        return hidden, mask
