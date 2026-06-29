"""Qwen3-VL tower: frozen VLM backbone → action cross-attention context + optional AR text."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

QWEN3VL_HIDDEN_DIM = 2048
DEFAULT_VLM_VARIANT = "Qwen/Qwen3-VL-2B-Instruct"
OFFICIAL_QWEN3VL_INSTRUCT = DEFAULT_VLM_VARIANT


@dataclass(frozen=True)
class GenerateTextConfig:
    """HF ``generate`` knobs for agent speech (Qwen3-VL official path)."""

    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    suppress_mm_tokens: bool = True


def _text_only_suppress_token_ids(vlm_model, *, first_special: int = 151644) -> list[int]:
    """Block vision/chat special tokens so AR emits plain text (Psi0 HE ckpt otherwise emits MM ids)."""
    vocab = int(getattr(vlm_model, "lm_head").out_features)
    return list(range(int(first_special), vocab))


def _vlm_forward_kwargs(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    mm_token_type_ids: torch.Tensor | None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
    if mm_token_type_ids is not None:
        out["mm_token_type_ids"] = mm_token_type_ids
    return out


def decode_generated_text(
    processor,
    input_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    *,
    skip_special_tokens: bool = True,
) -> List[str]:
    """Trim prompt tokens and batch-decode (Qwen3-VL web_demo / README pattern)."""
    prompt_len = int(input_ids.shape[1])
    new_tokens = generated_ids[:, prompt_len:]
    decode_fn = getattr(processor, "batch_decode", None)
    if decode_fn is not None:
        return decode_fn(
            new_tokens,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
    tok = getattr(processor, "tokenizer", processor)
    return tok.batch_decode(new_tokens, skip_special_tokens=skip_special_tokens)


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

    @torch.no_grad()
    def generate_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor | None = None,
        *,
        gen_cfg: GenerateTextConfig | None = None,
        **generate_kwargs: Any,
    ) -> List[str]:
        """Eval-only AR decode via HF ``generate``; not used in action train/infer."""
        gen_cfg = gen_cfg or GenerateTextConfig()
        input_ids = input_ids.to(device=self.device, non_blocking=True)
        attention_mask = attention_mask.to(device=self.device, non_blocking=True)
        pixel_values = pixel_values.to(device=self.device, non_blocking=True)
        image_grid_thw = image_grid_thw.to(device=self.device, non_blocking=True)
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.to(device=self.device, non_blocking=True)

        model_kwargs = _vlm_forward_kwargs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        hf_gen: Dict[str, Any] = {
            "max_new_tokens": int(gen_cfg.max_new_tokens),
            "do_sample": bool(gen_cfg.do_sample),
            "repetition_penalty": float(gen_cfg.repetition_penalty),
        }
        if gen_cfg.do_sample:
            hf_gen["temperature"] = float(gen_cfg.temperature)
            hf_gen["top_p"] = float(gen_cfg.top_p)
        hf_gen.update(generate_kwargs)
        if bool(gen_cfg.suppress_mm_tokens) and "suppress_tokens" not in hf_gen:
            hf_gen["suppress_tokens"] = _text_only_suppress_token_ids(self.vlm_model)

        amp_enabled = self.device.type == "cuda" and self.torch_dtype in {
            torch.float16,
            torch.bfloat16,
        }
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.torch_dtype,
            enabled=amp_enabled,
        ):
            generated_ids = self.vlm_model.generate(**model_kwargs, **hf_gen)
        return decode_generated_text(
            self.processor,
            input_ids,
            generated_ids,
        )

    def generate_text_from_vlm_batch(
        self,
        vlm_inputs: Dict[str, torch.Tensor],
        *,
        gen_cfg: GenerateTextConfig | None = None,
        **generate_kwargs: Any,
    ) -> List[str]:
        """Decode from a processor batch (same tensors as ``extract_action_context``)."""
        return self.generate_text(
            vlm_inputs["input_ids"],
            vlm_inputs["attention_mask"],
            vlm_inputs["pixel_values"],
            vlm_inputs["image_grid_thw"],
            vlm_inputs.get("mm_token_type_ids"),
            gen_cfg=gen_cfg,
            **generate_kwargs,
        )


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

    @torch.no_grad()
    def generate_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor | None = None,
        *,
        gen_cfg: GenerateTextConfig | None = None,
        **generate_kwargs: Any,
    ) -> List[str]:
        del gen_cfg, generate_kwargs, mm_token_type_ids
        batch = int(input_ids.shape[0])
        return [f"smoke_vlm_reply_{i}" for i in range(batch)]

    def generate_text_from_vlm_batch(
        self,
        vlm_inputs: Dict[str, torch.Tensor],
        *,
        gen_cfg: GenerateTextConfig | None = None,
        **generate_kwargs: Any,
    ) -> List[str]:
        return self.generate_text(
            vlm_inputs["input_ids"],
            vlm_inputs["attention_mask"],
            vlm_inputs["pixel_values"],
            vlm_inputs["image_grid_thw"],
            vlm_inputs.get("mm_token_type_ids"),
            gen_cfg=gen_cfg,
            **generate_kwargs,
        )


def load_agent_speech_tower(
    model_path: str,
    *,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
    local_files_only: bool = False,
) -> Qwen3VLTower:
    """Eval-only tower for agent AR; action path keeps ``vlm.model_path`` (Psi0)."""
    return Qwen3VLTower.from_pretrained(
        model_path=model_path,
        device=device,
        torch_dtype=torch_dtype,
        freeze=True,
        attn_implementation=attn_implementation,
        local_files_only=local_files_only,
    )
