"""Qwen3-VL input preprocessing aligned with Psi0 Psi0ModelTransform."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:  # pragma: no cover
    process_vision_info = None  # type: ignore[assignment]
    _QWEN_VL_UTILS_ERROR = exc
else:
    _QWEN_VL_UTILS_ERROR = None

# Psi0 `augmentation.ColorJitter` defaults (finetune-simple-psi0.sh uses img_aug).
PSI0_COLOR_JITTER = {
    "brightness": 0.2,
    "contrast": (0.8, 1.2),
    "saturation": (0.8, 1.2),
    "hue": 0.05,
}


def _require_qwen_vl_utils():
    if process_vision_info is None:
        raise ImportError(
            "qwen_vl_utils is required for Qwen3-VL; install with `pip install qwen-vl-utils`."
        ) from _QWEN_VL_UTILS_ERROR


def _v2():
    try:
        from torchvision.transforms import v2
    except ImportError:  # pragma: no cover
        from torchvision import transforms as v2  # type: ignore
    return v2


def make_psi0_vlm_image_transform(
    size: Tuple[int, int],
    *,
    img_aug: bool = False,
    training: bool = True,
):
    """Psi0 ``Psi0ModelTransform``: Resize(NEAREST) → CenterCrop → optional ColorJitter."""
    v2 = _v2()
    h, w = int(size[0]), int(size[1])
    ops = [
        v2.Resize((h, w), interpolation=v2.InterpolationMode.NEAREST),
        v2.CenterCrop((h, w)),
    ]
    if img_aug and training:
        ops.append(v2.ColorJitter(**PSI0_COLOR_JITTER))
    return v2.Compose(ops)


def tensor_frame_to_pil(frame: torch.Tensor) -> Image.Image:
    """Convert ``[C,H,W]`` float ``[0,1]`` to RGB PIL (Psi0 repacker output range)."""
    if frame.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(frame.shape)}")
    arr = (frame.detach().float().clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def preprocess_frame_for_vlm(
    frame: torch.Tensor,
    transform,
    *,
    vlm_image_size: Tuple[int, int] | None = None,
    img_aug: bool = False,
) -> Image.Image:
    """Apply Psi0 spatial transform on a single frame (always Resize NEAREST + CenterCrop)."""
    del vlm_image_size, img_aug  # kept for call-site compatibility
    return transform(tensor_frame_to_pil(frame))


@dataclass(frozen=True)
class _UniformInstructionKey:
    instruction: str
    num_views: int


# ponytail: process-global; single-task datasets reuse chat template + token layout.
_UNIFORM_CHAT_TEXT: dict[tuple[int, _UniformInstructionKey], str] = {}


def _chat_text_cache_key(processor, instruction: str, num_views: int) -> tuple[int, _UniformInstructionKey]:
    return (id(processor), _UniformInstructionKey(instruction, int(num_views)))


def get_cached_chat_text(processor, instruction: str, num_views: int) -> str | None:
    return _UNIFORM_CHAT_TEXT.get(_chat_text_cache_key(processor, instruction, num_views))


def _vlm_messages(
    images: Sequence[Image.Image],
    instruction: str,
) -> list[list[dict[str, Any]]]:
    content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": instruction})
    return [[{"role": "user", "content": content}]]


def build_qwenvl_inputs_single(
    processor,
    images: Sequence[Image.Image],
    instruction: str,
    *,
    chat_text: str | None = None,
) -> Dict[str, torch.Tensor]:
    """Match Psi0 ``Psi0ModelTransform.build_qwenvl_inputs`` (single sample)."""
    _require_qwen_vl_utils()
    instruction = normalize_vlm_instruction(instruction)
    messages = _vlm_messages(images, instruction)
    if chat_text is None:
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
    else:
        texts = [chat_text]
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
    return processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def _processor_tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def _processor_pad_token_id(processor) -> int:
    tok = _processor_tokenizer(processor)
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tok, "eos_token_id", 0)
    return int(pad_id)


def normalize_vlm_instruction(instruction: str) -> str:
    """Psi0 repack: ``instruction.lower()`` before Qwen chat template."""
    return str(instruction).strip().lower()


def _token_rows_identical(items: List[Dict[str, torch.Tensor]]) -> bool:
    ids0 = items[0]["input_ids"]
    if ids0.ndim == 2:
        ids0 = ids0.squeeze(0)
    for x in items[1:]:
        ids = x["input_ids"]
        if ids.ndim == 2:
            ids = ids.squeeze(0)
        if not torch.equal(ids0, ids):
            return False
    return True


def collate_vlm_inputs(
    items: List[Dict[str, torch.Tensor]],
    *,
    pad_token_id: int,
    model_max_length: int | None = None,
    uniform_tokens: bool = False,
) -> Dict[str, torch.Tensor]:
    """Pad token sequences like Psi0 ``PaddedCollatorForTogether``."""
    if not items:
        raise ValueError("items must be non-empty")
    keep_keys = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "mm_token_type_ids",
    }
    if len(items) == 1:
        out = {k: v for k, v in items[0].items() if k in keep_keys}
        if out["input_ids"].ndim == 1:
            out = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in out.items()}
        return out

    batch_size = len(items)
    if uniform_tokens and _token_rows_identical(items):
        ids = items[0]["input_ids"]
        if ids.ndim == 2:
            ids = ids.squeeze(0)
        if model_max_length is not None:
            ids = ids[: int(model_max_length)]
        input_ids = ids.unsqueeze(0).expand(batch_size, -1)
        attn = items[0]["attention_mask"]
        if attn.ndim == 2:
            attn = attn.squeeze(0)
        if model_max_length is not None:
            attn = attn[: int(model_max_length)]
        attention_mask = attn.unsqueeze(0).expand(batch_size, -1)
        mm_token_type_ids = None
        if items[0].get("mm_token_type_ids") is not None:
            mm = items[0]["mm_token_type_ids"]
            if mm.ndim == 2:
                mm = mm.squeeze(0)
            if model_max_length is not None:
                mm = mm[: int(model_max_length)]
            mm_token_type_ids = mm.unsqueeze(0).expand(batch_size, -1)
    else:
        input_ids = [
            x["input_ids"].squeeze(0) if x["input_ids"].ndim == 2 else x["input_ids"]
            for x in items
        ]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        if model_max_length is not None:
            input_ids = input_ids[:, : int(model_max_length)]
        attention_mask = input_ids.ne(pad_token_id)

        mm_token_type_ids = None
        if items[0].get("mm_token_type_ids") is not None:
            mm_rows = [
                x["mm_token_type_ids"].squeeze(0)
                if x["mm_token_type_ids"].ndim == 2
                else x["mm_token_type_ids"]
                for x in items
            ]
            mm_token_type_ids = pad_sequence(mm_rows, batch_first=True, padding_value=0)
            if model_max_length is not None:
                mm_token_type_ids = mm_token_type_ids[:, : int(model_max_length)]

    pixel_values = [x["pixel_values"] for x in items]
    if isinstance(pixel_values[0], torch.Tensor):
        pixel_values = torch.stack(pixel_values)
    else:
        pixel_values = {k: torch.stack([pv[k] for pv in pixel_values]) for k in pixel_values[0]}

    grid_rows = []
    for x in items:
        grid = x["image_grid_thw"]
        if grid.ndim == 3 and grid.shape[0] == 1:
            grid = grid.squeeze(0)
        grid_rows.append(grid.reshape(-1, 3))
    image_grid_thw = torch.cat(grid_rows, dim=0)
    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
    if mm_token_type_ids is not None:
        out["mm_token_type_ids"] = mm_token_type_ids
    return out


def build_vlm_chat_inputs(
    processor,
    images: Sequence[Sequence[Image.Image]],
    instructions: Sequence[str],
    *,
    model_max_length: int | None = None,
) -> Dict[str, torch.Tensor]:
    """Build batched Qwen3-VL tensors (Psi0 finetune path, user message only)."""
    if len(images) != len(instructions):
        raise ValueError("images and instructions must have the same batch size")

    pad_token_id = _processor_pad_token_id(processor)
    normed = [normalize_vlm_instruction(i) for i in instructions]
    uniform = len(set(normed)) == 1
    chat_text: str | None = None
    if uniform and len(images) > 0:
        key = _chat_text_cache_key(processor, normed[0], len(images[0]))
        chat_text = _UNIFORM_CHAT_TEXT.get(key)
        if chat_text is None:
            chat_text = processor.apply_chat_template(
                _vlm_messages(images[0], normed[0])[0],
                tokenize=False,
                add_generation_prompt=True,
            )
            _UNIFORM_CHAT_TEXT[key] = chat_text

    per_sample: List[Dict[str, torch.Tensor]] = []
    for imgs, instruction in zip(images, instructions):
        per_sample.append(
            build_qwenvl_inputs_single(
                processor,
                imgs,
                instruction,
                chat_text=chat_text if uniform else None,
            )
        )

    return collate_vlm_inputs(
        per_sample,
        pad_token_id=int(pad_token_id),
        model_max_length=model_max_length,
        uniform_tokens=uniform,
    )


def build_vlm_inputs_from_pixel_batch(
    processor,
    pixel: torch.Tensor,
    instructions: Sequence[str],
    *,
    vlm_image_size: Tuple[int, int],
    frame_index: int = 0,
    img_aug: bool = False,
    training: bool = True,
    model_max_length: int | None = None,
    transform=None,
    wrist_pixel: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Resize/crop observation frame(s) Psi0-style, then tokenize with Qwen3-VL processor."""
    if pixel.ndim == 6:
        if pixel.shape[1] < 1:
            raise ValueError(f"Expected at least one camera in pixel batch, got {tuple(pixel.shape)}")
        if wrist_pixel is None and pixel.shape[1] >= 2:
            wrist_pixel = pixel[:, 1]
        pixel = pixel[:, 0]
    if pixel.ndim != 5:
        raise ValueError(f"Expected [B,T,C,H,W], got {tuple(pixel.shape)}")
    idx = int(frame_index)
    if idx < 0:
        idx = pixel.shape[1] + idx
    if idx < 0 or idx >= pixel.shape[1]:
        raise ValueError(
            f"frame_index {frame_index} out of range for T={pixel.shape[1]}"
        )
    if wrist_pixel is not None:
        if wrist_pixel.ndim != 5:
            raise ValueError(f"Expected wrist_pixel [B,T,C,H,W], got {tuple(wrist_pixel.shape)}")
        if wrist_pixel.shape[0] != pixel.shape[0] or wrist_pixel.shape[1] != pixel.shape[1]:
            raise ValueError(
                "wrist_pixel batch/time dims must match pixel: "
                f"{tuple(wrist_pixel.shape)} vs {tuple(pixel.shape)}"
            )

    if transform is None:
        transform = make_psi0_vlm_image_transform(
            vlm_image_size,
            img_aug=img_aug,
            training=training,
        )
    images: List[List[Image.Image]] = []
    for b in range(pixel.shape[0]):
        view_pils = [
            preprocess_frame_for_vlm(
                pixel[b, idx],
                transform,
                vlm_image_size=vlm_image_size,
                img_aug=img_aug,
            )
        ]
        if wrist_pixel is not None:
            view_pils.append(
                preprocess_frame_for_vlm(
                    wrist_pixel[b, idx],
                    transform,
                    vlm_image_size=vlm_image_size,
                    img_aug=img_aug,
                )
            )
        images.append(view_pils)
    return build_vlm_chat_inputs(
        processor,
        images,
        instructions,
        model_max_length=model_max_length,
    )


def video_bcthw_to_pixel_batch(video: torch.Tensor) -> torch.Tensor:
    """``[B,3,T,H,W]`` in ``[-1,1]`` -> ``[B,1,3,H,W]`` in ``[0,1]`` (last frame)."""
    if video.ndim != 5:
        raise ValueError(f"Expected [B,3,T,H,W], got {tuple(video.shape)}")
    frame = (video[:, :, -1] + 1.0) * 0.5
    return frame.unsqueeze(1)


def build_training_vlm_inputs_from_pixels(
    vlm_processor,
    phi0_processor: Any,
    pixel: torch.Tensor,
    instructions: Sequence[str],
    *,
    model_max_length: int | None = None,
    wrist_pixel: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Training-path VLM tokenization (respects processor aug + train/eval mode)."""
    return build_vlm_inputs_from_pixel_batch(
        vlm_processor,
        pixel,
        instructions,
        vlm_image_size=phi0_processor.vlm_image_size,
        frame_index=0,
        img_aug=phi0_processor.vlm_img_aug,
        training=phi0_processor._is_train,
        model_max_length=model_max_length,
        transform=phi0_processor.vlm_image_transform(),
        wrist_pixel=wrist_pixel,
    )


def build_deploy_vlm_inputs_from_pixels(
    vlm_processor,
    phi0_processor: Any | None,
    pixel: torch.Tensor,
    instructions: Sequence[str],
    *,
    model_max_length: int | None = None,
    wrist_pixel: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Deterministic deploy VLM inputs aligned with training resize/crop (no ColorJitter)."""
    vlm_size = getattr(phi0_processor, "vlm_image_size", (180, 320))
    if phi0_processor is not None:
        transform = phi0_processor.vlm_image_transform()
    else:
        transform = make_psi0_vlm_image_transform(vlm_size, img_aug=False, training=False)
    return build_vlm_inputs_from_pixel_batch(
        vlm_processor,
        pixel,
        instructions,
        vlm_image_size=vlm_size,
        frame_index=0,
        img_aug=False,
        training=False,
        model_max_length=model_max_length,
        transform=transform,
        wrist_pixel=wrist_pixel,
    )
