"""Qwen3-VL input preprocessing aligned with Psi0 Psi0ModelTransform."""

from __future__ import annotations

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
) -> Image.Image:
    """Apply Psi0 spatial transform on a single frame."""
    return transform(tensor_frame_to_pil(frame))


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


def build_qwenvl_inputs_single(
    processor,
    images: Sequence[Image.Image],
    instruction: str,
) -> Dict[str, torch.Tensor]:
    """Match Psi0 ``Psi0ModelTransform.build_qwenvl_inputs`` (single sample)."""
    _require_qwen_vl_utils()
    instruction = normalize_vlm_instruction(instruction)
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": instruction})
    messages = [[{"role": "user", "content": content}]]
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
    return processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def collate_vlm_inputs(
    items: List[Dict[str, torch.Tensor]],
    *,
    pad_token_id: int,
    model_max_length: int | None = None,
) -> Dict[str, torch.Tensor]:
    """Pad token sequences like Psi0 ``PaddedCollatorForTogether``."""
    if not items:
        raise ValueError("items must be non-empty")
    if len(items) == 1:
        out = {k: v for k, v in items[0].items() if k in {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}}
        if out["input_ids"].ndim == 1:
            out = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in out.items()}
        return out

    input_ids = [
        x["input_ids"].squeeze(0) if x["input_ids"].ndim == 2 else x["input_ids"]
        for x in items
    ]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    if model_max_length is not None:
        input_ids = input_ids[:, : int(model_max_length)]
    attention_mask = input_ids.ne(pad_token_id)

    pixel_values = [x["pixel_values"] for x in items]
    if isinstance(pixel_values[0], torch.Tensor):
        pixel_values = torch.stack(pixel_values)
    else:
        pixel_values = {k: torch.stack([pv[k] for pv in pixel_values]) for k in pixel_values[0]}

    image_grid_thw = torch.stack(
        [x["image_grid_thw"].squeeze(0) for x in items]
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }


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

    per_sample: List[Dict[str, torch.Tensor]] = []
    for imgs, instruction in zip(images, instructions):
        per_sample.append(build_qwenvl_inputs_single(processor, imgs, instruction))

    return collate_vlm_inputs(
        per_sample,
        pad_token_id=int(pad_token_id),
        model_max_length=model_max_length,
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
) -> Dict[str, torch.Tensor]:
    """Resize/crop observation frame(s) Psi0-style, then tokenize with Qwen3-VL processor."""
    if pixel.ndim != 5:
        raise ValueError(f"Expected [B,T,C,H,W], got {tuple(pixel.shape)}")
    idx = int(frame_index)
    if idx < 0:
        idx = pixel.shape[1] + idx
    if idx < 0 or idx >= pixel.shape[1]:
        raise ValueError(
            f"frame_index {frame_index} out of range for T={pixel.shape[1]}"
        )

    if transform is None:
        transform = make_psi0_vlm_image_transform(
            vlm_image_size,
            img_aug=img_aug,
            training=training,
        )
    images: List[List[Image.Image]] = []
    for b in range(pixel.shape[0]):
        pil = preprocess_frame_for_vlm(pixel[b, idx], transform)
        images.append([pil])
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
    )


def build_deploy_vlm_inputs_from_pixels(
    vlm_processor,
    phi0_processor: Any | None,
    pixel: torch.Tensor,
    instructions: Sequence[str],
    *,
    model_max_length: int | None = None,
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
    )
