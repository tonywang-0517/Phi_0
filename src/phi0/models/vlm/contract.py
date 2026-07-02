"""VLM input contract aligned with Psi0 Qwen3-VL pretrain / finetune."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import torch

from phi0.data.temporal_align import (
    observation_subsampled_frame_index,
    proprio_current_control_step,
    video_sample_control_indices,
)
from phi0.models.vlm.preprocess import (
    QWEN_VL_IMAGE_PATCH_SIZE,
    build_qwenvl_inputs_single,
    normalize_vlm_instruction,
)


@dataclass(frozen=True)
class VlmPretrainContract:
    """Frozen knobs that must match Psi0 ``Psi0ModelTransform`` + Qwen3-VL pretrain."""

    image_size_hw: Tuple[int, int] = (180, 320)
    patch_size: int = QWEN_VL_IMAGE_PATCH_SIZE
    lowercase_instruction: bool = True
    add_generation_prompt: bool = True

    def validate_processor(self, processor: Any) -> None:
        del processor  # ponytail: spatial path is Resize NEAREST + CenterCrop in preprocess.


def resolve_vlm_observation_frame_index(
    *,
    pixel_time: int,
    past_action_window_size: int,
    subsampled_control_indices: Sequence[int] | None = None,
    seq_len: int | None = None,
    action_video_freq_ratio: int = 2,
) -> int:
    """Pick T-axis index into ``[B,T,C,H,W]`` for the proprio-aligned observation frame.

    Training passes ``subsampled_control_indices`` from the batch. Deploy past-only
    clips that end at the live control step use the last subsampled frame (``T-1``).
    """
    t = int(pixel_time)
    if t <= 0:
        raise ValueError(f"pixel_time must be positive, got {t}")
    if t == 1:
        return 0
    if subsampled_control_indices is not None and len(subsampled_control_indices) > 0:
        return observation_subsampled_frame_index(
            int(past_action_window_size),
            subsampled_control_indices,
        )
    if seq_len is not None:
        subs = video_sample_control_indices(int(seq_len), int(action_video_freq_ratio))
        return observation_subsampled_frame_index(int(past_action_window_size), subs)
    return t - 1


def slice_vlm_observation_pixel(
    pixel: torch.Tensor,
    frame_index: int,
) -> torch.Tensor:
    """``[B,T,C,H,W]`` or ``[B,1,C,H,W]`` -> single-frame ``[B,1,C,H,W]``."""
    if pixel.ndim != 5:
        raise ValueError(f"Expected pixel [B,T,C,H,W], got {tuple(pixel.shape)}")
    idx = int(frame_index)
    if idx < 0 or idx >= int(pixel.shape[1]):
        raise IndexError(f"frame_index {idx} out of range for T={pixel.shape[1]}")
    return pixel[:, idx : idx + 1]


def build_pretrain_aligned_vlm_inputs(
    processor,
    images: Sequence[Any],
    instruction: str,
    *,
    contract: VlmPretrainContract | None = None,
) -> Dict[str, torch.Tensor]:
    """Single-sample Qwen3-VL tensors with Psi0 instruction + patch contract."""
    contract = contract or VlmPretrainContract()
    if contract.lowercase_instruction:
        instruction = normalize_vlm_instruction(instruction)
    out = build_qwenvl_inputs_single(processor, images, instruction)
    required = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    missing = [k for k in required if k not in out]
    if missing:
        raise KeyError(f"VLM inputs missing keys: {missing}")
    return out


def assert_training_deploy_vlm_frame_parity(
    train_pixel: torch.Tensor,
    deploy_pixel: torch.Tensor,
    *,
    past_action_window_size: int,
    subsampled_control_indices: Sequence[int],
) -> None:
    """Raise if train vs deploy observation slices would diverge (train_obs T=1 case)."""
    t_train = int(train_pixel.shape[1])
    t_deploy = int(deploy_pixel.shape[1])
    if t_train != 1:
        raise AssertionError(f"expected train obs_pixel T=1, got T={t_train}")
    idx_deploy = resolve_vlm_observation_frame_index(
        pixel_time=t_deploy,
        past_action_window_size=past_action_window_size,
        subsampled_control_indices=None,
    )
    anchor = proprio_current_control_step(int(past_action_window_size))
    if int(subsampled_control_indices[0]) != anchor:
        raise AssertionError(
            f"train obs must align to control anchor {anchor}, got subs[0]={subsampled_control_indices[0]}"
        )
    if t_deploy > 1 and int(subsampled_control_indices[0]) != int(
        subsampled_control_indices[idx_deploy]
    ):
        raise AssertionError(
            "deploy observation frame must carry the same control index as training"
        )
