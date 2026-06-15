"""Pad-frame handling for multi-frame video tensors (Cosmos VAE + VGGT)."""

from __future__ import annotations

from typing import Optional

import torch


def apply_video_pad_replacement(
    video: torch.Tensor,
    image_is_pad: Optional[torch.Tensor],
) -> torch.Tensor:
    """Replace padded timesteps with the first valid frame (matches Cosmos VAE path).

    ``video``: ``[B, 3, T, H, W]`` in [-1, 1].
    ``image_is_pad``: ``[B, T]`` or ``[T]`` bool — True where the frame is padded.
    """
    if image_is_pad is None:
        return video
    if video.ndim != 5:
        raise ValueError(f"`video` must be [B,3,T,H,W], got {tuple(video.shape)}")

    out = video.clone()
    pad = image_is_pad.to(device=out.device)
    if pad.ndim == 1:
        pad = pad.unsqueeze(0)
    if pad.shape[0] != out.shape[0] or pad.shape[1] != out.shape[2]:
        raise ValueError(
            f"image_is_pad shape {tuple(pad.shape)} incompatible with video batch/time "
            f"{out.shape[0]}, T={out.shape[2]}"
        )

    batch_size, _, num_frames, _, _ = out.shape
    for bi in range(batch_size):
        valid = ~pad[bi]
        if not valid.any():
            continue
        ref_t = int(valid.nonzero(as_tuple=False)[0].item())
        for t in range(num_frames):
            if pad[bi, t]:
                out[bi, :, t] = out[bi, :, ref_t]
    return out
