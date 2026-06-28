"""Psi0-aligned vision: native-resolution loaders + VLM resize at processor time."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import torch

from phi0.data.simple_lerobot import _pt_to_chw_f32


def read_lerobot_video_hw(dataset_root: Path, video_key: str) -> Tuple[int, int]:
    """Return ``(H, W)`` from LeRobot ``meta/info.json`` feature shape."""
    info_path = Path(dataset_root) / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    shape = info["features"][video_key]["shape"]
    return int(shape[0]), int(shape[1])


def stack_video_frames_native(raw: Any) -> torch.Tensor:
    """LeRobot / predecoded batch → ``(T,C,H,W)`` float32 in ``[0,1]``, no spatial resize."""
    if torch.is_tensor(raw):
        t = raw.detach().cpu().float()
        if t.ndim == 4 and t.shape[1] in {1, 3}:
            if t.max() > 1.5:
                t = t / 255.0
            return t.clamp(0.0, 1.0)
        if t.ndim == 4 and t.shape[-1] in {1, 3}:
            frames = [_pt_to_chw_f32(t[i]) for i in range(t.shape[0])]
            return torch.stack(frames)
        if t.ndim == 3:
            return _pt_to_chw_f32(t).unsqueeze(0)
    if isinstance(raw, (list, tuple)):
        return torch.stack([_pt_to_chw_f32(f) for f in raw])
    return _pt_to_chw_f32(raw).unsqueeze(0)
