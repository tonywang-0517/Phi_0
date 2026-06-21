"""LIBERO RLDS frame dataset for Phi_0 ACT training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from phi0.benchmark.paths import libero_rlds_dir
from phi0.benchmark.rlds_adapters import (
    libero_rlds_action_to_train,
    libero_rlds_state_to_eef_7d,
)
from phi0.benchmark.rlds_io import RldsEpisode, iter_rlds_shards, libero_train_shard_glob
from phi0.schema.draw_schema import D_RAW

ROBOT_ACTION_DIM = 7  # VLA-Adapter LIBERO_CONSTANTS["ACTION_DIM"]


def _robot_action_7d_tensor(action_7d: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(
        np.asarray(action_7d, dtype=np.float32).reshape(ROBOT_ACTION_DIM)
    ).unsqueeze(0)


def _libero_dim_pad() -> torch.Tensor:
    """All action dims supervised (legacy D_raw placeholder; training overrides with 7D)."""
    return torch.zeros(D_RAW, dtype=torch.bool)


def _native_chw_f32(img: np.ndarray) -> torch.Tensor:
    """uint8 HWC -> float CHW in [0,1] at native resolution (no resize)."""
    arr = np.array(img, dtype=np.uint8, copy=True)
    t = torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0)
    return t.unsqueeze(0)


def _resize_chw(img: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    """uint8 HWC -> float CHW in [0,1], resized to (H,W)=size.

    RLDS JPEGs are already 180°-rotated in the OpenVLA conversion; do not flip here.
    """
    arr = np.array(img, dtype=np.uint8, copy=True)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    t = F.interpolate(t.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
    return t.unsqueeze(0)


class LiberoRldsFrameDataset(Dataset):
    """Per-frame samples from OpenVLA-style LIBERO RLDS tfrecords."""

    DATASET_NAME = "libero_spatial"

    def __init__(
        self,
        *,
        suite: str = "libero_spatial",
        rlds_root: str | Path | None = None,
        image_size: tuple[int, int] = (224, 224),
        max_episodes: int | None = None,
        max_shards: int | None = None,
        libero_delta_eef: bool = True,
        defer_cosmos_resize: bool = False,
        cache_native_frames: bool = False,
    ) -> None:
        self.libero_delta_eef = bool(libero_delta_eef)
        self.defer_cosmos_resize = bool(defer_cosmos_resize)
        self.cache_native_frames = bool(cache_native_frames)
        self.suite = str(suite).replace("_no_noops", "")
        self.DATASET_NAME = self.suite
        self.image_size = (int(image_size[0]), int(image_size[1]))
        if rlds_root is not None:
            root = Path(rlds_root)
            if (root / f"{self.suite}_no_noops" / "1.0.0").is_dir():
                root = root / f"{self.suite}_no_noops" / "1.0.0"
            elif (root / "1.0.0").is_dir():
                root = root / "1.0.0"
        else:
            root = libero_rlds_dir(self.suite)
        shard_pat = libero_train_shard_glob(self.suite, root)
        if not list(Path(shard_pat).parent.glob(Path(shard_pat).name)):
            raise FileNotFoundError(f"No LIBERO RLDS shards: {shard_pat}")

        self._episodes: list[RldsEpisode] = []
        for ep in iter_rlds_shards(shard_pat, benchmark="libero", max_shards=max_shards):
            self._episodes.append(ep)
            if max_episodes is not None and len(self._episodes) >= int(max_episodes):
                break
        if not self._episodes:
            raise RuntimeError(f"No LIBERO episodes loaded from {shard_pat}")

        self._frame_index: list[tuple[int, int]] = []
        for ep_i, ep in enumerate(self._episodes):
            for step_i in range(len(ep.steps)):
                self._frame_index.append((ep_i, step_i))

        self._native_image_cache: list[torch.Tensor] | None = None
        if self.cache_native_frames:
            self._native_image_cache = [
                _native_chw_f32(self._episodes[ep_i].steps[step_i].rgb_static)[0]
                for ep_i, step_i in self._frame_index
            ]

    def _frame_image(self, idx: int, rgb_static: np.ndarray) -> torch.Tensor:
        if self._native_image_cache is not None:
            return self._native_image_cache[idx].unsqueeze(0)
        if self.defer_cosmos_resize:
            return _native_chw_f32(rgb_static)
        return _resize_chw(rgb_static, self.image_size)

    def __len__(self) -> int:
        return len(self._frame_index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep_i, step_i = self._frame_index[idx]
        step = self._episodes[ep_i].steps[step_i]
        proprio_7d = libero_rlds_state_to_eef_7d(step.state)
        out: dict[str, Any] = {
            "dataset": self.DATASET_NAME,
            "idx": int(idx),
            "task": step.language or "complete the manipulation task",
            "action": torch.zeros(1, D_RAW, dtype=torch.float32),
            "action_dim_is_pad": _libero_dim_pad(),
            "images": {"ego_view": self._frame_image(idx, step.rgb_static)},
        }
        if self.libero_delta_eef:
            delta_7d = libero_rlds_action_to_train(step.action)
            out["robot_proprio_7d"] = _robot_action_7d_tensor(proprio_7d)
            out["robot_delta_7d"] = _robot_action_7d_tensor(delta_7d)
        else:
            out["robot_action_7d"] = _robot_action_7d_tensor(proprio_7d)
        return out
