"""Mixed dataset + processor with action normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, Dataset

from phi0.data.dit4dit_video import dit4dit_preprocess_video
from phi0.data.egodex import EgoDexDataset
from phi0.data.cosmos_video_size import DEFAULT_COSMOS_VIDEO_SIZE, cosmos_video_size_from_cfg
from phi0.data.xperience import XperienceDataset
from phi0.schema.draw_schema import D_RAW
from phi0.data.action_stats import load_action_stats, stats_to_tensors


class Phi0MixedDataset(Dataset):
    """Concatenate datasets; DataLoader shuffle interleaves clips across sources."""

    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.cumulative = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative.append(total)

    def __len__(self) -> int:
        return self.cumulative[-1] if self.cumulative else 0

    def _locate(self, idx: int):
        for i, end in enumerate(self.cumulative):
            if idx < end:
                start = 0 if i == 0 else self.cumulative[i - 1]
                return self.datasets[i], idx - start
        raise IndexError(idx)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ds, local = self._locate(idx)
        return ds[local]

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "dataset": [b["dataset"] for b in batch],
            "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
            "task": [b["task"] for b in batch],
            "image_is_pad": torch.stack([b["image_is_pad"] for b in batch]),
            "action_is_pad": torch.stack([b["action_is_pad"] for b in batch]),
            "action_dim_is_pad": torch.stack([b["action_dim_is_pad"] for b in batch]),
            "images": {"ego_view": torch.stack([b["images"]["ego_view"] for b in batch])},
            "action": torch.stack([b["action"] for b in batch]),
        }
        return out


def build_overfit_datasets(
    xperience_max_frames: int = 32,
    egodex_max_frames: int = 32,
    xperience_video: str | Path | None = None,
    cache_video: bool = True,
    image_size: Tuple[int, int] = DEFAULT_COSMOS_VIDEO_SIZE,
) -> Phi0MixedDataset:
    return Phi0MixedDataset(
        [
            XperienceDataset(
                max_frames=xperience_max_frames,
                video_path=xperience_video,
                cache_video=cache_video,
                image_size=image_size,
            ),
            EgoDexDataset(
                max_frames=egodex_max_frames,
                cache_video=cache_video,
                image_size=image_size,
            ),
        ]
    )


class Phi0Processor:
    """Normalize actions for training and deploy."""

    def __init__(
        self,
        action_dim: int = D_RAW,
        normalize: bool = True,
        *,
        cosmos_video_size: Tuple[int, int] = DEFAULT_COSMOS_VIDEO_SIZE,
        cosmos_video_crop_scale: Optional[float] = None,
    ):
        self.action_dim = action_dim
        self.normalize = normalize
        self.cosmos_video_size = (int(cosmos_video_size[0]), int(cosmos_video_size[1]))
        self.cosmos_video_crop_scale = cosmos_video_crop_scale
        self._is_train = True
        self.register_stats()

    def register_stats(self, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None):
        if mean is None:
            mean = torch.zeros(self.action_dim)
        if std is None:
            std = torch.ones(self.action_dim)
        self.mean = mean
        self.std = std.clamp(min=1e-6)

    def register_stats_from_dict(self, stats: dict) -> None:
        mean, std = stats_to_tensors(stats, self.action_dim)
        self.register_stats(mean, std)

    def load_stats_path(self, path: str | Path) -> None:
        self.register_stats_from_dict(load_action_stats(path))

    def stats_dict(self) -> dict:
        return {
            "version": 1,
            "action_dim": self.action_dim,
            "norm_mode": "z-score",
            "mean": self.mean.detach().cpu().tolist(),
            "std": self.std.detach().cpu().tolist(),
        }

    def train(self):
        self._is_train = True
        return self

    def eval(self):
        self._is_train = False
        return self

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return action
        m = self.mean.to(action.device, dtype=action.dtype)
        s = self.std.to(action.device, dtype=action.dtype)
        return (action - m) / s

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        m = self.mean.to(action.device, dtype=action.dtype)
        s = self.std.to(action.device, dtype=action.dtype)
        return action * s + m

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        action = batch["action"].float()
        action_norm = self._normalize_action(action)

        pixel = batch["images"]["ego_view"]
        if pixel.ndim == 5:
            pass
        elif pixel.ndim == 6:
            pixel = pixel[:, 0]
        else:
            raise ValueError(f"Expected ego_view [B,T,C,H,W] or [B,Cam,T,C,H,W], got {pixel.ndim}D")

        cosmos_pixel = dit4dit_preprocess_video(
            pixel.float(),
            size=self.cosmos_video_size,
            crop_scale=self.cosmos_video_crop_scale,
        )

        return {
            "instruction": batch["task"],
            "pixel_values": cosmos_pixel,
            "pixel_values_native": pixel,
            "image_is_pad": batch["image_is_pad"],
            "action": action_norm,
            "action_is_pad": batch["action_is_pad"],
            "action_dim_is_pad": batch["action_dim_is_pad"],
            "idx": batch["idx"],
        }

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        return self._denormalize_action(action)
