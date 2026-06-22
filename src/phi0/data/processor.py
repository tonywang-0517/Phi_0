"""Mixed dataset + processor with action normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, Dataset

from phi0.data.egodex import EgoDexDataset
from phi0.schema.draw_schema import D_RAW
from phi0.data.action_stats import load_action_stats, stats_to_tensors
from phi0.data.robot_action_norm import (
    denormalize_robot7d,
    normalize_robot7d,
    processor_stats_dict,
    stats_view_for_robot7d,
)


DEFAULT_VLM_IMAGE_SIZE = (180, 320)


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
    image_size: Tuple[int, int] = DEFAULT_VLM_IMAGE_SIZE,
) -> Phi0MixedDataset:
    from phi0.data.xperience import XperienceDataset

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
        vlm_image_size: Tuple[int, int] = DEFAULT_VLM_IMAGE_SIZE,
        vlm_img_aug: bool = False,
        use_wrist_view: bool = False,
    ):
        self.action_dim = action_dim
        self.normalize = normalize
        self.vlm_image_size = (int(vlm_image_size[0]), int(vlm_image_size[1]))
        self.vlm_img_aug = bool(vlm_img_aug)
        self.use_wrist_view = bool(use_wrist_view)
        self._is_train = True
        self._vlm_transform = None
        self._vlm_transform_key: tuple[Any, ...] | None = None
        self.action_norm_mode = "z-score"
        self.proprio_norm_mode = "z-score"
        self.robot_action_semantics = ""
        self.normalize_gripper = True
        self.register_stats()

    def register_stats(self, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None):
        if mean is None:
            mean = torch.zeros(self.action_dim)
        if std is None:
            std = torch.ones(self.action_dim)
        self.mean = mean
        self.std = std.clamp(min=1e-6)
        self.action_q01 = mean.clone()
        self.action_q99 = mean.clone()

    def register_stats_from_dict(self, stats: dict) -> None:
        mean, std, q01, q99 = stats_to_tensors(stats, self.action_dim)
        self.register_stats(mean, std)
        self.action_q01 = q01
        self.action_q99 = q99
        self.action_norm_mode = str(stats.get("norm_mode", "z-score")).strip().lower()
        self.robot_action_semantics = str(stats.get("robot_action_semantics", ""))
        self.normalize_gripper = bool(stats.get("normalize_gripper", True))

    def load_stats_path(self, path: str | Path) -> None:
        self.register_stats_from_dict(load_action_stats(path))

    def register_proprio_stats_from_dict(self, stats: dict) -> None:
        mean, std, q01, q99 = stats_to_tensors(stats, self.action_dim)
        self.proprio_mean = mean
        self.proprio_std = std.clamp(min=1e-6)
        self.proprio_q01 = q01
        self.proprio_q99 = q99
        self.proprio_norm_mode = str(stats.get("norm_mode", "z-score")).strip().lower()

    def load_proprio_stats_path(self, path: str | Path) -> None:
        self.register_proprio_stats_from_dict(load_action_stats(path))

    def stats_dict(self) -> dict:
        return processor_stats_dict(self)

    def train(self):
        self._is_train = True
        self._vlm_transform = None
        self._vlm_transform_key = None
        return self

    def eval(self):
        self._is_train = False
        self._vlm_transform = None
        self._vlm_transform_key = None
        return self

    def vlm_image_transform(self):
        """Cached Psi0-style VLM transform (invalidated on train/eval toggle)."""
        key = (self.vlm_image_size, self.vlm_img_aug, self._is_train)
        if self._vlm_transform_key != key:
            from phi0.models.vlm.preprocess import make_psi0_vlm_image_transform

            self._vlm_transform = make_psi0_vlm_image_transform(
                self.vlm_image_size,
                img_aug=self.vlm_img_aug,
                training=self._is_train,
            )
            self._vlm_transform_key = key
        return self._vlm_transform

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return action
        m = self.mean.to(action.device, dtype=action.dtype)
        s = self.std.to(action.device, dtype=action.dtype)
        return (action - m) / s

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_norm_mode == "bounds_q99":
            out = action.clone()
            stats = stats_view_for_robot7d(self, proprio=False)
            out[..., :7] = denormalize_robot7d(
                action[..., :7],
                stats,
                denormalize_gripper=self.normalize_gripper,
            )
            if action.shape[-1] > 7:
                m = self.mean[7:].to(action.device, dtype=action.dtype)
                s = self.std[7:].to(action.device, dtype=action.dtype)
                out[..., 7:] = action[..., 7:] * s + m
            return out
        m = self.mean.to(action.device, dtype=action.dtype)
        s = self.std.to(action.device, dtype=action.dtype)
        return action * s + m

    def denormalize_robot7d_future(self, pred_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize normalized future chunk to physical 7D controls."""
        stats = stats_view_for_robot7d(self, proprio=False)
        return denormalize_robot7d(
            pred_norm[..., :7],
            stats,
            denormalize_gripper=False,
        )

    def normalize_robot7d_tensor(
        self,
        action_7d: torch.Tensor,
        *,
        proprio: bool = False,
        normalize_gripper: Optional[bool] = None,
    ) -> torch.Tensor:
        stats = stats_view_for_robot7d(self, proprio=proprio)
        grip = self.normalize_gripper if normalize_gripper is None else normalize_gripper
        if proprio:
            grip = True
        return normalize_robot7d(action_7d, stats, normalize_gripper=grip)

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        action = batch["action"].float()
        action_norm = self._normalize_action(action)

        pixel = batch["images"]["ego_view"].float()
        if pixel.ndim == 5:
            pass
        elif pixel.ndim == 6:
            pixel = pixel[:, 0]
        else:
            raise ValueError(f"Expected ego_view [B,T,C,H,W] or [B,Cam,T,C,H,W], got {pixel.ndim}D")

        wrist_pixel = None
        if self.use_wrist_view:
            wrist = batch.get("images", {}).get("wrist_view")
            if wrist is None:
                raise ValueError("use_wrist_view=True but batch missing images.wrist_view")
            wrist_pixel = wrist.float()
            if wrist_pixel.ndim != 5:
                raise ValueError(
                    f"Expected wrist_view [B,T,C,H,W], got {wrist_pixel.ndim}D"
                )
            pixel = torch.stack([pixel, wrist_pixel], dim=1)

        from phi0.models.vlm.preprocess import normalize_vlm_instruction

        task = batch["task"]
        if isinstance(task, str):
            instruction = normalize_vlm_instruction(task)
        else:
            instruction = [normalize_vlm_instruction(t) for t in task]

        return {
            "instruction": instruction,
            "pixel_values": pixel,
            "image_is_pad": batch["image_is_pad"],
            "action": action_norm,
            "action_is_pad": batch["action_is_pad"],
            "action_dim_is_pad": batch["action_dim_is_pad"],
            "idx": batch["idx"],
        }

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        return self._denormalize_action(action)
