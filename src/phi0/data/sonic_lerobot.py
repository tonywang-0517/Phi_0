"""LeRobot dataset for SONIC unified 43-d state / 100-d action."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from phi0.data.sonic_unified_io import SONIC_ACTION_DIM, SONIC_STATE_DIM
from phi0.data.simple_lerobot import _pt_to_chw_f32
from phi0.data.psi0_image import stack_video_frames_native
from phi0.schema.draw_schema import D_RAW

EGO_IMAGE_KEY = "observation.images.egocentric"
LEFT_WRIST_IMAGE_KEY = "observation.images.left_wrist"


def _dim_pad_mask() -> torch.Tensor:
    mask = torch.zeros(D_RAW, dtype=torch.bool)
    mask[:SONIC_ACTION_DIM] = True
    return mask


class SonicUnifiedClipDataset(Dataset):
    """Egocentric + left-wrist frame, 43-d proprio, future 100-d action chunk."""

    DATASET_NAME = "sonic_unified"

    def __init__(
        self,
        *,
        root_dir: str | Path,
        repo_id: str,
        future_action_steps: int = 30,
        image_size: Tuple[int, int] = (180, 320),
        val: bool = False,
        val_ratio: float = 0.05,
        seed: int = 42,
        image_key: str = EGO_IMAGE_KEY,
        use_left_wrist: bool = True,
        left_wrist_image_key: str = LEFT_WRIST_IMAGE_KEY,
    ):
        self.root_dir = Path(root_dir)
        self.repo_id = str(repo_id)
        self.future_action_steps = int(future_action_steps)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.image_key = image_key
        self.use_left_wrist = bool(use_left_wrist)
        self.left_wrist_image_key = left_wrist_image_key
        dataset_path = self.root_dir / self.repo_id
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"SONIC unified dataset not found: {dataset_path}")

        from phi0.data.simple_lerobot import _import_lerobot

        LeRobotDataset, LeRobotDatasetMetadata = _import_lerobot()
        meta = LeRobotDatasetMetadata(self.repo_id, str(dataset_path))
        fps = float(meta.fps)
        self._fps = fps
        delta_timestamps = {
            self.image_key: [0.0],
            "states": [0.0],
            "action": [t / fps for t in range(self.future_action_steps)],
        }
        if self.use_left_wrist:
            delta_timestamps[self.left_wrist_image_key] = [0.0]
        self._base = LeRobotDataset(
            self.repo_id,
            root=str(dataset_path),
            delta_timestamps=delta_timestamps,
            image_transforms=None,
        )
        n = len(self._base)
        rng = np.random.default_rng(seed)
        indices = np.arange(n)
        rng.shuffle(indices)
        n_val = max(1, int(n * val_ratio)) if n > 1 else 0
        val_idx = set(indices[:n_val].tolist())
        if val:
            self._indices = [i for i in range(n) if i in val_idx]
        else:
            self._indices = [i for i in range(n) if i not in val_idx]
        if not self._indices:
            self._indices = list(range(n))

    @property
    def stats_path(self) -> Path:
        custom = getattr(self, "_stats_path", None)
        if custom is not None:
            return Path(custom)
        candidate = self.root_dir / self.repo_id / "meta" / "stats_sonic_unified.json"
        if candidate.is_file():
            return candidate
        return self.root_dir / self.repo_id / "meta" / "stats.json"

    def set_stats_path(self, path: str | Path) -> None:
        self._stats_path = Path(path)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._base[self._indices[idx]]
        img = stack_video_frames_native(sample[self.image_key])
        images: Dict[str, torch.Tensor] = {"ego_view": img}
        if self.use_left_wrist:
            if self.left_wrist_image_key not in sample:
                raise KeyError(
                    f"use_left_wrist=True but sample missing {self.left_wrist_image_key!r}"
                )
            images["wrist_view"] = stack_video_frames_native(sample[self.left_wrist_image_key])
        states = torch.as_tensor(sample["states"], dtype=torch.float32).reshape(-1)
        if states.numel() != SONIC_STATE_DIM:
            raise ValueError(f"expected states dim {SONIC_STATE_DIM}, got {states.numel()}")
        actions = torch.as_tensor(sample["action"], dtype=torch.float32)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        if actions.shape[-1] != SONIC_ACTION_DIM:
            raise ValueError(f"expected action dim {SONIC_ACTION_DIM}, got {actions.shape[-1]}")
        task = sample.get("task", "complete the manipulation task")
        if torch.is_tensor(task):
            task = str(task)
        instruction = str(task).lower()
        seq_len = 1 + self.future_action_steps
        return {
            "dataset": self.DATASET_NAME,
            "idx": int(self._indices[idx]),
            "task": instruction,
            "images": images,
            "image_is_pad": torch.zeros(img.shape[0], dtype=torch.bool),
            "action": torch.zeros(seq_len, D_RAW, dtype=torch.float32),
            "action_is_pad": torch.zeros(seq_len, dtype=torch.bool),
            "action_dim_is_pad": _dim_pad_mask(),
            "robot_proprio_43d": states.view(1, SONIC_STATE_DIM),
            "robot_future_100d": actions,
            "control_fps": self._fps,
            "action_video_freq_ratio": 1,
            "video_control_indices": torch.tensor([0], dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images: Dict[str, torch.Tensor] = {
            "ego_view": torch.stack([b["images"]["ego_view"] for b in batch]),
        }
        if "wrist_view" in batch[0]["images"]:
            images["wrist_view"] = torch.stack([b["images"]["wrist_view"] for b in batch])
        return {
            "dataset": [b["dataset"] for b in batch],
            "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
            "task": [b["task"] for b in batch],
            "image_is_pad": torch.stack([b["image_is_pad"] for b in batch]),
            "action_is_pad": torch.stack([b["action_is_pad"] for b in batch]),
            "action_dim_is_pad": torch.stack([b["action_dim_is_pad"] for b in batch]),
            "images": images,
            "action": torch.stack([b["action"] for b in batch]),
            "control_fps": batch[0].get("control_fps"),
            "action_video_freq_ratio": batch[0].get("action_video_freq_ratio"),
            "video_control_indices": batch[0].get("video_control_indices"),
            "robot_proprio_43d": torch.stack([b["robot_proprio_43d"] for b in batch]),
            "robot_future_100d": torch.stack([b["robot_future_100d"] for b in batch]),
        }
