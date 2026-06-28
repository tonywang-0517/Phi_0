"""LeRobot SIMPLE G1 whole-body clips for Phi_0 training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from phi0.data.simple_action_norm import SIMPLE_G1_DIM
from phi0.schema.draw_schema import D_RAW


def _import_lerobot():
    try:
        from lerobot.common.datasets.lerobot_dataset import (  # type: ignore
            LeRobotDataset,
            LeRobotDatasetMetadata,
        )
    except ImportError:  # pragma: no cover - lerobot>=0.4
        from lerobot.datasets.lerobot_dataset import (  # type: ignore
            LeRobotDataset,
            LeRobotDatasetMetadata,
        )
    return LeRobotDataset, LeRobotDatasetMetadata


def _pt_to_chw_f32(img) -> torch.Tensor:
    if torch.is_tensor(img):
        t = img.detach().cpu().float()
        if t.ndim == 3 and t.shape[0] in {1, 3}:
            return t
        if t.ndim == 3 and t.shape[-1] in {1, 3}:
            return t.permute(2, 0, 1)
    pil = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img))
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _resize_chw(img: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    h, w = int(size[0]), int(size[1])
    pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8))
    pil = pil.resize((w, h), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _dim_pad_mask(dim: int = SIMPLE_G1_DIM) -> torch.Tensor:
    mask = torch.zeros(D_RAW, dtype=torch.bool)
    mask[: min(dim, D_RAW)] = True
    return mask


class SimpleG1ClipDataset(Dataset):
    """One training clip = egocentric frame + proprio state + future action chunk."""

    DATASET_NAME = "simple_g1"

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
    ):
        self.root_dir = Path(root_dir)
        self.repo_id = str(repo_id)
        self.future_action_steps = int(future_action_steps)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        dataset_path = self.root_dir / self.repo_id
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"SIMPLE dataset not found: {dataset_path}")

        LeRobotDataset, LeRobotDatasetMetadata = _import_lerobot()
        meta = LeRobotDatasetMetadata(self.repo_id, str(dataset_path))
        fps = float(meta.fps)
        self._fps = fps
        delta_timestamps = {
            "observation.images.egocentric": [0.0],
            "states": [0.0],
            "action": [t / fps for t in range(self.future_action_steps)],
        }
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
        candidate = self.root_dir / self.repo_id / "meta" / "stats_psi0.json"
        if candidate.is_file():
            return candidate
        return self.root_dir / self.repo_id / "meta" / "stats.json"

    def set_stats_path(self, path: str | Path) -> None:
        self._stats_path = Path(path)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._base[self._indices[idx]]
        image_key = "observation.images.egocentric"
        img = _resize_chw(_pt_to_chw_f32(sample[image_key]), self.image_size).unsqueeze(0)
        states = torch.as_tensor(sample["states"], dtype=torch.float32).reshape(-1)
        if states.numel() < SIMPLE_G1_DIM:
            pad = torch.zeros(SIMPLE_G1_DIM - states.numel(), dtype=torch.float32)
            states = torch.cat([states, pad], dim=0)
        else:
            states = states[:SIMPLE_G1_DIM]
        actions = torch.as_tensor(sample["action"], dtype=torch.float32)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        if actions.shape[-1] < SIMPLE_G1_DIM:
            pad = torch.zeros(
                *actions.shape[:-1], SIMPLE_G1_DIM - actions.shape[-1], dtype=torch.float32
            )
            actions = torch.cat([actions, pad], dim=-1)
        else:
            actions = actions[..., :SIMPLE_G1_DIM]
        task = sample.get("task", "complete the manipulation task")
        if torch.is_tensor(task):
            task = str(task)
        instruction = str(task).lower()
        seq_len = 1 + self.future_action_steps
        return {
            "dataset": self.DATASET_NAME,
            "idx": int(self._indices[idx]),
            "task": instruction,
            "images": {"ego_view": img},
            "image_is_pad": torch.zeros(img.shape[0], dtype=torch.bool),
            "action": torch.zeros(seq_len, D_RAW, dtype=torch.float32),
            "action_is_pad": torch.zeros(seq_len, dtype=torch.bool),
            "action_dim_is_pad": _dim_pad_mask(),
            "robot_proprio_36d": states.view(1, SIMPLE_G1_DIM),
            "robot_future_36d": actions,
            "control_fps": self._fps,
            "action_video_freq_ratio": 1,
            "video_control_indices": torch.tensor([0], dtype=torch.long),
        }

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
            "control_fps": batch[0].get("control_fps"),
            "action_video_freq_ratio": batch[0].get("action_video_freq_ratio"),
            "video_control_indices": batch[0].get("video_control_indices"),
            "robot_proprio_36d": torch.stack([b["robot_proprio_36d"] for b in batch]),
            "robot_future_36d": torch.stack([b["robot_future_36d"] for b in batch]),
        }
        return out
