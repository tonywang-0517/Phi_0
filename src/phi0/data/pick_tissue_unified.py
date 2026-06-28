"""LeRobot clip dataset for pick-tissue unified 512-d action (same loader path as sonic/GR00T)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from phi0.data.simple_lerobot import _import_lerobot
from phi0.data.psi0_image import read_lerobot_video_hw, stack_video_frames_native
from phi0.data.temporal_align import (
    proprio_current_control_step,
    video_sample_control_indices,
)
from phi0.data.video_cache import LeRobotPredecodedVideo, LeRobotTimestampFrameCache
from phi0.data.predecoded_video import (
    predecoded_root,
    validate_predecoded_store,
)
from phi0.data.xperience_unified_gt import repack_clip_root_trans_local
from phi0.schema.unified_action_schema import D_UNIFIED, dim_mask_for_dataset

EGO_IMAGE_KEY = "observation.images.ego_view"
LEFT_WRIST_IMAGE_KEY = "observation.images.left_wrist"
DATASET_NAME = "g1_sonic"


def _dim_pad_mask() -> torch.Tensor:
    return torch.from_numpy(~dim_mask_for_dataset(DATASET_NAME))


def _as_action_sequence(raw: Any, *, steps: int, dim: int) -> torch.Tensor:
    arr = torch.as_tensor(raw, dtype=torch.float32)
    if arr.ndim == 1:
        if arr.numel() == dim:
            arr = arr.unsqueeze(0)
        else:
            raise ValueError(f"expected action dim {dim}, got {arr.numel()}")
    if arr.shape[-1] != dim:
        raise ValueError(f"expected action dim {dim}, got {arr.shape[-1]}")
    if arr.shape[0] != steps:
        raise ValueError(f"expected {steps} action steps, got {arr.shape[0]}")
    return arr


def training_video_load_spec(
    *,
    fps: float,
    seq_len: int,
    action_video_freq_ratio: int,
    train_obs_only_video: bool,
    obs_control_index: int = 0,
) -> tuple[list[float], list[int]]:
    """LeRobot ``delta_timestamps`` for video keys + ``video_control_indices`` metadata."""
    full_ctrl = video_sample_control_indices(seq_len, action_video_freq_ratio)
    if train_obs_only_video:
        obs = int(obs_control_index)
        return [obs / fps], [obs]
    return [t / fps for t in full_ctrl], full_ctrl


def _as_root_sequence(raw: Any, *, steps: int) -> torch.Tensor:
    arr = torch.as_tensor(raw, dtype=torch.float32)
    if arr.ndim == 1:
        if arr.numel() == 3:
            arr = arr.unsqueeze(0)
        else:
            raise ValueError(f"expected root dim 3, got {arr.numel()}")
    if arr.shape[-1] != 3:
        raise ValueError(f"expected root dim 3, got {arr.shape[-1]}")
    if arr.shape[0] != steps:
        raise ValueError(f"expected {steps} root steps, got {arr.shape[0]}")
    return arr


class PickTissueUnifiedClipDataset(Dataset):
    """Egocentric + left-wrist clip via LeRobot temporal indexing (GR00T-compatible)."""

    DATASET_NAME = DATASET_NAME

    def __init__(
        self,
        *,
        root_dir: str | Path,
        repo_id: str,
        seq_len: int = 33,
        action_video_freq_ratio: int = 2,
        image_size: Tuple[int, int] = (180, 320),
        val: bool = False,
        val_ratio: float = 0.05,
        seed: int = 42,
        image_key: str = EGO_IMAGE_KEY,
        use_left_wrist: bool = True,
        left_wrist_image_key: str = LEFT_WRIST_IMAGE_KEY,
        cache_video: bool = True,
        video_cache_max_frames: int = 2048,
        video_backend: str | None = "torchcodec",
        use_predecoded_video: bool = True,
        train_obs_only_video: bool = False,
        obs_control_index: int | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.repo_id = str(repo_id)
        self.seq_len = int(seq_len)
        self.action_video_freq_ratio = max(1, int(action_video_freq_ratio))
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.image_key = image_key
        self.use_left_wrist = bool(use_left_wrist)
        self.left_wrist_image_key = left_wrist_image_key
        dataset_path = self.root_dir / self.repo_id
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"pick-tissue unified dataset not found: {dataset_path}")
        self._native_image_size = read_lerobot_video_hw(dataset_path, image_key)

        LeRobotDataset, LeRobotDatasetMetadata = _import_lerobot()
        meta = LeRobotDatasetMetadata(self.repo_id, str(dataset_path))
        fps = float(meta.fps)
        self._fps = fps
        obs_ctrl = (
            int(obs_control_index)
            if obs_control_index is not None
            else proprio_current_control_step(1)
        )
        video_ts, self.video_control_indices = training_video_load_spec(
            fps=fps,
            seq_len=self.seq_len,
            action_video_freq_ratio=self.action_video_freq_ratio,
            train_obs_only_video=bool(train_obs_only_video),
            obs_control_index=obs_ctrl,
        )
        action_ts = [t / fps for t in range(self.seq_len)]
        delta_timestamps: dict[str, Sequence[float]] = {
            self.image_key: video_ts,
            "unified_action": action_ts,
            "target_root_trans_world": action_ts,
            "state_root_trans_world": [0.0],
            "betas": [0.0],
        }
        if self.use_left_wrist:
            delta_timestamps[self.left_wrist_image_key] = video_ts

        lerobot_kwargs: dict[str, Any] = {
            "delta_timestamps": delta_timestamps,
            "image_transforms": None,
        }
        if video_backend:
            lerobot_kwargs["video_backend"] = str(video_backend)
        base = LeRobotDataset(self.repo_id, root=str(dataset_path), **lerobot_kwargs)
        store_root = predecoded_root(dataset_path, self._native_image_size)
        predecoded_ok = False
        if use_predecoded_video and store_root.is_dir():
            errs = validate_predecoded_store(dataset_path, store_root)
            predecoded_ok = len(errs) == 0
        if predecoded_ok:
            from phi0.data.predecoded_video import PredecodedVideoStore

            self._base = LeRobotPredecodedVideo(base, PredecodedVideoStore(store_root))
        elif cache_video:
            self._base = LeRobotTimestampFrameCache(
                base,
                max_frames=video_cache_max_frames,
                image_size=None,
            )
        else:
            self._base = base
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

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._base[self._indices[idx]]
        ego = stack_video_frames_native(sample[self.image_key])
        images: Dict[str, torch.Tensor] = {"ego_view": ego}
        if self.use_left_wrist:
            if self.left_wrist_image_key not in sample:
                raise KeyError(
                    f"use_left_wrist=True but sample missing {self.left_wrist_image_key!r}"
                )
            images["wrist_view"] = stack_video_frames_native(
                sample[self.left_wrist_image_key]
            )

        actions = _as_action_sequence(
            sample["unified_action"], steps=self.seq_len, dim=D_UNIFIED
        )
        roots = _as_root_sequence(
            sample["target_root_trans_world"], steps=self.seq_len
        )
        repacked = repack_clip_root_trans_local(
            actions.numpy(),
            roots.numpy(),
            anchor_index=0,
        )
        action_ctrl = torch.from_numpy(repacked)

        task = sample.get("task", "pick tissue")
        if torch.is_tensor(task):
            task = str(task)
        instruction = str(task).lower()

        mask = _dim_pad_mask()
        dim_pad = mask.unsqueeze(0).expand(self.seq_len, -1).clone()

        return {
            "dataset": self.DATASET_NAME,
            "idx": int(self._indices[idx]),
            "task": instruction,
            "images": images,
            "image_is_pad": torch.zeros(ego.shape[0], dtype=torch.bool),
            "action": action_ctrl,
            "action_is_pad": torch.zeros(self.seq_len, dtype=torch.bool),
            "action_dim_is_pad": dim_pad,
            "control_fps": self._fps,
            "action_video_freq_ratio": self.action_video_freq_ratio,
            "video_control_indices": torch.tensor(self.video_control_indices, dtype=torch.long),
            "state_root_trans_world": torch.as_tensor(
                sample["state_root_trans_world"], dtype=torch.float32
            ).reshape(3),
            "betas": torch.as_tensor(sample["betas"], dtype=torch.float32).reshape(-1),
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
            "state_root_trans_world": torch.stack(
                [b["state_root_trans_world"] for b in batch]
            ),
            "betas": torch.stack([b["betas"] for b in batch]),
        }
