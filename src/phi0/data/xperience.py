"""Xperience-10M sample loader → Phi_0 D_raw + masks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from phi0.data.video_cache import preload_mp4_frames
from phi0.data.cosmos_video_size import DEFAULT_COSMOS_VIDEO_SIZE
from phi0.schema.draw_schema import D_RAW, DrawLayout, pack_xperience_keypoints

DEFAULT_HDF5 = Path(
    "/mnt/data2/wpy/workspace/Isaac-GR00T/demo_data/xperience-10m-sample/annotation.hdf5"
)
DEFAULT_VIDEO = Path(
    "/mnt/data2/wpy/workspace/Isaac-GR00T/demo_data/xperience-10m-sample/stereo_left.mp4"
)
# Fallback if demo lives on data1 mount
if not DEFAULT_HDF5.exists():
    DEFAULT_HDF5 = Path(
        "/mnt/data1/wpy/workspace/Isaac-GR00T/demo_data/xperience-10m-sample/annotation.hdf5"
    )
if not DEFAULT_VIDEO.exists():
    DEFAULT_VIDEO = Path(
        "/mnt/data1/wpy/workspace/Isaac-GR00T/demo_data/xperience-10m-sample/stereo_left.mp4"
    )


def resolve_xperience_video_path(explicit: str | Path | None = None) -> Path | None:
    """Resolve monocular left-eye video for Xperience (stereo_left.mp4 only)."""
    if explicit is not None:
        p = Path(explicit)
        return p if p.exists() else None
    if DEFAULT_VIDEO.exists():
        return DEFAULT_VIDEO
    return None


class XperienceDataset(Dataset):
    """Frame-level dataset from Xperience annotation.hdf5."""

    DATASET_NAME = "xperience"
    NATIVE_FPS = 20.0

    def __init__(
        self,
        hdf5_path: str | Path = DEFAULT_HDF5,
        max_frames: Optional[int] = None,
        frame_stride: int = 1,
        start_frame: int = 0,
        video_path: Optional[str | Path] = None,
        image_size: Tuple[int, int] = DEFAULT_COSMOS_VIDEO_SIZE,
        cache_video: bool = True,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.frame_stride = int(frame_stride)
        self.start_frame = int(start_frame)
        self.image_size = image_size
        resolved = resolve_xperience_video_path(video_path)
        self.video_path = resolved
        self.layout = DrawLayout()
        self.action_dim_is_pad = self.layout.dim_mask_for_dataset(self.DATASET_NAME)

        with h5py.File(self.hdf5_path, "r") as f:
            self.n_total = int(f["full_body_mocap/body_quats"].shape[0])
            caption_raw = f["caption"][()]
            if isinstance(caption_raw, bytes):
                caption_raw = caption_raw.decode("utf-8")
            self.caption = json.loads(caption_raw)
            self.task_text = self.caption.get("config", {}).get("Main Task", "human egocentric task")

        end = self.n_total if max_frames is None else min(self.n_total, start_frame + max_frames)
        self.frame_indices = list(range(start_frame, end, self.frame_stride))
        self._h5: Optional[h5py.File] = None
        self._video_frames: Optional[List[torch.Tensor]] = None
        if cache_video and self.video_path is not None:
            self._video_frames = preload_mp4_frames(
                self.video_path,
                self.image_size,
                max_frames=end,
            )

    def __len__(self) -> int:
        return len(self.frame_indices)

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def _load_frame_action(self, t: int) -> np.ndarray:
        f = self._get_h5()
        keypoints = f["full_body_mocap/keypoints"][t].astype(np.float32)
        betas = f["full_body_mocap/betas"][t].astype(np.float32)
        tactile = np.zeros(10, dtype=np.float32)
        return pack_xperience_keypoints(keypoints, betas, tactile)

    def _placeholder_image(self, t: int) -> torch.Tensor:
        h, w = self.image_size
        rng = np.random.RandomState(t)
        img = rng.rand(h, w, 3).astype(np.float32)
        return torch.from_numpy(img).permute(2, 0, 1)

    def _load_image(self, t: int) -> tuple[torch.Tensor, bool]:
        """Return (RGB tensor, uses_real_video). Placeholder if MP4 unavailable."""
        if self._video_frames is not None and 0 <= t < len(self._video_frames):
            return self._video_frames[t], True

        h, w = self.image_size
        if self.video_path is not None and self.video_path.exists():
            try:
                import cv2

                cap = cv2.VideoCapture(str(self.video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, t)
                ok, frame = cap.read()
                cap.release()
                if ok:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
                    return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0, True
            except Exception:
                pass
        return self._placeholder_image(t), False

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        t = self.frame_indices[idx]
        action = self._load_frame_action(t)
        image, uses_real_video = self._load_image(t)
        return {
            "dataset": self.DATASET_NAME,
            "idx": idx,
            "frame_index": t,
            "task": self.task_text,
            "uses_real_video": uses_real_video,
            "images": {"ego_view": image.unsqueeze(0)},
            "image_is_pad": torch.zeros(1, dtype=torch.bool),
            "action": torch.from_numpy(action).unsqueeze(0),
            "action_is_pad": torch.zeros(1, dtype=torch.bool),
            "action_dim_is_pad": torch.from_numpy(~self.action_dim_is_pad),
        }

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "dataset": [b["dataset"] for b in batch],
            "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
            "task": [b["task"] for b in batch],
            "image_is_pad": torch.stack([b["image_is_pad"] for b in batch]),
            "action_is_pad": torch.stack([b["action_is_pad"] for b in batch]),
            "action_dim_is_pad": batch[0]["action_dim_is_pad"],
        }
        out["images"] = {
            "ego_view": torch.stack([b["images"]["ego_view"] for b in batch]),
        }
        out["action"] = torch.stack([b["action"] for b in batch])
        return out
