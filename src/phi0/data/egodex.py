"""EgoDex loader: video + language + sparse SMPL+H D_raw (from preprocessed HDF5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from phi0.data.cosmos_video_size import DEFAULT_COSMOS_VIDEO_SIZE
from phi0.data.egodex_keypoints import pack_egodex_keypoints_d_raw
from phi0.data.egodex_smplh import default_processed_path, resolve_processed_hdf5
from phi0.data.video_cache import preload_mp4_frames
from phi0.schema.draw_schema import D_RAW, DrawLayout

_LEGACY_D_RAW = 237


def _pad_to_current_d_raw(arr: np.ndarray) -> np.ndarray:
    """Upgrade legacy 237-d processed arrays to current D_RAW (256)."""
    if arr.shape[-1] == D_RAW:
        return arr
    if arr.shape[-1] == _LEGACY_D_RAW:
        out = np.zeros(arr.shape[:-1] + (D_RAW,), dtype=arr.dtype)
        out[..., :_LEGACY_D_RAW] = arr
        return out
    raise ValueError(f"Expected d_raw/dim last dim {_LEGACY_D_RAW} or {D_RAW}, got {arr.shape[-1]}")

DEFAULT_ROOT = Path("/mnt/data1/wpy/workspace/Isaac-GR00T/demo_data/egodex")
DEFAULT_HDF5 = DEFAULT_ROOT / "test/add_remove_lid/0.hdf5"
DEFAULT_MP4 = DEFAULT_ROOT / "test/add_remove_lid/0.mp4"


class EgoDexDataset(Dataset):
    DATASET_NAME = "egodex"
    NATIVE_FPS = 30.0

    def __init__(
        self,
        hdf5_path: str | Path = DEFAULT_HDF5,
        mp4_path: str | Path = DEFAULT_MP4,
        processed_hdf5: str | Path | None = None,
        max_frames: Optional[int] = None,
        frame_stride: int = 1,
        image_size: Tuple[int, int] = DEFAULT_COSMOS_VIDEO_SIZE,
        require_processed: bool = True,
        cache_video: bool = True,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.mp4_path = Path(mp4_path)
        self.frame_stride = int(frame_stride)
        self.image_size = image_size
        self.layout = DrawLayout()
        self.require_processed = require_processed

        processed = resolve_processed_hdf5(self.hdf5_path, processed_hdf5)
        if processed is None and require_processed:
            expected = default_processed_path(self.hdf5_path)
            raise FileNotFoundError(
                f"Missing sparse SMPL+H cache for {self.hdf5_path}. "
                f"Run: python scripts/preprocess_egodex_smplh.py {self.hdf5_path}"
            )
        self.processed_hdf5 = processed

        with h5py.File(self.hdf5_path, "r") as f:
            self.n_total = int(f["transforms/camera"].shape[0])
            raw_task = f.attrs.get("task", f.attrs.get("llm_description", None))
            if raw_task is not None:
                self.task_text = str(raw_task)
            else:
                self.task_text = self.hdf5_path.parent.name.replace("_", " ")

        if self.processed_hdf5 is not None:
            with h5py.File(self.processed_hdf5, "r") as pf:
                self._proc_indices = pf["frame_indices"][:].astype(int).tolist()
                dim_union = _pad_to_current_d_raw(pf["dim_available"][:].astype(bool))
                if "dim_available_frame" in pf:
                    self._proc_dim_frame = _pad_to_current_d_raw(pf["dim_available_frame"][:].astype(bool))
                else:
                    self._proc_dim_frame = None
                self._proc_d_raw = _pad_to_current_d_raw(pf["d_raw"][:].astype(np.float32))
            self.dim_available_union = dim_union
        else:
            self._proc_indices = list(range(self.n_total))
            self._proc_dim_frame = None
            self._proc_d_raw = None
            self.dim_available_union = np.zeros(D_RAW, dtype=bool)

        end = len(self._proc_indices) if max_frames is None else min(len(self._proc_indices), max_frames)
        proc_slice = self._proc_indices[:end]
        self.frame_indices = proc_slice[:: self.frame_stride]
        self.proc_rows = list(range(len(proc_slice)))[:: self.frame_stride]
        self._video_frames: Optional[List[torch.Tensor]] = None
        if cache_video and self.mp4_path.exists():
            self._video_frames = preload_mp4_frames(
                self.mp4_path,
                self.image_size,
                max_frames=max(self.frame_indices) + 1 if self.frame_indices else None,
            )

    def __len__(self) -> int:
        return len(self.frame_indices)

    def _load_action(self, proc_row: int, t: int) -> tuple[np.ndarray, np.ndarray]:
        if self._proc_d_raw is None:
            return np.zeros(D_RAW, dtype=np.float32), np.ones(D_RAW, dtype=bool)
        d_raw_quat = self._proc_d_raw[proc_row].copy()
        if self._proc_dim_frame is not None:
            dim_quat = self._proc_dim_frame[proc_row].astype(bool)
        else:
            dim_quat = self.dim_available_union.astype(bool)
        return pack_egodex_keypoints_d_raw(d_raw_quat, dim_quat)

    def _placeholder_image(self, t: int) -> torch.Tensor:
        h, w = self.image_size
        rng = np.random.RandomState(t + 1000)
        img = rng.rand(h, w, 3).astype(np.float32)
        return torch.from_numpy(img).permute(2, 0, 1)

    def _load_image(self, t: int) -> tuple[torch.Tensor, bool]:
        if self._video_frames is not None and 0 <= t < len(self._video_frames):
            return self._video_frames[t], True

        h, w = self.image_size
        if self.mp4_path.exists():
            try:
                import cv2

                cap = cv2.VideoCapture(str(self.mp4_path))
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
        proc_row = self.proc_rows[idx]
        action, dim_avail = self._load_action(proc_row, t)
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
            "action_dim_is_pad": torch.from_numpy(~dim_avail),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "dataset": [b["dataset"] for b in batch],
            "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
            "task": [b["task"] for b in batch],
            "image_is_pad": torch.stack([b["image_is_pad"] for b in batch]),
            "action_is_pad": torch.stack([b["action_is_pad"] for b in batch]),
        }
        dim_pad = batch[0]["action_dim_is_pad"].clone()
        for b in batch[1:]:
            dim_pad = dim_pad & b["action_dim_is_pad"]
        out["action_dim_is_pad"] = dim_pad
        out["images"] = {"ego_view": torch.stack([b["images"]["ego_view"] for b in batch])}
        out["action"] = torch.stack([b["action"] for b in batch])
        return out
