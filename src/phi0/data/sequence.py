"""Temporal sequence sampling for Phi_0 training."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import torch
from torch.utils.data import Dataset

from phi0.data.temporal_align import (
    DEFAULT_DATASET_NATIVE_FPS,
    max_native_span_frames,
    native_span_frames,
    resample_action_sequence,
    resample_bool_sequence,
    resample_image_sequence,
    video_sample_control_indices,
)


def sequence_dataset_from_cfg(base: Dataset, data_cfg: Mapping[str, Any]) -> "SequenceDataset":
    native_fps = dict(data_cfg.get("dataset_native_fps") or DEFAULT_DATASET_NATIVE_FPS)
    return SequenceDataset(
        base,
        seq_len=int(data_cfg.get("seq_len", 5)),
        stride=int(data_cfg.get("clip_stride", 1)),
        control_fps=float(data_cfg.get("control_fps", 20.0)),
        action_video_freq_ratio=int(data_cfg.get("action_video_freq_ratio", 2)),
        native_fps=native_fps,
        future_action_steps=int(data_cfg.get("future_action_steps", 0)) or None,
    )


class SequenceDataset(Dataset):
    """Fixed-length clips on a unified control timeline (action T) + subsampled video for Cosmos."""

    def __init__(
        self,
        base: Dataset,
        seq_len: int = 5,
        stride: int = 1,
        *,
        control_fps: float = 20.0,
        action_video_freq_ratio: int = 2,
        native_fps: Optional[Mapping[str, float]] = None,
        future_action_steps: Optional[int] = None,
    ):
        self.base = base
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.control_fps = float(control_fps)
        self.action_video_freq_ratio = max(1, int(action_video_freq_ratio))
        self.native_fps = dict(native_fps or DEFAULT_DATASET_NATIVE_FPS)
        self.future_action_steps = (
            int(future_action_steps) if future_action_steps is not None else None
        )
        self.video_control_indices = video_sample_control_indices(
            self.seq_len, self.action_video_freq_ratio
        )
        self.video_seq_len = len(self.video_control_indices)

        n = len(base)
        max_span = max_native_span_frames(self.seq_len, self.control_fps, self.native_fps)
        self._segment_ranges = self._resolve_segment_ranges(base)
        self.starts = self._build_starts(n, max_span)
        if not self.starts and n > 0:
            self.starts = [0]

    def _resolve_segment_ranges(self, base: Dataset) -> list[tuple[int, int]]:
        """Inclusive start, exclusive end for each concatenated sub-dataset."""
        if hasattr(base, "cumulative") and hasattr(base, "datasets"):
            ranges: list[tuple[int, int]] = []
            offset = 0
            for ds in base.datasets:
                end = offset + len(ds)
                ranges.append((offset, end))
                offset = end
            return ranges
        return [(0, len(base))]

    def _native_span_for_start(self, start: int) -> int:
        ds_name = self.base[start]["dataset"]
        native_fps = float(self.native_fps[ds_name])
        return native_span_frames(self.seq_len, self.control_fps, native_fps)

    def _build_starts(self, n: int, max_span: int) -> list[int]:
        del max_span
        starts: list[int] = []
        for seg_start, seg_end in self._segment_ranges:
            seg_len = seg_end - seg_start
            if seg_len <= 0:
                continue
            last_start = seg_start + max(0, seg_len - 1)
            for s in range(seg_start, last_start + 1, self.stride):
                span = self._native_span_for_start(s)
                if s + span <= seg_end:
                    starts.append(s)
        return starts

    def _clip_end_for_start(self, start: int) -> int:
        for seg_start, seg_end in self._segment_ranges:
            if seg_start <= start < seg_end:
                return seg_end
        return len(self.base)

    def __len__(self) -> int:
        return max(len(self.starts), 1)

    def _load_native_clip(self, start: int, native_span: int, clip_end: int) -> List[Dict[str, Any]]:
        frames: List[Dict[str, Any]] = []
        last_idx = max(start, clip_end - 1)
        for i in range(native_span):
            idx = min(start + i, last_idx)
            frames.append(self.base[idx])
        return frames

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not self.starts:
            return self._item_from_native_frames([self.base[0]], padded=True)

        start = self.starts[idx % len(self.starts)]
        return self.sample_at_start(start)

    def sample_at_start(self, start: int, seq_len: int | None = None) -> Dict[str, Any]:
        """Build a clip anchored at ``start`` with optional override ``seq_len``."""
        seq_len = int(seq_len or self.seq_len)
        ds_name = self.base[start]["dataset"]
        native_fps = float(self.native_fps[ds_name])
        native_span = native_span_frames(seq_len, self.control_fps, native_fps)
        clip_end = self._clip_end_for_start(start)
        native_frames = self._load_native_clip(start, native_span, clip_end)
        actual_native = min(native_span, max(0, clip_end - start))
        padded = actual_native < native_span
        item = self._item_from_native_frames(native_frames, padded=padded, seq_len=seq_len)
        item["native_start"] = int(start)
        return item

    def _item_from_native_frames(
        self,
        native_frames: List[Dict[str, Any]],
        *,
        padded: bool,
        seq_len: int | None = None,
    ) -> Dict[str, Any]:
        src_len = len(native_frames)
        if src_len == 0:
            raise ValueError("empty native clip")

        actions = torch.stack([f["action"][0] for f in native_frames])
        robot_actions = None
        if "robot_action_7d" in native_frames[0]:
            robot_actions = torch.stack([f["robot_action_7d"][0] for f in native_frames])
        proprio_abs = None
        delta_actions = None
        if "robot_proprio_7d" in native_frames[0]:
            proprio_abs = torch.stack([f["robot_proprio_7d"][0] for f in native_frames])
        if "robot_delta_7d" in native_frames[0]:
            delta_actions = torch.stack([f["robot_delta_7d"][0] for f in native_frames])
        dim_pad = torch.stack(
            [
                f["action_dim_is_pad"].view(-1)
                if torch.is_tensor(f["action_dim_is_pad"])
                else torch.as_tensor(f["action_dim_is_pad"]).view(-1)
                for f in native_frames
            ]
        )

        seq_len_eff = int(seq_len or self.seq_len)
        video_idx = video_sample_control_indices(seq_len_eff, self.action_video_freq_ratio)
        if src_len == seq_len_eff:
            images_out = torch.stack(
                [native_frames[i]["images"]["ego_view"][0] for i in video_idx]
            )
            wrist_out = None
            if "wrist_view" in native_frames[0]["images"]:
                wrist_out = torch.stack(
                    [native_frames[i]["images"]["wrist_view"][0] for i in video_idx]
                )
        else:
            images = torch.stack([f["images"]["ego_view"][0] for f in native_frames])
            images_ctrl = resample_image_sequence(images, src_len, seq_len_eff)
            images_out = images_ctrl[video_idx]
            wrist_out = None
            if "wrist_view" in native_frames[0]["images"]:
                wrist = torch.stack([f["images"]["wrist_view"][0] for f in native_frames])
                wrist_ctrl = resample_image_sequence(wrist, src_len, seq_len_eff)
                wrist_out = wrist_ctrl[video_idx]
        action_ctrl = resample_action_sequence(actions, src_len, seq_len_eff)
        robot_ctrl = None
        if robot_actions is not None:
            robot_ctrl = resample_action_sequence(robot_actions, src_len, seq_len_eff)
        proprio_ctrl = None
        delta_ctrl = None
        if proprio_abs is not None:
            proprio_ctrl = resample_action_sequence(proprio_abs, src_len, seq_len_eff)
        if delta_actions is not None:
            delta_ctrl = resample_action_sequence(delta_actions, src_len, seq_len_eff)
        dim_pad_ctrl = resample_bool_sequence(dim_pad, src_len, seq_len_eff)

        native_pad = torch.zeros(src_len, dtype=torch.bool)
        if padded:
            native_pad[-1] = True
        pad_ctrl = resample_bool_sequence(native_pad, src_len, seq_len_eff)

        out: Dict[str, Any] = {
            "dataset": native_frames[0]["dataset"],
            "idx": native_frames[0]["idx"],
            "task": native_frames[0]["task"],
            "images": {"ego_view": images_out},
            "image_is_pad": pad_ctrl[video_idx].clone(),
            "action": action_ctrl,
            "action_is_pad": pad_ctrl.clone(),
            "action_dim_is_pad": dim_pad_ctrl,
            "control_fps": self.control_fps,
            "action_video_freq_ratio": self.action_video_freq_ratio,
            "video_control_indices": video_idx,
        }
        if wrist_out is not None:
            out["images"]["wrist_view"] = wrist_out
        if robot_ctrl is not None:
            out["robot_action_7d"] = robot_ctrl
        if proprio_ctrl is not None and delta_ctrl is not None:
            future_steps = self.future_action_steps
            if future_steps is None:
                future_steps = max(1, seq_len_eff - int(proprio_ctrl.shape[0]) + 1)
            past_w = int(seq_len_eff - int(future_steps))
            if past_w <= 0:
                raise ValueError(
                    f"Invalid proprio/delta split: seq_len={seq_len_eff}, future={future_steps}"
                )
            out["robot_proprio_7d"] = proprio_ctrl[:past_w]
            # VLA-Adapter: first future step pairs with last proprio (current) control index.
            future_end = int(seq_len_eff - 1)
            future_start = int(past_w - 1)
            if future_end <= future_start:
                raise ValueError(
                    f"Invalid future delta slice: start={future_start}, end={future_end}, "
                    f"seq_len={seq_len_eff}, past_w={past_w}"
                )
            out["robot_future_delta_7d"] = delta_ctrl[future_start:future_end]
        return out

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
        }
        if "wrist_view" in batch[0]["images"]:
            out["images"]["wrist_view"] = torch.stack(
                [b["images"]["wrist_view"] for b in batch]
            )
        if "robot_action_7d" in batch[0]:
            out["robot_action_7d"] = torch.stack([b["robot_action_7d"] for b in batch])
        if "robot_proprio_7d" in batch[0]:
            out["robot_proprio_7d"] = torch.stack([b["robot_proprio_7d"] for b in batch])
        if "robot_future_delta_7d" in batch[0]:
            out["robot_future_delta_7d"] = torch.stack([b["robot_future_delta_7d"] for b in batch])
        return out
