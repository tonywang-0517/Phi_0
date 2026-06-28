"""Video decode helpers for training loaders."""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def preload_mp4_frames(
    video_path: str | Path | None,
    image_size: Tuple[int, int],
    *,
    max_frames: Optional[int] = None,
) -> Optional[List[torch.Tensor]]:
    """Read MP4 once into a list of [C,H,W] float tensors in [0,1] (single-file datasets)."""
    if video_path is None:
        return None
    path = Path(video_path)
    if not path.is_file():
        return None

    import cv2

    h, w = image_size
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.warning("VideoFrameCache: failed to open %s", path)
        return None

    frames: List[torch.Tensor] = []
    try:
        while True:
            if max_frames is not None and len(frames) >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            frames.append(torch.from_numpy(frame).permute(2, 0, 1).contiguous().float() / 255.0)
    finally:
        cap.release()

    if not frames:
        logger.warning("VideoFrameCache: no frames read from %s", path)
        return None

    logger.info("VideoFrameCache: preloaded %d frames from %s", len(frames), path.name)
    return frames


def _chw_to_uint8(chw: torch.Tensor, image_size: Tuple[int, int] | None) -> torch.Tensor:
    t = chw.detach().cpu()
    if t.dtype != torch.uint8:
        if t.is_floating_point() and float(t.max()) <= 1.5:
            t = (t.clamp(0.0, 1.0) * 255.0).round()
        t = t.clamp(0, 255).to(torch.uint8)
    if image_size is not None:
        h, w = image_size
        if t.shape[-2] != h or t.shape[-1] != w:
            t = (
                torch.nn.functional.interpolate(
                    t.float().unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
                )
                .squeeze(0)
                .round()
                .to(torch.uint8)
            )
    return t.contiguous()


class LeRobotTimestampFrameCache:
    """LeRobot wrapper: decode only requested timestamps; bounded uint8 frame LRU per worker."""

    def __init__(
        self,
        base: Any,
        *,
        max_frames: int = 2048,
        image_size: Tuple[int, int] | None = None,
    ) -> None:
        self._base = base
        self._max_frames = max(64, int(max_frames))
        self._image_size = (
            (int(image_size[0]), int(image_size[1])) if image_size is not None else None
        )
        self._frames: OrderedDict[tuple[int, str, int], torch.Tensor] = OrderedDict()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self._base.hf_dataset[idx]
        ep_idx = int(item["episode_index"].item())
        query_indices = None
        if self._base.delta_indices is not None:
            query_indices, padding = self._base._get_query_indices(idx, ep_idx)
            query_result = self._base._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val
        if len(self._base.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._base._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos_cached(query_timestamps, ep_idx)
            item = {**video_frames, **item}
        if self._base.image_transforms is not None:
            for cam in self._base.meta.camera_keys:
                item[cam] = self._base.image_transforms(item[cam])
        task_idx = int(item["task_index"].item())
        item["task"] = self._base.meta.tasks[task_idx]
        return item

    def _query_videos_cached(
        self, query_timestamps: dict[str, list[float]], ep_idx: int
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for vid_key, query_ts in query_timestamps.items():
            out[vid_key] = self._frames_for_timestamps(ep_idx, vid_key, query_ts)
        return out

    def _frames_for_timestamps(
        self, ep_idx: int, vid_key: str, query_ts: list[float]
    ) -> torch.Tensor:
        fps = float(self._base.meta.fps)
        frame_ids = [int(round(float(ts) * fps)) for ts in query_ts]
        picked: list[torch.Tensor | None] = [None] * len(query_ts)
        missing_ts: list[float] = []
        missing_slots: list[int] = []
        missing_keys: list[tuple[int, str, int]] = []

        for slot, (ts, fi) in enumerate(zip(query_ts, frame_ids)):
            key = (ep_idx, vid_key, fi)
            cached = self._frames.get(key)
            if cached is not None:
                self._frames.move_to_end(key)
                picked[slot] = cached.float() / 255.0
            else:
                missing_ts.append(float(ts))
                missing_slots.append(slot)
                missing_keys.append(key)

        if missing_ts:
            from lerobot.datasets.video_utils import decode_video_frames

            video_path = self._base.root / self._base.meta.get_video_file_path(ep_idx, vid_key)
            decoded = decode_video_frames(
                video_path, missing_ts, self._base.tolerance_s, self._base.video_backend
            )
            for j, slot in enumerate(missing_slots):
                chw_u8 = _chw_to_uint8(decoded[j], self._image_size)
                key = missing_keys[j]
                while len(self._frames) >= self._max_frames:
                    self._frames.popitem(last=False)
                self._frames[key] = chw_u8
                picked[slot] = chw_u8.float() / 255.0

        assert all(p is not None for p in picked)
        if len(picked) == 1:
            return picked[0]  # type: ignore[return-value]
        return torch.stack(picked)  # type: ignore[arg-type]


class LeRobotPredecodedVideo:
    """LeRobot wrapper that reads frames from offline ``videos_decoded`` store."""

    def __init__(self, base: Any, store: Any) -> None:
        self._base = base
        self._store = store

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self._base.hf_dataset[idx]
        ep_idx = int(item["episode_index"].item())
        query_indices = None
        if self._base.delta_indices is not None:
            query_indices, padding = self._base._get_query_indices(idx, ep_idx)
            query_result = self._base._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val
        if len(self._base.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._base._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos_predecoded(query_timestamps, ep_idx)
            item = {**video_frames, **item}
        if self._base.image_transforms is not None:
            for cam in self._base.meta.camera_keys:
                item[cam] = self._base.image_transforms(item[cam])
        task_idx = int(item["task_index"].item())
        item["task"] = self._base.meta.tasks[task_idx]
        return item

    def _query_videos_predecoded(
        self, query_timestamps: dict[str, list[float]], ep_idx: int
    ) -> dict[str, torch.Tensor]:
        fps = float(self._base.meta.fps)
        out: dict[str, torch.Tensor] = {}
        for vid_key, query_ts in query_timestamps.items():
            frame_ids = [int(round(float(ts) * fps)) for ts in query_ts]
            out[vid_key] = self._store.get_frames_tensor(ep_idx, vid_key, frame_ids)
        return out
