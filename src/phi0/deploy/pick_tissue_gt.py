"""Pick-tissue LeRobot episode → deploy unified GT (512-d) with State_t anchors."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from PIL import Image

from phi0.data.pick_tissue_unified import EGO_IMAGE_KEY, LEFT_WRIST_IMAGE_KEY
from phi0.data.simple_lerobot import _import_lerobot
from phi0.data.xperience_unified_gt import write_root_trans_local
from phi0.schema.unified_action_schema import D_UNIFIED


@dataclass(frozen=True)
class PickTissueEpisodeSpan:
    episode_index: int
    frame_start: int
    frame_count: int


def control_index_to_global_frame(
    frame_start: int,
    control_idx: int,
    *,
    native_fps: float,
    control_fps: float,
) -> int:
    offset = int(round(float(control_idx) * float(native_fps) / float(control_fps)))
    return int(frame_start) + offset


def _as_vec3(raw: Any) -> np.ndarray:
    return np.asarray(raw, dtype=np.float32).reshape(3)


def _row_to_rgb_uint8(raw: Any) -> np.ndarray:
    """LeRobot row image → HWC uint8 RGB."""
    if hasattr(raw, "detach"):
        t = raw.detach().cpu()
        if t.ndim == 3 and t.shape[0] in {1, 3}:
            arr = t.permute(1, 2, 0).numpy()
        else:
            arr = t.numpy()
    else:
        arr = np.asarray(raw)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        arr = np.asarray(Image.fromarray(arr).convert("RGB"))
    return arr


def _as_unified(raw: Any) -> np.ndarray:
    out = np.asarray(raw, dtype=np.float32).reshape(D_UNIFIED)
    if out.shape != (D_UNIFIED,):
        raise ValueError(f"expected unified dim {D_UNIFIED}, got {out.shape}")
    return out


class PickTissueGtReader:
    """Frame-level GT from pick-tissue unified LeRobot parquet."""

    def __init__(self, *, root_dir: str | Path, repo_id: str):
        self.root_dir = Path(root_dir)
        self.repo_id = str(repo_id)
        dataset_path = self.root_dir / self.repo_id
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"pick-tissue dataset not found: {dataset_path}")
        self._dataset_path = dataset_path
        LeRobotDataset, LeRobotDatasetMetadata = _import_lerobot()
        self._meta = LeRobotDatasetMetadata(self.repo_id, str(dataset_path))
        self._dataset = LeRobotDataset(self.repo_id, root=str(dataset_path))
        self._native_fps = float(self._meta.fps)

    @property
    def native_fps(self) -> float:
        return self._native_fps

    def episode_span(self, episode_index: int) -> PickTissueEpisodeSpan:
        ep = self._meta.episodes[int(episode_index)]
        return PickTissueEpisodeSpan(
            episode_index=int(episode_index),
            frame_start=int(ep["dataset_from_index"]),
            frame_count=int(ep["length"]),
        )

    def _clamp_global(self, global_frame: int, span: PickTissueEpisodeSpan) -> int:
        last = span.frame_start + span.frame_count - 1
        return int(min(max(int(global_frame), span.frame_start), last))

    def read_target_root(self, global_frame: int, span: PickTissueEpisodeSpan) -> np.ndarray:
        row = self._dataset[self._clamp_global(global_frame, span)]
        return _as_vec3(row["target_root_trans_world"])

    def read_camera_rgb(
        self,
        global_frame: int,
        span: PickTissueEpisodeSpan,
        *,
        key: str = EGO_IMAGE_KEY,
    ) -> np.ndarray:
        row = self._dataset[self._clamp_global(global_frame, span)]
        if key not in row:
            raise KeyError(f"camera key missing in dataset row: {key!r}")
        return _row_to_rgb_uint8(row[key])

    def read_ego_wrist_pair(
        self,
        global_frame: int,
        span: PickTissueEpisodeSpan,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.read_camera_rgb(global_frame, span, key=EGO_IMAGE_KEY),
            self.read_camera_rgb(global_frame, span, key=LEFT_WRIST_IMAGE_KEY),
        )

    def read_repacked_action(
        self,
        span: PickTissueEpisodeSpan,
        *,
        control_idx: int,
        anchor_control: int,
        native_fps: float,
        control_fps: float,
    ) -> np.ndarray:
        """Unified action repacked relative to ``anchor_control`` (matches training clip)."""
        global_t = control_index_to_global_frame(
            span.frame_start, control_idx, native_fps=native_fps, control_fps=control_fps
        )
        global_anchor = control_index_to_global_frame(
            span.frame_start, anchor_control, native_fps=native_fps, control_fps=control_fps
        )
        row_t = self._dataset[self._clamp_global(global_t, span)]
        anchor_root = self.read_target_root(global_anchor, span)
        target_root = _as_vec3(row_t["target_root_trans_world"])
        return write_root_trans_local(_as_unified(row_t["unified_action"]), target_root - anchor_root)

    def pack_deploy_frame(
        self,
        span: PickTissueEpisodeSpan,
        *,
        control_idx: int,
        state_control_idx: int,
        native_fps: float,
        control_fps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(d_raw, anchor_root)`` for deploy / FK (State_t at ``state_control_idx``)."""
        global_t = control_index_to_global_frame(
            span.frame_start, control_idx, native_fps=native_fps, control_fps=control_fps
        )
        global_state = control_index_to_global_frame(
            span.frame_start, state_control_idx, native_fps=native_fps, control_fps=control_fps
        )
        row_t = self._dataset[self._clamp_global(global_t, span)]
        anchor_root = self.read_target_root(global_state, span)
        target_root = _as_vec3(row_t["target_root_trans_world"])
        d_raw = write_root_trans_local(_as_unified(row_t["unified_action"]), target_root - anchor_root)
        return d_raw, anchor_root


@lru_cache(maxsize=4)
def _cached_reader(root_dir: str, repo_id: str) -> PickTissueGtReader:
    return PickTissueGtReader(root_dir=root_dir, repo_id=repo_id)


def reader_from_data_cfg(data_cfg: Mapping[str, Any]) -> PickTissueGtReader:
    root = str(data_cfg.get("pick_tissue_root", "./data"))
    repo = str(data_cfg.get("pick_tissue_repo_id", "pick_tissue_xperience_unified"))
    return _cached_reader(root, repo)


def episode_span_for_clip_item(
    reader: PickTissueGtReader,
    clip_item: Mapping[str, Any],
) -> PickTissueEpisodeSpan:
    return reader.episode_span(int(clip_item["idx"]))


def clip_dataset_index_for_episode(
    eval_ds: Any,
    episode_index: int,
    *,
    data_cfg: Mapping[str, Any],
) -> int:
    """Map LeRobot ``episode_index`` -> ``PickTissueUnifiedClipDataset`` row index."""
    reader = reader_from_data_cfg(data_cfg)
    target = int(reader.episode_span(int(episode_index)).frame_start)
    indices = getattr(eval_ds, "_indices", None)
    if indices is None:
        raise TypeError("eval_ds missing _indices (expected PickTissueUnifiedClipDataset)")
    for i, gidx in enumerate(indices):
        if int(gidx) == target:
            return int(i)
    raise KeyError(
        f"no training clip at episode_index={episode_index} frame_start={target}"
    )
