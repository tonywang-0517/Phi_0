"""Pick-tissue GT camera frames for eval (predecoded model input or raw MP4 letterbox)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping

import cv2
import numpy as np

from phi0.deploy.pick_tissue_gt import PickTissueEpisodeSpan
from phi0.data.pick_tissue_unified import EGO_IMAGE_KEY, LEFT_WRIST_IMAGE_KEY
from phi0.data.predecoded_video import (
    PredecodedVideoMeta,
    episode_npy_path,
    load_episode_frames_mmap,
    meta_path,
    predecoded_root,
    read_episodes_jsonl,
)
from phi0.data.psi0_image import read_lerobot_video_hw
from phi0.models.vlm.preprocess import make_psi0_vlm_image_transform
from phi0.paths import workspace_root

ViewFitMode = Literal["letterbox_raw", "model_input"]


def letterbox_rgb(img: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Scale to fit inside ``target_hw`` (H,W), pad with black (no crop)."""
    th, tw = int(target_hw[0]), int(target_hw[1])
    rgb = np.asarray(img)[..., :3]
    ih, iw = rgb.shape[:2]
    scale = min(tw / iw, th / ih)
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((th, tw, 3), dtype=np.uint8)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def _default_image_size() -> tuple[int, int]:
    return (180, 320)


class PickTissuePredecodedReader:
    """Ego + wrist GT from predecoded npy and/or source MP4."""

    def __init__(
        self,
        *,
        root_dir: str | Path,
        repo_id: str,
        image_size: tuple[int, int] | None = None,
        view_fit: ViewFitMode = "letterbox_raw",
        panel_size: tuple[int, int] | None = None,
    ):
        self.dataset_root = Path(root_dir) / str(repo_id)
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"pick-tissue dataset not found: {self.dataset_root}")
        self.image_size = tuple(image_size or _default_image_size())
        self.view_fit: ViewFitMode = view_fit
        self.panel_size = tuple(panel_size or self.image_size)
        self._native_image_size = read_lerobot_video_hw(self.dataset_root, EGO_IMAGE_KEY)
        self.store_root = predecoded_root(self.dataset_root, self._native_image_size)
        self._vlm_transform = make_psi0_vlm_image_transform(
            self.panel_size, img_aug=False, training=False
        )
        if meta_path(self.store_root).is_file():
            raw = json.loads(meta_path(self.store_root).read_text(encoding="utf-8"))
            self._store_meta = PredecodedVideoMeta.from_dict(raw)
        else:
            info = json.loads((self.dataset_root / "meta/info.json").read_text(encoding="utf-8"))
            self._store_meta = PredecodedVideoMeta(
                version=1,
                image_size=self._native_image_size,
                layout="THWC",
                dtype="uint8",
                fps=float(info["fps"]),
                video_keys=(EGO_IMAGE_KEY, LEFT_WRIST_IMAGE_KEY),
                total_episodes=0,
                backend="mp4",
            )
        self._episodes = read_episodes_jsonl(self.dataset_root / "meta")
        info = json.loads((self.dataset_root / "meta/info.json").read_text(encoding="utf-8"))
        self._chunk_size = int(info.get("chunks_size", 1000))
        self._frame_cache: dict[tuple[int, str], np.ndarray] = {}
        self._mp4_caps: dict[str, cv2.VideoCapture] = {}

    @property
    def native_fps(self) -> float:
        return float(self._store_meta.fps)

    def episode_span(self, episode_index: int) -> PickTissueEpisodeSpan:
        ep = self._episodes[int(episode_index)]
        return PickTissueEpisodeSpan(
            episode_index=int(episode_index),
            frame_start=int(ep["dataset_from_index"]),
            frame_count=int(ep["length"]),
        )

    def _clamp_local(self, global_frame: int, span: PickTissueEpisodeSpan) -> int:
        last = span.frame_count - 1
        local = int(global_frame) - int(span.frame_start)
        return int(min(max(local, 0), last))

    def _mp4_path(self, episode_index: int, video_key: str) -> Path:
        chunk = int(episode_index) // self._chunk_size
        return (
            self.dataset_root
            / "videos"
            / f"chunk-{chunk:03d}"
            / video_key
            / f"episode_{int(episode_index):06d}.mp4"
        )

    def _read_mp4_frame(self, episode_index: int, video_key: str, local_frame: int) -> np.ndarray:
        path = self._mp4_path(episode_index, video_key)
        if not path.is_file():
            raise FileNotFoundError(f"missing source video: {path}")
        key = str(path)
        cap = self._mp4_caps.get(key)
        if cap is None:
            cap = cv2.VideoCapture(key)
            if not cap.isOpened():
                raise RuntimeError(f"failed to open {path}")
            self._mp4_caps[key] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(local_frame))
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"failed to read frame {local_frame} from {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _episode_video(self, episode_index: int, video_key: str) -> np.ndarray:
        key = (int(episode_index), str(video_key))
        if key not in self._frame_cache:
            path = episode_npy_path(
                self.store_root,
                episode_index=int(episode_index),
                video_key=str(video_key),
            )
            self._frame_cache[key] = load_episode_frames_mmap(path)
        return self._frame_cache[key]

    def read_camera_rgb(
        self,
        global_frame: int,
        span: PickTissueEpisodeSpan,
        *,
        key: str = EGO_IMAGE_KEY,
    ) -> np.ndarray:
        local = self._clamp_local(global_frame, span)
        if self.view_fit == "model_input":
            raw = self._read_mp4_frame(span.episode_index, key, local)
            from PIL import Image

            out = self._vlm_transform(Image.fromarray(raw))
            return np.asarray(out, dtype=np.uint8)
        raw = self._read_mp4_frame(span.episode_index, key, local)
        return letterbox_rgb(raw, self.panel_size)

    def read_ego_wrist_pair(
        self,
        global_frame: int,
        span: PickTissueEpisodeSpan,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.read_camera_rgb(global_frame, span, key=EGO_IMAGE_KEY),
            self.read_camera_rgb(global_frame, span, key=LEFT_WRIST_IMAGE_KEY),
        )

    def close(self) -> None:
        for cap in self._mp4_caps.values():
            cap.release()
        self._mp4_caps.clear()


@lru_cache(maxsize=8)
def _cached_predecoded_reader(
    root_dir: str,
    repo_id: str,
    view_fit: str,
) -> PickTissuePredecodedReader:
    return PickTissuePredecodedReader(
        root_dir=root_dir,
        repo_id=repo_id,
        view_fit=view_fit,  # type: ignore[arg-type]
    )


def reader_from_meta(
    meta: Mapping[str, Any],
    *,
    view_fit: ViewFitMode = "letterbox_raw",
) -> PickTissuePredecodedReader:
    root = str(
        meta.get("pick_tissue_root", f"{workspace_root()}/Isaac-GR00T/data")
    )
    repo = str(meta.get("pick_tissue_repo_id", "pick_tissue_xperience_unified"))
    return _cached_predecoded_reader(root, repo, view_fit)
