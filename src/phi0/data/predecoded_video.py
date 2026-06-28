"""Offline LeRobot MP4 → uint8 THWC frame stores for training (mmap-friendly)."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

PREDECODED_VERSION = 1
LAYOUT = "THWC"
DTYPE = "uint8"


@dataclass(frozen=True)
class PredecodedVideoMeta:
    version: int
    image_size: tuple[int, int]
    layout: str
    dtype: str
    fps: float
    video_keys: tuple[str, ...]
    total_episodes: int
    backend: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["image_size"] = list(self.image_size)
        d["video_keys"] = list(self.video_keys)
        return d

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PredecodedVideoMeta":
        return cls(
            version=int(raw["version"]),
            image_size=(int(raw["image_size"][0]), int(raw["image_size"][1])),
            layout=str(raw.get("layout", LAYOUT)),
            dtype=str(raw.get("dtype", DTYPE)),
            fps=float(raw["fps"]),
            video_keys=tuple(str(k) for k in raw["video_keys"]),
            total_episodes=int(raw["total_episodes"]),
            backend=str(raw.get("backend", "cv2")),
        )


@dataclass(frozen=True)
class PredecodedEpisodeRecord:
    episode_index: int
    length: int
    video_key: str
    path: str
    shape: tuple[int, int, int, int]


def predecoded_root(dataset_root: Path, image_size: tuple[int, int]) -> Path:
    h, w = int(image_size[0]), int(image_size[1])
    return Path(dataset_root) / "videos_decoded" / f"{h}x{w}"


def meta_path(store_root: Path) -> Path:
    return store_root / "meta.json"


def episode_npy_path(
    store_root: Path,
    *,
    episode_index: int,
    video_key: str,
    chunk_size: int = 1000,
) -> Path:
    chunk = episode_index // chunk_size
    safe_key = video_key.replace("/", "__")
    return (
        store_root
        / f"chunk-{chunk:03d}"
        / safe_key
        / f"episode_{episode_index:06d}.npy"
    )


def decode_mp4_to_uint8_thwc(
    video_path: Path,
    image_size: tuple[int, int] | None,
    *,
    fps: float,
    expected_length: int | None = None,
    backend: str = "cv2",
    tolerance_s: float = 0.04,
) -> np.ndarray:
    """Decode one episode MP4 to ``(T,H,W,C)`` uint8 RGB.

    ``image_size=None`` keeps source resolution (Psi0: resize at VLM transform time).
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    if image_size is None:
        if backend != "cv2":
            logger.warning("native decode ignores backend=%s; using cv2", backend)
        return _decode_mp4_cv2_thwc(path, None, expected_length=expected_length)

    h, w = int(image_size[0]), int(image_size[1])
    # ponytail: offline full-episode decode uses cv2 sequential read (~100x faster than
    # torchcodec timestamp batch on CPU). torchcodec stays for training random access.
    if backend == "torchcodec":
        try:
            from lerobot.datasets.video_utils import decode_video_frames

            length = expected_length
            if length is None:
                length = _mp4_frame_count_cv2(path)
            timestamps = [i / float(fps) for i in range(length)]
            decoded = decode_video_frames(path, timestamps, tolerance_s, "torchcodec")
            frames = []
            for i in range(int(decoded.shape[0])):
                frames.append(_chw_to_uint8_hwc(decoded[i], (h, w)))
            return np.stack(frames, axis=0)
        except Exception as exc:
            logger.warning("torchcodec decode failed for %s (%s); falling back to cv2", path.name, exc)

    return _decode_mp4_cv2_thwc(path, (h, w), expected_length=expected_length)


def _mp4_frame_count_cv2(video_path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {video_path}")
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count > 0:
            return count
        n = 0
        while cap.read()[0]:
            n += 1
        return n
    finally:
        cap.release()


def _decode_mp4_cv2_thwc(
    video_path: Path,
    image_size: tuple[int, int] | None,
    *,
    expected_length: int | None,
) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {video_path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if image_size is not None:
                h, w = image_size
                if rgb.shape[0] != h or rgb.shape[1] != w:
                    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            frames.append(np.ascontiguousarray(rgb, dtype=np.uint8))
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    out = np.stack(frames, axis=0)
    if expected_length is not None and out.shape[0] != int(expected_length):
        raise ValueError(
            f"{video_path.name}: decoded {out.shape[0]} frames, expected {expected_length}"
        )
    return out


def _chw_to_uint8_hwc(chw: torch.Tensor, image_size: tuple[int, int]) -> np.ndarray:
    h, w = image_size
    t = chw.detach().cpu()
    if t.dtype != torch.uint8:
        if t.is_floating_point() and float(t.max()) <= 1.5:
            t = (t.clamp(0.0, 1.0) * 255.0).round()
        arr = t.clamp(0, 255).to(torch.uint8)
    else:
        arr = t
    if arr.ndim != 3:
        raise ValueError(f"expected CHW, got {tuple(arr.shape)}")
    if arr.shape[0] == 3:
        hwc = arr.permute(1, 2, 0)
    else:
        hwc = arr
    if hwc.shape[0] != h or hwc.shape[1] != w:
        hwc = (
            torch.nn.functional.interpolate(
                hwc.permute(2, 0, 1).float().unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
            .round()
            .to(torch.uint8)
        )
    return np.ascontiguousarray(hwc.numpy(), dtype=np.uint8)


def save_episode_frames(path: Path, frames_thwc: np.ndarray) -> None:
    if frames_thwc.dtype != np.uint8 or frames_thwc.ndim != 4 or frames_thwc.shape[-1] != 3:
        raise ValueError(f"expected (T,H,W,3) uint8, got {frames_thwc.dtype} {frames_thwc.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, frames_thwc)


def load_episode_frames_mmap(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.dtype != np.uint8 or arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"invalid predecoded file {path}: {arr.dtype} {arr.shape}")
    return arr


def frames_at_indices(
    frames_thwc: np.ndarray,
    frame_indices: Sequence[int],
) -> torch.Tensor:
    """Return ``(T,C,H,W)`` float32 in ``[0,1]`` for training."""
    t = len(frames_thwc)
    idx = [max(0, min(t - 1, int(i))) for i in frame_indices]
    if len(idx) == 1:
        chw = torch.from_numpy(np.asarray(frames_thwc[idx[0]])).permute(2, 0, 1).float() / 255.0
        return chw
    batch = np.asarray(frames_thwc[idx])
    return torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0


def read_episodes_jsonl(meta_dir: Path) -> list[dict[str, Any]]:
    path = meta_dir / "episodes.jsonl"
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_store_meta(store_root: Path, meta: PredecodedVideoMeta) -> None:
    store_root.mkdir(parents=True, exist_ok=True)
    meta_path(store_root).write_text(json.dumps(meta.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_store_meta(store_root: Path) -> PredecodedVideoMeta:
    raw = json.loads(meta_path(store_root).read_text(encoding="utf-8"))
    return PredecodedVideoMeta.from_dict(raw)


def predecode_one_episode(
    *,
    dataset_root: Path,
    store_root: Path,
    episode_index: int,
    episode_length: int,
    video_key: str,
    image_size: tuple[int, int] | None,
    fps: float,
    chunks_size: int,
    backend: str,
    tolerance_s: float,
    overwrite: bool,
) -> PredecodedEpisodeRecord:
    out_path = episode_npy_path(
        store_root,
        episode_index=episode_index,
        video_key=video_key,
        chunk_size=chunks_size,
    )
    if out_path.is_file() and not overwrite:
        arr = load_episode_frames_mmap(out_path)
        if arr.shape[0] != episode_length:
            raise ValueError(
                f"cached {out_path.name}: {arr.shape[0]} frames != meta length {episode_length}"
            )
        return PredecodedEpisodeRecord(
            episode_index=episode_index,
            length=arr.shape[0],
            video_key=video_key,
            path=str(out_path.relative_to(store_root)),
            shape=tuple(int(x) for x in arr.shape),
        )

    chunk = episode_index // chunks_size
    src = (
        dataset_root
        / "videos"
        / f"chunk-{chunk:03d}"
        / video_key
        / f"episode_{episode_index:06d}.mp4"
    )
    frames = decode_mp4_to_uint8_thwc(
        src,
        image_size,
        fps=fps,
        expected_length=episode_length,
        backend=backend,
        tolerance_s=tolerance_s,
    )
    save_episode_frames(out_path, frames)
    return PredecodedEpisodeRecord(
        episode_index=episode_index,
        length=int(frames.shape[0]),
        video_key=video_key,
        path=str(out_path.relative_to(store_root)),
        shape=tuple(int(x) for x in frames.shape),
    )


def validate_predecoded_store(
    dataset_root: Path,
    store_root: Path,
    *,
    video_keys: Iterable[str] | None = None,
    max_episodes: int | None = None,
) -> list[str]:
    """Return list of error strings; empty means OK."""
    errors: list[str] = []
    if not meta_path(store_root).is_file():
        return [f"missing {meta_path(store_root)}"]
    store_meta = load_store_meta(store_root)
    keys = tuple(video_keys) if video_keys is not None else store_meta.video_keys
    episodes = read_episodes_jsonl(Path(dataset_root) / "meta")
    if max_episodes is not None:
        episodes = episodes[: int(max_episodes)]

    for row in episodes:
        ep = int(row["episode_index"])
        length = int(row["length"])
        for key in keys:
            npy = episode_npy_path(store_root, episode_index=ep, video_key=key)
            if not npy.is_file():
                errors.append(f"missing {npy}")
                continue
            try:
                arr = load_episode_frames_mmap(npy)
            except Exception as exc:
                errors.append(f"corrupt {npy}: {exc}")
                continue
            if arr.shape[0] != length:
                errors.append(f"{npy.name}: {arr.shape[0]} frames != episodes.jsonl length {length}")
            if tuple(arr.shape[1:3]) != store_meta.image_size:
                errors.append(
                    f"{npy.name}: shape {arr.shape} != (*,{store_meta.image_size[0]},{store_meta.image_size[1]},3)"
                )
    return errors


class PredecodedVideoStore:
    """Mmap reader for offline-decoded episode videos."""

    def __init__(self, store_root: Path, *, max_open_episodes: int = 128):
        self.store_root = Path(store_root)
        self.meta = load_store_meta(self.store_root)
        self._max_open = max(8, int(max_open_episodes))
        # ponytail: LRU cap per worker; unbounded _open OOMs on long random-episode training.
        self._open: OrderedDict[tuple[int, str], np.ndarray] = OrderedDict()

    def get_frames_tensor(
        self,
        episode_index: int,
        video_key: str,
        frame_indices: Sequence[int],
    ) -> torch.Tensor:
        key = (int(episode_index), str(video_key))
        arr = self._open.get(key)
        if arr is None:
            path = episode_npy_path(self.store_root, episode_index=episode_index, video_key=video_key)
            arr = load_episode_frames_mmap(path)
            self._open[key] = arr
            while len(self._open) > self._max_open:
                self._open.popitem(last=False)
        else:
            self._open.move_to_end(key)
        return frames_at_indices(arr, frame_indices)
