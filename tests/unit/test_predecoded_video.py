"""Unit tests for offline LeRobot video predecode (run before long batch jobs)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phi0.data.predecoded_video import (
    PREDECODED_VERSION,
    PredecodedVideoMeta,
    decode_mp4_to_uint8_thwc,
    episode_npy_path,
    frames_at_indices,
    load_episode_frames_mmap,
    predecode_one_episode,
    predecoded_root,
    save_episode_frames,
    validate_predecoded_store,
    write_store_meta,
)


def _write_synthetic_mp4(path: Path, *, num_frames: int = 12, h: int = 48, w: int = 64) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 50.0, (w, h))
    assert writer.isOpened()
    try:
        for i in range(num_frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[..., 0] = i
            frame[..., 1] = 100 + i
            frame[..., 2] = 200
            writer.write(frame)
    finally:
        writer.release()


def _write_minimal_lerobot_tree(root: Path, *, num_episodes: int = 2) -> tuple[int, int]:
    """Minimal LeRobot layout with synthetic MP4s for predecode tests."""
    h, w, fps = 48, 64, 50
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    lengths = [10, 7][:num_episodes]
    episodes = []
    offset = 0
    for ep, length in enumerate(lengths):
        for key in ("observation.images.ego_view", "observation.images.left_wrist"):
            mp4 = (
                root
                / "videos"
                / "chunk-000"
                / key
                / f"episode_{ep:06d}.mp4"
            )
            _write_synthetic_mp4(mp4, num_frames=length, h=h, w=w)
        episodes.append(
            {
                "episode_index": ep,
                "tasks": ["pick tissue"],
                "length": length,
                "dataset_from_index": offset,
                "dataset_to_index": offset + length - 1,
            }
        )
        offset += length
    (meta / "episodes.jsonl").write_text(
        "\n".join(json.dumps(r) for r in episodes) + "\n",
        encoding="utf-8",
    )
    (meta / "info.json").write_text(
        json.dumps({"fps": fps, "chunks_size": 1000}, indent=2) + "\n",
        encoding="utf-8",
    )
    return h, w


def test_decode_synthetic_mp4_native(tmp_path: Path):
    mp4 = tmp_path / "clip_native.mp4"
    _write_synthetic_mp4(mp4, num_frames=5, h=48, w=64)
    out = decode_mp4_to_uint8_thwc(mp4, None, fps=50.0, expected_length=5, backend="cv2")
    assert out.shape == (5, 48, 64, 3)


def test_decode_synthetic_mp4_cv2_roundtrip(tmp_path: Path):
    mp4 = tmp_path / "clip.mp4"
    _write_synthetic_mp4(mp4, num_frames=8, h=32, w=40)
    out = decode_mp4_to_uint8_thwc(mp4, (32, 40), fps=50.0, expected_length=8, backend="cv2")
    assert out.shape == (8, 32, 40, 3)
    assert out.dtype == np.uint8
    assert out.max() <= 255 and out.min() >= 0


def test_save_load_and_frame_lookup(tmp_path: Path):
    frames = np.stack(
        [np.full((24, 32, 3), i, dtype=np.uint8) for i in range(5)],
        axis=0,
    )
    path = tmp_path / "episode_000000.npy"
    save_episode_frames(path, frames)
    loaded = load_episode_frames_mmap(path)
    assert loaded.shape == (5, 24, 32, 3)
    t = frames_at_indices(loaded, [0, 2, 4])
    assert t.shape == (3, 3, 24, 32)
    assert pytest.approx(float(t[1, 0, 0, 0].item()), rel=1e-5) == 2.0 / 255.0


def test_predecode_episode_matches_source_length(tmp_path: Path):
    h, w = _write_minimal_lerobot_tree(tmp_path, num_episodes=1)
    image_size = (h, w)
    store = predecoded_root(tmp_path, image_size)
    rec = predecode_one_episode(
        dataset_root=tmp_path,
        store_root=store,
        episode_index=0,
        episode_length=10,
        video_key="observation.images.ego_view",
        image_size=image_size,
        fps=50.0,
        chunks_size=1000,
        backend="cv2",
        tolerance_s=0.02,
        overwrite=True,
    )
    assert rec.length == 10
    arr = load_episode_frames_mmap(store / rec.path)
    assert arr.shape == (10, h, w, 3)
    assert not np.array_equal(arr[0], arr[-1])


def test_validate_store_catches_missing_and_wrong_length(tmp_path: Path):
    h, w = _write_minimal_lerobot_tree(tmp_path, num_episodes=2)
    image_size = (h, w)
    store = predecoded_root(tmp_path, image_size)
    store.mkdir(parents=True)
    write_store_meta(
        store,
        PredecodedVideoMeta(
            version=PREDECODED_VERSION,
            image_size=image_size,
            layout="THWC",
            dtype="uint8",
            fps=50.0,
            video_keys=("observation.images.ego_view", "observation.images.left_wrist"),
            total_episodes=2,
            backend="cv2",
        ),
    )
    errs = validate_predecoded_store(tmp_path, store)
    assert errs

    predecode_one_episode(
        dataset_root=tmp_path,
        store_root=store,
        episode_index=0,
        episode_length=10,
        video_key="observation.images.ego_view",
        image_size=image_size,
        fps=50.0,
        chunks_size=1000,
        backend="cv2",
        tolerance_s=0.02,
        overwrite=True,
    )
    errs = validate_predecoded_store(tmp_path, store)
    assert any("missing" in e for e in errs)

    for ep, length in ((0, 10), (1, 7)):
        for key in ("observation.images.ego_view", "observation.images.left_wrist"):
            predecode_one_episode(
                dataset_root=tmp_path,
                store_root=store,
                episode_index=ep,
                episode_length=length,
                video_key=key,
                image_size=image_size,
                fps=50.0,
                chunks_size=1000,
                backend="cv2",
                tolerance_s=0.02,
                overwrite=True,
            )
    assert validate_predecoded_store(tmp_path, store) == []


def test_episode_npy_path_uses_safe_key():
    store = Path("/data/videos_decoded/180x320")
    p = episode_npy_path(
        store,
        episode_index=5,
        video_key="observation.images.ego_view",
    )
    assert p == store / "chunk-000" / "observation.images.ego_view" / "episode_000005.npy"


@pytest.mark.skipif(
    not Path(
        "/mnt/data2/wpy/workspace/Isaac-GR00T/data/pick_tissue_xperience_unified/videos/chunk-000/observation.images.ego_view/episode_000000.mp4"
    ).is_file(),
    reason="pick-tissue dataset not available",
)
def test_real_episode_cv2_matches_episodes_jsonl_length():
    root = Path("/mnt/data2/wpy/workspace/Isaac-GR00T/data/pick_tissue_xperience_unified")
    with open(root / "meta" / "episodes.jsonl", encoding="utf-8") as f:
        row = json.loads(f.readline())
    length = int(row["length"])
    mp4 = root / "videos/chunk-000/observation.images.ego_view/episode_000000.mp4"
    out = decode_mp4_to_uint8_thwc(
        mp4,
        (180, 320),
        fps=50.0,
        expected_length=length,
        backend="cv2",
    )
    assert out.shape == (length, 180, 320, 3)
