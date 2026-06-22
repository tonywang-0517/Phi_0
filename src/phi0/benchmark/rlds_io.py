"""Lightweight RLDS (tf.train.Example) reader — no tensorflow_datasets required."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
from PIL import Image

BenchmarkName = Literal["libero", "calvin"]


@dataclass(frozen=True)
class RldsStep:
    rgb_static: np.ndarray | None
    rgb_gripper: np.ndarray | None
    state: np.ndarray
    action: np.ndarray
    language: str


@dataclass(frozen=True)
class RldsEpisode:
    steps: list[RldsStep]
    episode_id: str = ""


def _require_tensorflow():
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "RLDS bridge training requires tensorflow. "
            "Install with: pip install tensorflow-cpu"
        ) from exc
    tf.get_logger().setLevel("ERROR")
    return tf


def _decode_image(jpeg_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(jpeg_bytes)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _parse_example(raw: bytes, benchmark: BenchmarkName) -> RldsEpisode:
    tf = _require_tensorflow()
    ex = tf.train.Example()
    ex.ParseFromString(raw)
    feat = ex.features.feature

    def n_steps() -> int:
        return len(feat["steps/is_first"].int64_list.value)

    t = n_steps()
    actions = np.asarray(feat["steps/action"].float_list.value, dtype=np.float32).reshape(t, 7)
    langs = [x.decode("utf-8", errors="replace") for x in feat["steps/language_instruction"].bytes_list.value]

    if benchmark == "libero":
        states = np.asarray(feat["steps/observation/state"].float_list.value, dtype=np.float32).reshape(t, -1)
        static_key, grip_key = "steps/observation/image", "steps/observation/wrist_image"
    else:
        states = np.asarray(feat["steps/observation/state"].float_list.value, dtype=np.float32).reshape(t, -1)
        static_key, grip_key = "steps/observation/rgb_static", "steps/observation/rgb_gripper"

    static_bytes = list(feat[static_key].bytes_list.value)
    grip_bytes = list(feat[grip_key].bytes_list.value)
    ep_id = ""
    if "episode_metadata/episode_id" in feat:
        ep_id = str(feat["episode_metadata/episode_id"].int64_list.value[0])
    elif "episode_metadata/file_path" in feat:
        ep_id = feat["episode_metadata/file_path"].bytes_list.value[0].decode("utf-8", errors="replace")

    steps: list[RldsStep] = []
    for i in range(t):
        steps.append(
            RldsStep(
                rgb_static=_decode_image(static_bytes[i]),
                rgb_gripper=_decode_image(grip_bytes[i]),
                state=states[i],
                action=actions[i],
                language=langs[i],
            )
        )
    return RldsEpisode(steps=steps, episode_id=ep_id)


def iter_rlds_shards(
    shard_glob: str | Path,
    *,
    benchmark: BenchmarkName,
    max_shards: int | None = None,
) -> Iterator[RldsEpisode]:
    paths = sorted(Path(p) for p in Path(shard_glob).parent.glob(Path(shard_glob).name))
    if max_shards is not None:
        paths = paths[: int(max_shards)]
    tf = _require_tensorflow()
    for path in paths:
        for raw in tf.data.TFRecordDataset(str(path)):
            yield _parse_example(bytes(raw.numpy()), benchmark)


def libero_train_shard_glob(suite: str, data_root: Path | None = None) -> str:
    """Return glob for LIBERO train shards (handles liber_o10 typo in OpenVLA RLDS)."""
    from phi0.benchmark.paths import libero_rlds_dir

    root = libero_rlds_dir(suite) if data_root is None else data_root
    base = suite.replace("_no_noops", "")
    patterns = [f"{base}-train.tfrecord-*", "liber_o10-train.tfrecord-*"]
    for pat in patterns:
        if list(root.glob(pat)):
            return str(root / pat)
    return str(root / f"{base}-train.tfrecord-*")


def calvin_shard_glob(data_root: Path | None = None) -> str:
    from phi0.benchmark.paths import CALVIN_RLDS_ROOT

    root = CALVIN_RLDS_ROOT if data_root is None else data_root
    return str(root / "calvin_abc-train.tfrecord-*")
