"""Convert Isaac-GR00T pick-tissue LeRobot to Phi_0 unified 512-d action format.

Per frame:
  0:346   SMPL-H from teleop (groot_unified_io)
  346:360 g1_gripper_joints_14 from action.wbc
  360:396 g1_body_qpos_36: observation.state dof29 + robot root quat + base_trans xyz

Also exports state_root_trans_world, target_root_trans_world, betas, ego + left_wrist video.
Root xyz from observation.base_trans (g1_debug base_trans_measured); legacy parquets fall back to smpl pelvis.
CODE_VERSION v2.8.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset, Features, Sequence, Value
from datasets.utils.logging import set_verbosity_error
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from phi0.data.action_stats import (  # noqa: E402
    masked_unified_action_stats,
    merge_masked_unified_action_stats,
)
from phi0.data.g1_qpos_teacher import (  # noqa: E402
    attach_g1_qpos_to_parquet_rows,
    attach_sonic_motion_token_to_parquet_rows,
)
from phi0.data.groot_unified_io import (  # noqa: E402
    STATS_SEMANTICS_PICK_TISSUE_UNIFIED,
    pack_groot_unified_frame_lists,
    prepare_groot_row_for_unified,
)
from phi0.schema.unified_action_schema import D_UNIFIED, dim_mask_for_dataset  # noqa: E402

CODE_VERSION = "v2.8"
G1_SONIC_SUPERVISED = dim_mask_for_dataset("g1_sonic")
SRC_VIDEO_KEY = "observation.images.ego_view"
SRC_LEFT_WRIST_KEY = "observation.images.left_wrist"
DST_VIDEO_KEY = SRC_VIDEO_KEY
DST_LEFT_WRIST_KEY = SRC_LEFT_WRIST_KEY
BETAS_DIM = 16
ROOT_DIM = 3

set_verbosity_error()
logging.getLogger("pyarrow").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)


@dataclass
class InfoDict:
    codebase_version: str
    robot_type: str
    total_episodes: int
    total_frames: int
    total_tasks: int
    total_videos: int
    total_chunks: int
    chunks_size: int
    fps: int
    data_path: str
    video_path: str
    features: dict[str, Any]


def append_jsonl_line_atomic(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n"
    fd = os.open(str(path), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        try:
            import fcntl

            fcntl.flock(f, fcntl.LOCK_EX)
        except Exception:
            pass
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class XperienceUnifiedLeRobotConverter:
    """GR00T pick-tissue LeRobot -> unified 512-d LeRobot."""

    def __init__(self, fps: int):
        self.fps = fps
        self.features = Features(
            {
                "unified_action": Sequence(Value("float32")),
                "state_root_trans_world": Sequence(Value("float32")),
                "target_root_trans_world": Sequence(Value("float32")),
                "betas": Sequence(Value("float32")),
                "timestamp": Value("float32"),
                "frame_index": Value("int64"),
                "episode_index": Value("int64"),
                "index": Value("int64"),
                "task_index": Value("int64"),
                "next.done": Value("bool"),
            }
        )
        self.tasks_meta: dict[int, str] = {}
        self.episode_sources: list[tuple[int, Path, Path, Path, int]] = []
        self.lengths_by_episode: dict[int, int] = {}
        self.chunks_size = 1000

    def make_one_episode(
        self,
        task_index: int,
        episode_index: int,
        src_parquet: Path,
        src_video: Path,
        src_left_wrist_video: Path,
        out_base: Path,
        chunks_size: int,
    ) -> tuple[int, int]:
        chunk_path = out_base / f"chunk-{episode_index // chunks_size:03d}"
        chunk_path.mkdir(parents=True, exist_ok=True)
        parquet_path = chunk_path / f"episode_{episode_index:06d}.parquet"

        ego_dir = (
            out_base.parent
            / "videos"
            / f"chunk-{episode_index // chunks_size:03d}"
            / DST_VIDEO_KEY
        )
        ego_dir.mkdir(parents=True, exist_ok=True)
        vid_path = ego_dir / f"episode_{episode_index:06d}.mp4"
        wrist_dir = (
            out_base.parent
            / "videos"
            / f"chunk-{episode_index // chunks_size:03d}"
            / DST_LEFT_WRIST_KEY
        )
        wrist_dir.mkdir(parents=True, exist_ok=True)
        wrist_vid_path = wrist_dir / f"episode_{episode_index:06d}.mp4"

        df = pd.read_parquet(src_parquet)
        n = len(df)
        assert n > 0, f"empty parquet {src_parquet}"

        rows: list[dict[str, Any]] = []
        groot_rows: list[dict[str, Any]] = []
        repaired = 0
        last_valid: dict[str, Any] | None = None
        for i in range(n):
            raw = df.iloc[i].to_dict()
            prepared, last_valid, was_repaired = prepare_groot_row_for_unified(raw, last_valid)
            if was_repaired:
                repaired += 1
            groot_rows.append(prepared)
            packed = pack_groot_unified_frame_lists(prepared)
            rows.append(
                {
                    **packed,
                    "timestamp": i * (1.0 / self.fps),
                    "frame_index": i,
                    "episode_index": episode_index,
                    "index": i,
                    "task_index": task_index,
                    "next.done": (i == n - 1),
                }
            )
        if repaired:
            print(
                f"  episode {episode_index:06d}: forward-filled {repaired}/{n} invalid SMPL ticks",
                flush=True,
            )

        attach_g1_qpos_to_parquet_rows(rows, groot_rows)
        attach_sonic_motion_token_to_parquet_rows(rows, groot_rows)

        tmp_dir = out_base / f"_tmp_ep_{episode_index:06d}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        parquet_tmp = tmp_dir / "episode.parquet"
        Dataset.from_list(rows, features=self.features).to_parquet(str(parquet_tmp))
        os.replace(parquet_tmp, parquet_path)
        shutil.copyfile(src_video, vid_path)
        assert src_left_wrist_video.is_file(), f"missing source left wrist video: {src_left_wrist_video}"
        shutil.copyfile(src_left_wrist_video, wrist_vid_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        acts = np.array([r["unified_action"] for r in rows], dtype=np.float32)
        episode_stats = {
            "episode_index": episode_index,
            "stats": {
                "unified_action": masked_unified_action_stats(
                    acts, supervised_mask=G1_SONIC_SUPERVISED
                ),
                "timestamp": {
                    "min": [0.0],
                    "max": [(n - 1) / self.fps],
                    "mean": [((n - 1) / 2) / self.fps],
                    "std": [n / (2 * self.fps * math.sqrt(3))],
                    "count": [n],
                },
            },
        }
        append_jsonl_line_atomic(out_base.parent / "meta" / "episodes_stats.jsonl", episode_stats)
        return episode_index, n

    def run(self, data_root: Path, work_dir: Path, chunks_size: int, num_workers: int) -> None:
        self.chunks_size = chunks_size
        data_dir = work_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        task_to_idx: dict[str, int] = {}
        for r in read_jsonl(data_root / "meta" / "tasks.jsonl"):
            task_to_idx[r["task"]] = r["task_index"]
            self.tasks_meta[r["task_index"]] = r["task"]
        ep_task: dict[int, str] = {}
        for r in read_jsonl(data_root / "meta" / "episodes.jsonl"):
            tasks = r.get("tasks", [])
            ep_task[r["episode_index"]] = tasks[0] if tasks else ""

        self.episode_sources = []
        out_ep = 0
        for pq in sorted((data_root / "data").rglob("episode_*.parquet")):
            src_ep = int(pq.stem.split("_")[1])
            desc = ep_task.get(src_ep, "")
            task_idx = task_to_idx.get(desc, 0)
            chunk = f"chunk-{src_ep // chunks_size:03d}"
            ep_name = f"episode_{src_ep:06d}.mp4"
            video = data_root / "videos" / chunk / SRC_VIDEO_KEY / ep_name
            assert video.is_file(), f"missing source video: {video}"
            left_wrist = data_root / "videos" / chunk / SRC_LEFT_WRIST_KEY / ep_name
            assert left_wrist.is_file(), f"missing source left wrist video: {left_wrist}"
            self.episode_sources.append((task_idx, pq, video, left_wrist, out_ep))
            out_ep += 1

        print(f"Found {len(self.episode_sources)} episodes, {len(self.tasks_meta)} tasks.")
        if not self.episode_sources:
            print("No episodes found.")
            return

        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [
                ex.submit(
                    self.make_one_episode,
                    task_idx,
                    oi,
                    pq,
                    vid,
                    wrist_vid,
                    data_dir,
                    chunks_size,
                )
                for (task_idx, pq, vid, wrist_vid, oi) in self.episode_sources
            ]
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="Processing episodes", unit="ep"
            ):
                ep_idx, n_frames = fut.result()
                self.lengths_by_episode[ep_idx] = n_frames

        self.num_episodes = len(self.lengths_by_episode)
        self.total_frames = sum(self.lengths_by_episode.values())
        print(f"Now total episodes: {self.num_episodes}, frames: {self.total_frames}")

    def write_meta(self, out_dir: Path) -> None:
        meta_dir = out_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        dataset_cursor = 0
        ep_rows_meta = []
        for (_task_idx, _pq, _vid, _wvid, ep_index) in sorted(self.episode_sources, key=lambda x: x[4]):
            n = self.lengths_by_episode.get(ep_index, 0)
            if n <= 0:
                continue
            task_idx = _task_idx
            ep_rows_meta.append(
                {
                    "episode_index": ep_index,
                    "tasks": [self.tasks_meta.get(task_idx, "")],
                    "length": n,
                    "dataset_from_index": dataset_cursor,
                    "dataset_to_index": dataset_cursor + (n - 1),
                    "robot_type": "g1",
                    "instruction": self.tasks_meta.get(task_idx, ""),
                }
            )
            dataset_cursor += n
        episodes_df = pd.DataFrame(ep_rows_meta).sort_values("episode_index").reset_index(drop=True)

        task_rows = [
            {"task_index": ti, "task": desc, "category": "default", "description": desc}
            for ti, desc in sorted(self.tasks_meta.items())
        ]
        tasks_df = pd.DataFrame(task_rows).sort_values("task_index").reset_index(drop=True)

        video_info = {
            "video.fps": float(self.fps),
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        }
        features_meta = {
            DST_VIDEO_KEY: {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
                "video_info": video_info,
            },
            DST_LEFT_WRIST_KEY: {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
                "video_info": video_info,
            },
            "unified_action": {"dtype": "float32", "shape": [D_UNIFIED]},
            "state_root_trans_world": {"dtype": "float32", "shape": [ROOT_DIM]},
            "target_root_trans_world": {"dtype": "float32", "shape": [ROOT_DIM]},
            "betas": {"dtype": "float32", "shape": [BETAS_DIM]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        }

        info = InfoDict(
            codebase_version=CODE_VERSION,
            robot_type="g1",
            total_episodes=self.num_episodes,
            total_frames=self.total_frames,
            total_tasks=len(self.tasks_meta),
            total_videos=self.num_episodes * 2,
            total_chunks=max(1, math.ceil(self.num_episodes / self.chunks_size)),
            chunks_size=self.chunks_size,
            fps=self.fps,
            data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            video_path="videos/chunk-{episode_chunk:03d}/"
            + DST_VIDEO_KEY
            + "/episode_{episode_index:06d}.mp4",
            features=features_meta,
        )
        with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
            json.dump(asdict(info), f, indent=2)

        episodes_df.to_json(meta_dir / "episodes.jsonl", orient="records", lines=True)
        tasks_df.to_json(meta_dir / "tasks.jsonl", orient="records", lines=True)

        self._write_global_stats(meta_dir)

    def _write_global_stats(self, meta_dir: Path) -> None:
        ep_stats_path = meta_dir / "episodes_stats.jsonl"
        if not ep_stats_path.is_file():
            return
        rows = read_jsonl(ep_stats_path)
        merged = merge_masked_unified_action_stats(
            [row["stats"] for row in rows],
            supervised_mask=G1_SONIC_SUPERVISED,
        )
        if merged is None:
            return
        total = sum(int(row["stats"]["unified_action"]["count"][0]) for row in rows)
        stats_out = {
            "version": 2,
            "action_dim": D_UNIFIED,
            "num_frames": total,
            "robot_action_semantics": STATS_SEMANTICS_PICK_TISSUE_UNIFIED,
            "norm_mode": "z-score",
            "supervised_mask": G1_SONIC_SUPERVISED.tolist(),
            **merged,
        }
        with open(meta_dir / "stats_pick_tissue_unified.json", "w", encoding="utf-8") as f:
            json.dump(stats_out, f, indent=2)
        with open(meta_dir / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats_out, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/data2/wpy/workspace/Isaac-GR00T/data/pick_tissue_valid"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/data2/wpy/workspace/Isaac-GR00T/data/pick_tissue_xperience_unified"),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=0,
        help="Output fps (0 = read from source meta/info.json)",
    )
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = args.fps
    if fps <= 0:
        with open(args.data_root / "meta" / "info.json", encoding="utf-8") as f:
            fps = int(json.load(f).get("fps", 50))
    print(f"Using fps={fps}")

    converter = XperienceUnifiedLeRobotConverter(fps=fps)
    converter.run(args.data_root, out_dir, args.chunks_size, args.num_workers)
    converter.write_meta(out_dir)
    print(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
