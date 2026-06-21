#!/usr/bin/env python3
"""Replay OpenVLA LIBERO RLDS expert actions in sim (GT sanity check before training).

Matches each RLDS episode to a LIBERO benchmark task by language, searches for an
init_state whose proprio aligns with the episode's first observation, then replays
RLDS actions through the same delta-OSC + gripper pipeline used at deploy time.

Outputs per-episode success, proprio tracking error vs RLDS states, and optional mp4.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LIBERO_ROOT = ROOT / "third_party" / "LIBERO"
if str(LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_ROOT))

# LIBERO init_states pickles need weights_only=False on PyTorch 2.6+.
_torch_load = __import__("torch").load


def _torch_load_libero(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


import torch  # noqa: E402

torch.load = _torch_load_libero  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

LIBERO_TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _norm_task_key(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _save_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def _libero_frame(obs: dict[str, Any]) -> np.ndarray:
    return np.asarray(obs["agentview_image"])[::-1, ::-1]


@dataclass
class ReplayResult:
    episode_id: str
    language: str
    task_id: int
    init_idx: int
    steps: int
    success: bool
    init_match_err: float
    mean_proprio_err: float
    max_proprio_err: float
    video_path: str | None = None


def _proprio_err(sim_7d: np.ndarray, rlds_state: np.ndarray) -> float:
    from phi0.benchmark.adapters import libero_obs_to_eef_7d
    from phi0.benchmark.rlds_adapters import libero_rlds_state_to_eef_7d

    if sim_7d.ndim == 1:
        tgt = libero_rlds_state_to_eef_7d(rlds_state)
        return float(np.linalg.norm(sim_7d - tgt))
    raise ValueError("expected 1D sim proprio")


def _find_matching_init(
    env,
    init_states: np.ndarray,
    rlds_state0: np.ndarray,
    *,
    steps_wait: int,
    max_inits: int | None,
) -> tuple[int, float, dict[str, Any]]:
    from phi0.benchmark.adapters import libero_obs_to_eef_7d
    from phi0.benchmark.rlds_adapters import libero_rlds_state_to_eef_7d

    target = libero_rlds_state_to_eef_7d(rlds_state0)
    n = int(init_states.shape[0])
    if max_inits is not None:
        n = min(n, int(max_inits))
    best_idx, best_err, best_obs = 0, float("inf"), None
    for idx in range(n):
        env.reset()
        obs = env.set_init_state(init_states[idx])
        for _ in range(int(steps_wait)):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        sim7 = libero_obs_to_eef_7d(obs)
        err = float(np.linalg.norm(sim7 - target))
        if err < best_err:
            best_err, best_idx, best_obs = err, idx, obs
        if err < 0.02:
            break
    if best_obs is None:
        raise RuntimeError("failed to evaluate init states")
    return best_idx, best_err, best_obs


def replay_episode(
    episode,
    *,
    task_suite,
    suite_name: str,
    steps_wait: int = 10,
    max_steps: int | None = None,
    init_match_max: int | None = 50,
    init_match_threshold: float = 0.08,
    save_video: bool = False,
    video_dir: Path | None = None,
    video_fps: int = 20,
    seed: int = 7,
) -> ReplayResult:
    from phi0.benchmark.adapters import libero_obs_to_eef_7d, process_vla_action
    from phi0.benchmark.libero_env import make_libero_offscreen_env
    from phi0.benchmark.rlds_adapters import libero_rlds_action_to_train, libero_rlds_state_to_eef_7d
    from libero.libero import get_libero_path

    lang = episode.steps[0].language if episode.steps else ""
    key = _norm_task_key(lang)
    task_id = None
    for i in range(task_suite.n_tasks):
        if _norm_task_key(task_suite.get_task(i).language) == key:
            task_id = i
            break
    if task_id is None:
        raise ValueError(f"No LIBERO task match for language: {lang!r}")

    task = task_suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init_states = task_suite.get_task_init_states(task_id)
    env = make_libero_offscreen_env(
        bddl_file_name=bddl,
        camera_heights=256,
        camera_widths=256,
        osc_absolute=False,
    )
    env.seed(int(seed))

    init_idx, init_err, obs = _find_matching_init(
        env,
        init_states,
        episode.steps[0].state,
        steps_wait=steps_wait,
        max_inits=init_match_max,
    )
    if init_err > float(init_match_threshold):
        logger.warning(
            "Weak init match for %r: best err=%.4f (threshold %.4f, init_idx=%d)",
            lang,
            init_err,
            init_match_threshold,
            init_idx,
        )

    cap = max_steps if max_steps is not None else int(LIBERO_TASK_MAX_STEPS.get(suite_name, 520))
    cap = min(cap, len(episode.steps))
    frames: list[np.ndarray] = []
    if save_video:
        frames.append(_libero_frame(obs))

    proprio_errs: list[float] = []
    done = False
    for t in range(cap):
        rlds_step = episode.steps[t]
        sim7 = libero_obs_to_eef_7d(obs)
        tgt7 = libero_rlds_state_to_eef_7d(rlds_step.state)
        proprio_errs.append(float(np.linalg.norm(sim7 - tgt7)))

        train7 = libero_rlds_action_to_train(rlds_step.action)
        sim_action = process_vla_action(train7, invert_openvla_gripper=True)
        obs, _, done, _ = env.step(sim_action.tolist())
        if save_video:
            frames.append(_libero_frame(obs))
        if done:
            break

    env.close()

    video_path = None
    if save_video and video_dir is not None and frames:
        tag = "success" if done else "fail"
        safe = key.replace(" ", "_")
        ep_tag = str(episode.episode_id or "unknown").replace("/", "_").replace(" ", "_")
        video_path = str((video_dir / safe / f"ep_{ep_tag}_{tag}.mp4").resolve())
        _save_video(frames, Path(video_path), fps=int(video_fps))

    return ReplayResult(
        episode_id=str(episode.episode_id or ""),
        language=lang,
        task_id=int(task_id),
        init_idx=int(init_idx),
        steps=int(t + 1 if episode.steps else 0),
        success=bool(done),
        init_match_err=float(init_err),
        mean_proprio_err=float(np.mean(proprio_errs)) if proprio_errs else float("nan"),
        max_proprio_err=float(np.max(proprio_errs)) if proprio_errs else float("nan"),
        video_path=video_path,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay LIBERO RLDS GT actions in sim")
    p.add_argument("--suite", type=str, default="libero_spatial")
    p.add_argument("--max-episodes", type=int, default=10)
    p.add_argument("--max-shards", type=int, default=1)
    p.add_argument("--steps-wait", type=int, default=10)
    p.add_argument("--init-match-max", type=int, default=50, help="Search at most N init states")
    p.add_argument("--init-match-threshold", type=float, default=0.08)
    p.add_argument("--save-videos", action="store_true")
    p.add_argument("--video-dir", type=str, default=str(ROOT / "experiments" / "gt_replay_videos"))
    p.add_argument("--video-fps", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=str, default=str(ROOT / "experiments" / "gt_replay_report.json"))
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    from libero.libero import benchmark
    from phi0.benchmark.rlds_io import iter_rlds_shards, libero_train_shard_glob

    suite_name = str(args.suite)
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise ValueError(f"Unknown suite: {suite_name}")
    task_suite = benchmark_dict[suite_name]()

    shard_pat = libero_train_shard_glob(suite_name)
    video_dir = Path(args.video_dir) if args.save_videos else None
    results: list[ReplayResult] = []

    for ep_i, episode in enumerate(
        iter_rlds_shards(shard_pat, benchmark="libero", max_shards=int(args.max_shards))
    ):
        if ep_i >= int(args.max_episodes):
            break
        logger.info("Replaying episode %d (%d steps): %s", ep_i, len(episode.steps), episode.steps[0].language)
        try:
            res = replay_episode(
                episode,
                task_suite=task_suite,
                suite_name=suite_name,
                steps_wait=int(args.steps_wait),
                init_match_max=int(args.init_match_max),
                init_match_threshold=float(args.init_match_threshold),
                save_video=bool(args.save_videos),
                video_dir=video_dir,
                video_fps=int(args.video_fps),
                seed=int(args.seed),
            )
            results.append(res)
            logger.info(
                "  init_err=%.4f init_idx=%d success=%s mean_proprio_err=%.4f max=%.4f steps=%d",
                res.init_match_err,
                res.init_idx,
                res.success,
                res.mean_proprio_err,
                res.max_proprio_err,
                res.steps,
            )
            if res.video_path:
                logger.info("  video: %s", res.video_path)
        except Exception as exc:
            logger.exception("Episode %d failed: %s", ep_i, exc)

    n = len(results)
    succ = sum(int(r.success) for r in results)
    report = {
        "suite": suite_name,
        "episodes": n,
        "success_rate": float(succ / max(1, n)),
        "mean_init_match_err": float(np.mean([r.init_match_err for r in results])) if results else None,
        "mean_proprio_err": float(np.mean([r.mean_proprio_err for r in results])) if results else None,
        "results": [r.__dict__ for r in results],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"written": str(out.resolve()), **report}, indent=2))


if __name__ == "__main__":
    main()
