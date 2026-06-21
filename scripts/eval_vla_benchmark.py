#!/usr/bin/env python3
"""Evaluate Phi_0 on LIBERO/CALVIN with VLA-Adapter I/O contract."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

# LIBERO init_states use numpy pickles; PyTorch 2.6+ defaults weights_only=True.
_torch_load = torch.load


def _torch_load_libero(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_libero  # type: ignore[misc, assignment]

ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = ROOT / "third_party" / "LIBERO"
if str(LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_ROOT))

logger = logging.getLogger(__name__)

sys.path.insert(0, str(ROOT / "src"))

LIBERO_TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phi_0 VLA-format benchmark eval")
    p.add_argument("--benchmark", type=str, choices=["libero", "calvin", "cavin"], required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config-dir", type=str, default=str(ROOT / "configs"))
    p.add_argument("--config-name", type=str, default="train_libero_spatial_act_300")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--min-free-gb", type=float, default=18.0)
    p.add_argument("--num-open-loop-steps", type=int, default=8, help="VLA-Adapter NUM_ACTIONS_CHUNK for LIBERO")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--action-mode", type=str, choices=["heuristic", "bridge", "robot7d"], default="robot7d")
    p.add_argument("--bridge-checkpoint", type=str, default=None)
    p.add_argument(
        "--bridge-input-mode",
        type=str,
        choices=["keypoints_chunk", "latent_norm"],
        default="keypoints_chunk",
    )

    # LIBERO options
    p.add_argument("--libero-suite", type=str, default="libero_spatial")
    p.add_argument("--libero-trials-per-task", type=int, default=1)
    p.add_argument(
        "--libero-max-tasks",
        type=int,
        default=None,
        help="Evaluate only the first N tasks (default: all tasks in suite)",
    )
    p.add_argument("--libero-env-img-res", type=int, default=256)
    p.add_argument("--libero-steps-wait", type=int, default=10)
    p.add_argument("--libero-seed", type=int, default=7)
    p.add_argument(
        "--libero-osc-absolute",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="LIBERO OSC_POSE with control_delta=false (absolute EEF targets)",
    )
    p.add_argument("--save-videos", action="store_true", help="Save LIBERO rollout mp4 per trial")
    p.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory for rollout videos (default: <output_parent>/videos)",
    )
    p.add_argument("--video-fps", type=int, default=20)

    # CALVIN options
    p.add_argument("--calvin-root", type=str, default=None)
    p.add_argument("--calvin-num-sequences", type=int, default=100)
    p.add_argument("--calvin-episode-len", type=int, default=360)
    p.add_argument("--calvin-debug", action="store_true")
    return p.parse_args()


def _make_policy(args: argparse.Namespace):
    from phi0.benchmark.policy import Phi0VLAPolicy

    return Phi0VLAPolicy.from_paths(
        checkpoint=args.checkpoint,
        config_dir=args.config_dir,
        config_name=args.config_name,
        device=args.device,
        min_free_gb=float(args.min_free_gb),
        num_open_loop_steps=int(args.num_open_loop_steps),
        action_mode=str(args.action_mode),
        bridge_checkpoint=args.bridge_checkpoint,
        bridge_input_mode=str(args.bridge_input_mode),
    )


def _libero_frame(obs: dict[str, Any]) -> np.ndarray:
    """Agentview RGB for rollout video (VLA-Adapter eval orientation)."""
    return np.asarray(obs["agentview_image"])[::-1, ::-1]


def _save_episode_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def eval_libero(args: argparse.Namespace, policy) -> dict[str, Any]:
    try:
        from libero.libero import benchmark
        from libero.libero import get_libero_path
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "LIBERO 未安装或不可导入。请先安装 libero 并配置好仿真环境。"
        ) from exc

    from phi0.benchmark.libero_env import make_libero_offscreen_env

    np.random.seed(int(args.libero_seed))
    benchmark_dict = benchmark.get_benchmark_dict()
    if args.libero_suite not in benchmark_dict:
        raise ValueError(f"Unknown LIBERO suite: {args.libero_suite}")
    task_suite = benchmark_dict[args.libero_suite]()
    max_steps = int(LIBERO_TASK_MAX_STEPS.get(args.libero_suite, 520))
    total_episodes, total_success = 0, 0
    task_stats: dict[str, dict[str, int]] = {}
    video_dir: Path | None = None
    if args.save_videos:
        if args.video_dir:
            video_dir = Path(args.video_dir)
        elif args.output:
            video_dir = Path(args.output).resolve().parent / "videos"
        else:
            video_dir = Path("experiments") / "eval_videos"
        video_dir.mkdir(parents=True, exist_ok=True)

    n_tasks = int(task_suite.n_tasks)
    if args.libero_max_tasks is not None:
        n_tasks = min(n_tasks, int(args.libero_max_tasks))

    for task_id in tqdm(range(n_tasks), desc="libero-tasks", unit="task"):
        task = task_suite.get_task(task_id)
        task_desc = task.language
        task_key = task_desc.lower().replace(" ", "_")
        init_states = task_suite.get_task_init_states(task_id)
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        env = make_libero_offscreen_env(
            bddl_file_name=task_bddl_file,
            camera_heights=int(args.libero_env_img_res),
            camera_widths=int(args.libero_env_img_res),
            osc_absolute=bool(getattr(args, "libero_osc_absolute", True)),
        )
        env.seed(int(args.libero_seed))
        succ = 0
        ep = 0
        for ep_idx in range(int(args.libero_trials_per_task)):
            policy.reset()
            env.reset()
            obs = env.set_init_state(init_states[ep_idx])
            policy.observe(obs, benchmark="libero", step=0)
            action_queue: deque[np.ndarray] = deque(maxlen=int(args.num_open_loop_steps))
            done = False
            steps = 0
            frames: list[np.ndarray] = []
            if video_dir is not None:
                frames.append(_libero_frame(obs))
            while steps < (max_steps + int(args.libero_steps_wait)):
                if steps < int(args.libero_steps_wait):
                    obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                    steps += 1
                    policy.observe(obs, benchmark="libero", step=steps)
                    if video_dir is not None:
                        frames.append(_libero_frame(obs))
                    continue
                if len(action_queue) == 0:
                    actions = policy.step(obs, task_desc, steps, benchmark="libero")
                    action_queue.extend(actions)
                action = action_queue.popleft()
                obs, _, done, _ = env.step(action.tolist())
                steps += 1
                policy.observe(obs, benchmark="libero", step=steps)
                if video_dir is not None:
                    frames.append(_libero_frame(obs))
                if done:
                    succ += 1
                    break
            if video_dir is not None and frames:
                tag = "success" if done else "fail"
                out_path = video_dir / task_key / f"trial{ep_idx}_{tag}.mp4"
                _save_episode_video(frames, out_path, fps=int(args.video_fps))
                logger.info("Saved rollout video: %s", out_path)
            ep += 1
            total_episodes += 1
            total_success += int(done)
        env.close()
        task_stats[task_key] = {"success": int(succ), "total": int(ep)}

    per_task_sr = {
        task: float(stats["success"] / max(1, stats["total"])) for task, stats in task_stats.items()
    }
    result = {
        "benchmark": "libero",
        "suite": args.libero_suite,
        "total_episodes": int(total_episodes),
        "total_success": int(total_success),
        "success_rate": float(total_success / max(1, total_episodes)),
        "task_stats": task_stats,
        "per_task_sr": per_task_sr,
    }
    if video_dir is not None:
        result["video_dir"] = str(video_dir.resolve())
    return result


def eval_calvin(args: argparse.Namespace, policy) -> dict[str, Any]:
    from phi0.benchmark.calvin_env_wrapper import CalvinEnvWrapperRaw
    from phi0.benchmark.paths import calvin_eval_root

    if args.calvin_root:
        calvin_root = Path(args.calvin_root)
    else:
        calvin_root = calvin_eval_root()

    try:
        import hydra
        from calvin_agent.evaluation.multistep_sequences import get_sequences
        from calvin_agent.evaluation.utils import (
            count_success,
            get_env_state_for_initial_condition,
        )
        from omegaconf import OmegaConf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "CALVIN 依赖未就绪。请运行: bash scripts/install_calvin_deps.sh && bash scripts/setup_benchmark_data.sh"
        ) from exc

    os.environ["CALVIN_ROOT"] = str(calvin_root)
    conf_dir = Path(f"{calvin_root}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    observation_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": ["depth_static", "depth_gripper"],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }
    val_folder = calvin_root / "dataset/task_ABC_D/validation"
    if not val_folder.is_dir():
        raise FileNotFoundError(
            f"CALVIN validation data not found: {val_folder}\n"
            "Run: bash scripts/setup_benchmark_data.sh"
        )
    env = CalvinEnvWrapperRaw(val_folder, observation_space, "cuda")
    eval_sequences = get_sequences(int(args.calvin_num_sequences))
    results: list[int] = []
    task_success = Counter()
    task_total = Counter()

    for seq_idx, (initial_state, sequence) in enumerate(
        tqdm(eval_sequences, desc="calvin-seq", unit="seq")
    ):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        succ_len = 0
        for subtask in sequence:
            obs = env.get_obs()
            policy.reset()
            lang = val_annotations[subtask][0]
            action_queue: deque[np.ndarray] = deque(maxlen=int(args.num_open_loop_steps))
            start_info = env.get_info()
            done_subtask = False
            for step in range(int(args.calvin_episode_len)):
                if len(action_queue) == 0:
                    actions = policy.step(obs, lang, step, benchmark="calvin")
                    action_queue.extend(actions)
                action = action_queue.popleft()
                obs, _, _, current_info = env.step(action.tolist())
                current_task_info = task_oracle.get_task_info_for_set(
                    start_info, current_info, {subtask}
                )
                if len(current_task_info) > 0:
                    done_subtask = True
                    break
            task_total[subtask] += 1
            if done_subtask:
                task_success[subtask] += 1
                succ_len += 1
            else:
                break
        results.append(succ_len)
        if args.calvin_debug:
            logger.info("sequence %d success_len=%d", seq_idx, succ_len)

    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
    task_stats = {
        k: {"success": int(task_success[k]), "total": int(task_total[k])}
        for k in sorted(task_total.keys())
    }
    per_task_sr = {
        k: float(task_success[k] / max(1, task_total[k])) for k in sorted(task_total.keys())
    }
    return {
        "benchmark": "calvin",
        "num_sequences": int(len(results)),
        "avg_seq_len": float(np.mean(results) if results else 0.0),
        "chain_sr": chain_sr,
        "task_stats": task_stats,
        "per_task_sr": per_task_sr,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    policy = _make_policy(args)
    bench = "calvin" if args.benchmark == "cavin" else args.benchmark
    if bench == "libero":
        report = eval_libero(args, policy)
    else:
        report = eval_calvin(args, policy)
    report["elapsed_sec"] = float(time.time() - t0)
    report["checkpoint"] = str(Path(args.checkpoint).resolve())
    report["config_name"] = args.config_name
    report["num_open_loop_steps"] = int(args.num_open_loop_steps)
    report["action_mode"] = str(args.action_mode)
    if args.bridge_checkpoint:
        report["bridge_checkpoint"] = str(Path(args.bridge_checkpoint).resolve())
    report["bridge_input_mode"] = str(args.bridge_input_mode)

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote report to {out}")


if __name__ == "__main__":
    main()

