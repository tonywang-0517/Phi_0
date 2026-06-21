#!/usr/bin/env python3
"""Train Phi_0 -> 7D bridge head from LIBERO or CALVIN RLDS demos."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phi0.benchmark.bridge_head import (  # noqa: E402
    BridgeHeadConfig,
    bridge_loss,
    build_bridge_head,
    save_bridge_checkpoint,
)
from phi0.benchmark.paths import CALVIN_RLDS_ROOT, LIBERO_RLDS_ROOT, libero_rlds_dir
from phi0.benchmark.rlds_adapters import (
    calvin_rlds_action_to_train,
    calvin_rlds_to_env_obs,
    libero_rlds_action_to_train,
    libero_rlds_to_vla_obs,
)
from phi0.benchmark.rlds_io import calvin_shard_glob, iter_rlds_shards, libero_train_shard_glob

if TYPE_CHECKING:
    from phi0.benchmark.policy import Phi0VLAPolicy

logger = logging.getLogger(__name__)


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VLA bridge head from RLDS")
    p.add_argument("--benchmark", choices=["libero", "calvin"], required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config-dir", type=str, default=str(ROOT / "configs"))
    p.add_argument("--config-name", type=str, default="train_act_proprio_800")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--min-free-gb", type=float, default=18.0)
    p.add_argument("--num-open-loop-steps", type=int, default=8)
    p.add_argument("--bridge-input-mode", choices=["keypoints_chunk", "latent_norm"], default="keypoints_chunk")
    p.add_argument("--bridge-head-type", choices=["mlp", "transformer"], default="mlp")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gripper-loss-weight", type=float, default=2.0)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-samples", type=int, default=20000)
    p.add_argument("--max-episodes", type=int, default=800)
    p.add_argument("--max-shards", type=int, default=None, help="Limit RLDS shards (debug)")
    p.add_argument("--libero-suite", type=str, default="libero_spatial")
    p.add_argument("--data-root", type=str, default=None, help="Override RLDS root")
    p.add_argument("--feature-cache", type=str, default=None)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--save-name", type=str, default=None)
    return p.parse_args()


def _default_cache(args: argparse.Namespace) -> Path:
    if args.feature_cache:
        return Path(args.feature_cache)
    tag = args.libero_suite if args.benchmark == "libero" else "calvin_abc"
    return ROOT / "data" / args.benchmark / f"bridge_cache_{tag}.npz"


def _default_save_dir(args: argparse.Namespace) -> Path:
    if args.save_dir:
        return Path(args.save_dir)
    return ROOT / "experiments" / f"{args.benchmark}_bridge"


def build_samples_from_rlds(args: argparse.Namespace, policy: "Phi0VLAPolicy") -> tuple[np.ndarray, np.ndarray]:
    benchmark = str(args.benchmark)
    horizon = int(args.num_open_loop_steps)
    max_samples = int(args.max_samples)
    max_episodes = int(args.max_episodes)

    if benchmark == "libero":
        suite = str(args.libero_suite).replace("_no_noops", "")
        root = Path(args.data_root) if args.data_root else libero_rlds_dir(suite)
        shard_pat = libero_train_shard_glob(suite, root)
        bench_tag = "libero"
    else:
        root = Path(args.data_root) if args.data_root else CALVIN_RLDS_ROOT
        shard_pat = str(root / "calvin_abc-train.tfrecord-*")
        bench_tag = "calvin"

    if not list(Path(shard_pat).parent.glob(Path(shard_pat).name)):
        raise FileNotFoundError(f"No RLDS shards match {shard_pat}")

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    ep_count = 0

    for ep in tqdm(
        iter_rlds_shards(shard_pat, benchmark=bench_tag, max_shards=args.max_shards),
        desc="rlds-episodes",
        unit="ep",
    ):
        if ep_count >= max_episodes or len(x_list) >= max_samples:
            break
        steps = ep.steps
        if len(steps) <= horizon:
            continue
        for start in range(0, len(steps) - horizon):
            if len(x_list) >= max_samples:
                break
            cur = steps[start]
            instruction = cur.language or "complete the manipulation task"
            if benchmark == "libero":
                vla_obs = libero_rlds_to_vla_obs(cur)
                obs = {
                    "agentview_image": cur.rgb_static,
                    "robot0_eye_in_hand_image": cur.rgb_gripper,
                    "robot0_eef_pos": cur.state[:3],
                    "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
                    "robot0_gripper_qpos": cur.state[6:8] if cur.state.shape[0] >= 8 else cur.state[-1:],
                }
                gt = np.stack(
                    [libero_rlds_action_to_train(steps[i].action) for i in range(start, start + horizon)],
                    axis=0,
                )
            else:
                obs = calvin_rlds_to_env_obs(cur)
                gt = np.stack(
                    [calvin_rlds_action_to_train(steps[i].action) for i in range(start, start + horizon)],
                    axis=0,
                )

            policy.reset()
            pred_norm = policy.predict_phi0_chunk(obs, instruction, benchmark=bench_tag)
            feats = policy.build_bridge_features(pred_norm, mode=str(args.bridge_input_mode))
            cur_len = min(feats.shape[0], gt.shape[0])
            if cur_len <= 0:
                continue
            x_list.append(feats[:cur_len])
            y_list.append(gt[:cur_len])
        ep_count += 1

    if not x_list:
        raise RuntimeError("No bridge samples built from RLDS")
    return np.concatenate(x_list, axis=0).astype(np.float32), np.concatenate(y_list, axis=0).astype(np.float32)


def train_epoch(model, loader, optimizer, device, gripper_loss_weight):
    model.train()
    total_loss = total_pose = total_grip = total_n = 0.0
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss, parts = bridge_loss(logits, y, gripper_loss_weight=gripper_loss_weight)
        loss.backward()
        optimizer.step()
        bs = int(x.shape[0])
        total_n += bs
        total_loss += parts["loss_total"] * bs
        total_pose += parts["loss_pose"] * bs
        total_grip += parts["loss_gripper"] * bs
    denom = max(1, total_n)
    return {
        "loss_total": total_loss / denom,
        "loss_pose": total_pose / denom,
        "loss_gripper": total_grip / denom,
    }


@torch.no_grad()
def eval_epoch(model, loader, device, gripper_loss_weight):
    model.eval()
    total_loss = total_pose = total_grip = total_n = 0.0
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        logits = model(x)
        _, parts = bridge_loss(logits, y, gripper_loss_weight=gripper_loss_weight)
        bs = int(x.shape[0])
        total_n += bs
        total_loss += parts["loss_total"] * bs
        total_pose += parts["loss_pose"] * bs
        total_grip += parts["loss_gripper"] * bs
    denom = max(1, total_n)
    return {
        "loss_total": total_loss / denom,
        "loss_pose": total_pose / denom,
        "loss_gripper": total_grip / denom,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    cache_path = _default_cache(args)
    if cache_path.is_file() and not args.rebuild_cache:
        logger.info("Loading cached features from %s", cache_path)
        cached = np.load(cache_path, allow_pickle=False)
        x, y = cached["x"].astype(np.float32), cached["y"].astype(np.float32)
    else:
        from phi0.benchmark.policy import Phi0VLAPolicy

        policy = Phi0VLAPolicy.from_paths(
            checkpoint=args.checkpoint,
            config_dir=args.config_dir,
            config_name=args.config_name,
            device=args.device,
            min_free_gb=float(args.min_free_gb),
            num_open_loop_steps=int(args.num_open_loop_steps),
            action_mode="heuristic",
            bridge_input_mode=str(args.bridge_input_mode),
        )
        x, y = build_samples_from_rlds(args, policy)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, x=x, y=y)
        logger.info("Saved cache %s (%d samples)", cache_path, x.shape[0])

    perm = np.random.permutation(x.shape[0])
    x, y = x[perm], y[perm]
    n_val = max(1, int(round(float(args.val_ratio) * x.shape[0])))
    x_val, y_val = x[:n_val], y[:n_val]
    x_train, y_train = x[n_val:], y[n_val:]

    train_loader = DataLoader(ArrayDataset(x_train, y_train), batch_size=int(args.batch_size), shuffle=True)
    val_loader = DataLoader(ArrayDataset(x_val, y_val), batch_size=int(args.batch_size), shuffle=False)

    cfg = BridgeHeadConfig(
        input_dim=int(x.shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        head_type=str(args.bridge_head_type),
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_bridge_head(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    save_dir = _default_save_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.save_name or f"bridge_{args.benchmark}_{int(time.time())}"
    best_path = save_dir / f"{run_name}_best.pt"
    last_path = save_dir / f"{run_name}_last.pt"
    history: list[dict] = []
    best_val = float("inf")

    for epoch in range(1, int(args.epochs) + 1):
        train_stats = train_epoch(model, train_loader, optimizer, device, float(args.gripper_loss_weight))
        val_stats = eval_epoch(model, val_loader, device, float(args.gripper_loss_weight))
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in val_stats.items()}}
        history.append(row)
        logger.info(json.dumps(row, ensure_ascii=False))
        if row["val_loss_total"] < best_val:
            best_val = float(row["val_loss_total"])
            save_bridge_checkpoint(
                best_path, model, config=cfg, input_mode=str(args.bridge_input_mode),
                extra={"history": history, "args": vars(args), "checkpoint": str(args.checkpoint)},
            )

    save_bridge_checkpoint(
        last_path, model, config=cfg, input_mode=str(args.bridge_input_mode),
        extra={"history": history, "args": vars(args), "checkpoint": str(args.checkpoint)},
    )
    print(json.dumps({
        "benchmark": args.benchmark,
        "run_name": run_name,
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "feature_dim": int(x.shape[1]),
        "cache_path": str(cache_path),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "bridge_config": asdict(cfg),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
