#!/usr/bin/env python3
"""Train a small bridge head: Phi_0 output -> CALVIN 7D action."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

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
    p = argparse.ArgumentParser(description="Train CALVIN bridge head for Phi_0")
    p.add_argument("--checkpoint", type=str, required=True, help="Phi_0 checkpoint path")
    p.add_argument("--config-dir", type=str, default=str(ROOT / "configs"))
    p.add_argument("--config-name", type=str, default="train_full")
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
    p.add_argument("--max-episodes", type=int, default=6000)
    p.add_argument(
        "--calvin-split-dir",
        type=str,
        default=str(ROOT / "data/calvin/dataset/task_ABC_D/training"),
        help="CALVIN split dir containing episode_*.npz",
    )
    p.add_argument("--feature-cache", type=str, default=str(ROOT / "data/calvin/bridge_cache.npz"))
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--save-dir", type=str, default=str(ROOT / "experiments/calvin_bridge"))
    p.add_argument("--save-name", type=str, default=None)
    return p.parse_args()


def _episode_id(path: Path) -> int:
    m = re.search(r"episode_(\d+)\.npz$", path.name)
    if m is None:
        raise ValueError(f"Unexpected episode file name: {path.name}")
    return int(m.group(1))


def _load_lang_ranges(split_dir: Path) -> list[tuple[int, int, str]]:
    ann_path = split_dir / "lang_annotations" / "auto_lang_ann.npy"
    if not ann_path.is_file():
        logger.warning("Language annotation file not found: %s", ann_path)
        return []
    payload = np.load(ann_path, allow_pickle=True).item()
    ranges = payload.get("info", {}).get("indx", [])
    anns = payload.get("language", {}).get("ann", [])
    out: list[tuple[int, int, str]] = []
    for rng, text in zip(ranges, anns):
        start, end = int(rng[0]), int(rng[1])
        out.append((start, end, str(text)))
    return out


def _resolve_instruction(frame_id: int, lang_ranges: Iterable[tuple[int, int, str]]) -> str:
    for start, end, text in lang_ranges:
        if start <= frame_id <= end:
            return text
    return "complete the manipulation task"


def _load_obs(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as data:
        return {
            "rgb_obs": {
                "rgb_static": data["rgb_static"].astype(np.uint8),
                "rgb_gripper": data["rgb_gripper"].astype(np.uint8),
            },
            "robot_obs": data["robot_obs"].astype(np.float32),
        }


def _load_gt_actions(npz_paths: list[Path]) -> np.ndarray:
    steps: list[np.ndarray] = []
    for p in npz_paths:
        with np.load(p, allow_pickle=False) as data:
            a = np.asarray(data["rel_actions"], dtype=np.float32).reshape(-1)
            if a.shape[0] < 7:
                raise ValueError(f"rel_actions dims < 7 in {p}")
            step = a[:7].copy()
            step[6] = 1.0 if step[6] > 0 else 0.0
            steps.append(step)
    return np.stack(steps, axis=0)


def build_samples(args: argparse.Namespace, policy: "Phi0VLAPolicy") -> tuple[np.ndarray, np.ndarray]:
    split_dir = Path(args.calvin_split_dir)
    episode_files = sorted(split_dir.glob("episode_*.npz"))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.npz in {split_dir}")
    max_eps = min(len(episode_files), int(args.max_episodes))
    episode_files = episode_files[:max_eps]
    if len(episode_files) <= int(args.num_open_loop_steps):
        raise RuntimeError("Not enough episodes for requested open-loop length")

    lang_ranges = _load_lang_ranges(split_dir)
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    max_samples = int(args.max_samples)
    horizon = int(args.num_open_loop_steps)

    for idx in tqdm(range(0, len(episode_files) - horizon), desc="build-bridge-samples", unit="step"):
        if len(x_list) >= max_samples:
            break
        src_file = episode_files[idx]
        frame_id = _episode_id(src_file)
        instruction = _resolve_instruction(frame_id, lang_ranges)
        obs = _load_obs(src_file)
        gt = _load_gt_actions(episode_files[idx : idx + horizon])

        policy.reset()
        pred_norm = policy.predict_phi0_chunk(obs, instruction, benchmark="calvin")
        feats = policy.build_bridge_features(pred_norm, mode=str(args.bridge_input_mode))
        cur = min(feats.shape[0], gt.shape[0])
        if cur <= 0:
            continue
        x_list.append(feats[:cur])
        y_list.append(gt[:cur])

    if not x_list:
        raise RuntimeError("No bridge samples were built")
    x = np.concatenate(x_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.float32)
    return x, y


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gripper_loss_weight: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_pose = 0.0
    total_grip = 0.0
    total_n = 0
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
def eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    gripper_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_pose = 0.0
    total_grip = 0.0
    total_n = 0
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

    cache_path = Path(args.feature_cache)
    if cache_path.is_file() and not args.rebuild_cache:
        logger.info("Loading cached bridge samples from %s", cache_path)
        cached = np.load(cache_path, allow_pickle=False)
        x = cached["x"].astype(np.float32)
        y = cached["y"].astype(np.float32)
    else:
        logger.info("Building bridge samples from CALVIN trajectories ...")
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
        x, y = build_samples(args, policy)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, x=x, y=y)
        logger.info("Saved bridge cache: %s", cache_path)

    if x.shape[0] < 32:
        raise RuntimeError(f"Too few samples for training: {x.shape[0]}")
    perm = np.random.permutation(x.shape[0])
    x = x[perm]
    y = y[perm]
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.save_name or f"bridge_{int(time.time())}"
    best_path = save_dir / f"{run_name}_best.pt"
    last_path = save_dir / f"{run_name}_last.pt"
    history: list[dict[str, float | int]] = []
    best_val = float("inf")

    for epoch in range(1, int(args.epochs) + 1):
        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            gripper_loss_weight=float(args.gripper_loss_weight),
        )
        val_stats = eval_epoch(
            model,
            val_loader,
            device,
            gripper_loss_weight=float(args.gripper_loss_weight),
        )
        row = {
            "epoch": int(epoch),
            "train_loss_total": float(train_stats["loss_total"]),
            "train_loss_pose": float(train_stats["loss_pose"]),
            "train_loss_gripper": float(train_stats["loss_gripper"]),
            "val_loss_total": float(val_stats["loss_total"]),
            "val_loss_pose": float(val_stats["loss_pose"]),
            "val_loss_gripper": float(val_stats["loss_gripper"]),
        }
        history.append(row)
        logger.info(json.dumps(row, ensure_ascii=False))
        if row["val_loss_total"] < best_val:
            best_val = float(row["val_loss_total"])
            save_bridge_checkpoint(
                best_path,
                model,
                config=cfg,
                input_mode=str(args.bridge_input_mode),
                extra={"history": history, "args": vars(args), "checkpoint": str(args.checkpoint)},
            )

    save_bridge_checkpoint(
        last_path,
        model,
        config=cfg,
        input_mode=str(args.bridge_input_mode),
        extra={"history": history, "args": vars(args), "checkpoint": str(args.checkpoint)},
    )
    summary = {
        "run_name": run_name,
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "feature_dim": int(x.shape[1]),
        "cache_path": str(cache_path),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "bridge_config": asdict(cfg),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
