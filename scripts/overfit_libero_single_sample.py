#!/usr/bin/env python3
"""Single-batch overfit — last-100-step mean norm_l1 must stay below 0.01."""

from __future__ import annotations

import copy
import logging
import sys
from collections import deque
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from phi0.data.sequence import SequenceDataset, sequence_dataset_from_cfg  # noqa: E402
from phi0.runtime import (  # noqa: E402
    build_base_dataset,
    build_optimizer,
    build_processor,
    create_phi0,
    prepare_model_batch_cpu,
    prepare_model_batch_gpu,
    sync_model_action_norm,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("overfit_libero")

MAX_STEPS = 1000
TARGET_NORM_L1 = 0.01
AVG_WINDOW = 100
MIN_STEPS_BEFORE_PASS = 200
LOG_EVERY = 50
# 408M DiT + Adam: lr=2e-4 oscillates; step decay @500 explodes momentum state.
LEARNING_RATE = 3e-5


def _pick_typical_batch(seq: SequenceDataset, *, scan: int = 32) -> dict:
    """Prefer a clip with small |xyz| delta (skip episode-start outliers)."""
    best_idx, best_score = 0, float("inf")
    for i in range(min(scan, len(seq))):
        item = seq[i]
        if "robot_future_delta_7d" not in item:
            continue
        d = item["robot_future_delta_7d"][..., :3].abs().mean().item()
        if d < best_score:
            best_score, best_idx = d, i
    start = best_idx
    return SequenceDataset.collate_fn([seq[start]])


def main() -> None:
    out_dir = ROOT / "experiments/libero_delta_overfit_1sample"
    out_dir.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(
            config_name="train_libero_spatial_act_delta_15k_single",
            overrides=[
                f"output_dir={out_dir}",
                "batch_size=1",
                "num_workers=0",
                "compile_action_expert=false",
                "auto_resume=false",
                "resume_ckpt=null",
                f"learning_rate_action={LEARNING_RATE}",
                "gradient_clipping=1.0",
                "mixed_precision=bf16",
                "data.libero_max_episodes=1",
                "device=cuda:0",
                "model.action_dit_config.hidden_dim=2048",
                "model.action_dit_config.ffn_dim=8192",
                "model.action_dit_config.num_layers=6",
                "model.action_dit_config.future_placeholder_noise_std=0.02",
            ],
        )

    processor = build_processor(cfg)
    torch.manual_seed(0)
    model = create_phi0(cfg)
    sync_model_action_norm(model, processor)
    assert model.uses_robot7d_action()
    logger.info(
        "arch hidden=%s layers=%s perturbation=%s text_embed=%s vggt_embed=%s",
        getattr(model.action_expert, "hidden_dim", "?"),
        len(getattr(model.action_expert, "blocks", [])),
        getattr(model.action_expert, "future_placeholder_noise_std", 0.0),
        model.action_expert.text_embedding is not None,
        getattr(model.action_expert, "vggt_embedding", None) is not None,
    )

    base = build_base_dataset(cfg)
    seq = sequence_dataset_from_cfg(base, cfg.data)
    raw_batch = _pick_typical_batch(seq)
    cpu_payload = prepare_model_batch_cpu(model, processor, raw_batch)
    gpu_batch = prepare_model_batch_gpu(model, processor, cpu_payload)

    w = int(model.past_action_window_size)
    tgt_norm = gpu_batch["action"][:, w:]
    logger.info(
        "batch action=%s future_norm=%s |xyz|=%.4f",
        tuple(gpu_batch["action"].shape),
        tuple(tgt_norm.shape),
        float(raw_batch["robot_future_delta_7d"][..., :3].abs().mean()),
    )

    optim = build_optimizer(model, cfg)
    device = model.device
    model.train()
    model.set_frozen_towers_eval()

    best = float("inf")
    best_step = -1
    best_state: dict[str, torch.Tensor] | None = None
    recent: deque[float] = deque(maxlen=AVG_WINDOW)
    pass_step = -1
    for step in range(MAX_STEPS):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            loss, _ = model.training_loss(gpu_batch)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        lv = float(loss.item())
        recent.append(lv)
        avg_recent = sum(recent) / len(recent)
        if lv < best:
            best, best_step = lv, step
            best_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items() if k.startswith("action_expert.")}
            )
        if step == 0 or step % LOG_EVERY == 0 or step == MAX_STEPS - 1:
            logger.info(
                "step=%d instant=%.6f avg%d=%.6f best=%.6f@%d",
                step,
                lv,
                len(recent),
                avg_recent,
                best,
                best_step,
            )
        if (
            pass_step < 0
            and step + 1 >= MIN_STEPS_BEFORE_PASS
            and len(recent) == AVG_WINDOW
            and avg_recent <= TARGET_NORM_L1
        ):
            pass_step = step
            logger.info(
                "avg%d=%.6f <= %.2e at step %d — stable overfit reached",
                AVG_WINDOW,
                avg_recent,
                TARGET_NORM_L1,
                step,
            )
            break

    if best_state is not None:
        model.load_state_dict({**model.state_dict(), **best_state}, strict=False)
        torch.save(
            {
                "step": best_step,
                "norm_l1": best,
                "avg_window": AVG_WINDOW,
                "pass_step": pass_step,
                "action_expert": best_state,
            },
            out_dir / "overfit_best.pt",
        )

    if pass_step >= 0:
        logger.info(
            "PASS avg%d norm_l1=%.2e <= %.2e (best=%.2e @ step %d)",
            AVG_WINDOW,
            avg_recent,
            TARGET_NORM_L1,
            best,
            best_step,
        )
        return

    logger.error(
        "FAIL avg%d=%.6f best=%.6f (need avg <= %.2e after step %d)",
        len(recent),
        avg_recent,
        best,
        TARGET_NORM_L1,
        MIN_STEPS_BEFORE_PASS,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
