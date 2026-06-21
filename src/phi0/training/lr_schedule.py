"""Learning-rate helpers for action-head training."""

from __future__ import annotations

import math
from typing import Literal, Sequence

LrScaleMode = Literal["none", "sqrt", "linear"]
LrSchedulerType = Literal["none", "constant", "cosine", "cosine_with_min_lr"]


def scaled_action_learning_rate(
    *,
    per_device_batch: int,
    reference_batch: int = 16,
    reference_lr: float = 1.5e-4,
    scale: LrScaleMode = "sqrt",
    explicit_lr: float | None = None,
) -> float:
    """Scale action LR from a reference batch (default: LIBERO single-GPU batch=16)."""
    if explicit_lr is not None and scale == "none":
        return float(explicit_lr)
    ref_bs = max(1, int(reference_batch))
    bs = max(1, int(per_device_batch))
    ref_lr = float(reference_lr)
    if scale == "linear":
        return ref_lr * (bs / ref_bs)
    if scale == "sqrt":
        return ref_lr * math.sqrt(bs / ref_bs)
    if explicit_lr is not None:
        return float(explicit_lr)
    return ref_lr


def apply_warmup_lr(
    optimizer,
    step: int,
    warmup_steps: int,
    base_lrs: Sequence[float],
) -> None:
    """Linear LR warmup: step 0 uses ``base/warmup``; step warmup-1 reaches ``base``."""
    if warmup_steps <= 0:
        return
    scale = min(1.0, float(step + 1) / float(warmup_steps))
    for group, base in zip(optimizer.param_groups, base_lrs):
        group["lr"] = float(base) * scale


def build_lr_scheduler(
    optimizer,
    *,
    scheduler_type: str,
    max_steps: int,
    warmup_steps: int,
    min_lr: float = 0.0,
):
    """Build HF-style LR scheduler (Psi0 ``get_scheduler`` path)."""
    name = str(scheduler_type).strip().lower()
    if name in {"", "none", "constant"}:
        return None
    from transformers.optimization import get_scheduler

    kwargs = {}
    if name == "cosine_with_min_lr":
        kwargs["min_lr"] = float(min_lr)
    return get_scheduler(
        name,
        optimizer=optimizer,
        num_warmup_steps=int(warmup_steps),
        num_training_steps=int(max_steps),
        scheduler_specific_kwargs=kwargs or None,
    )
