"""Closed-loop inference scheduling: prefetch trigger + step-aligned chunk index."""

from __future__ import annotations

import math


def resolve_min_inference_interval(inference_rate_hz: float) -> float | None:
    """Return minimum seconds between triggers, or None for ASAP (no rate cap)."""
    if inference_rate_hz <= 0:
        return None
    return 1.0 / float(inference_rate_hz)


def prefetch_trigger_steps(
    latency_budget_s: float,
    control_fps: float,
    *,
    safety_steps: int = 2,
) -> int:
    """Start the next infer when this many chunk indices remain (inclusive)."""
    budget = max(float(latency_budget_s), 0.0)
    fps = max(float(control_fps), 1.0)
    return int(math.ceil(budget * fps)) + int(safety_steps)


def chunk_index_when_result_arrives(
    steps_since_chunk_consumed: int,
    infer_trigger_steps: int,
    action_horizon: int,
) -> int:
    """Map control steps elapsed during infer -> starting index in the new chunk."""
    delta = max(0, int(steps_since_chunk_consumed) - int(infer_trigger_steps))
    return min(delta, int(action_horizon) - 1)


def inference_pipeline_idle(
    *,
    worker_busy: bool,
    inference_queue_pending: bool,
    result_queue_pending: bool,
) -> bool:
    return not worker_busy and not inference_queue_pending and not result_queue_pending


def should_trigger_chunk_prefetch(
    *,
    cached_chunk_exists: bool,
    pipeline_idle: bool,
    action_chunk_index: int,
    action_horizon: int,
    prefetch_steps: int,
    min_interval_s: float | None,
    time_since_last_trigger_s: float,
    steps_since_chunk_consumed: int,
) -> bool:
    """Trigger when the cached chunk is running out or bootstrap is needed."""
    if not pipeline_idle:
        return False
    if not cached_chunk_exists:
        return True
    last_index = int(action_horizon) - 1
    steps_remaining = last_index - int(action_chunk_index)
    steps_elapsed = int(steps_since_chunk_consumed)
    # Just swapped in a new chunk with headroom — play at least one frame first.
    if steps_elapsed == 0 and steps_remaining > 0:
        return False
    if steps_remaining <= int(prefetch_steps):
        return True
    # Overlap: while infer is still faster than chunk length, start next pass early.
    if steps_elapsed == 1 and int(action_chunk_index) <= int(prefetch_steps):
        return True
    if min_interval_s is not None and time_since_last_trigger_s >= min_interval_s:
        # ponytail: legacy fallback if index gate missed; prefer prefetch_steps in normal use
        return steps_remaining <= last_index
    return False
