"""Tests for LR scaling and training batch pipeline."""

from __future__ import annotations

import pytest

from phi0.data.train_prefetch import AsyncGpuPrepPipeline, GpuBatchPipeline, iter_training_batches
from phi0.training.lr_schedule import apply_warmup_lr, scaled_action_learning_rate


def test_scaled_action_learning_rate_sqrt():
    lr = scaled_action_learning_rate(
        per_device_batch=256,
        reference_batch=16,
        reference_lr=1.5e-4,
        scale="sqrt",
    )
    assert lr == pytest.approx(6.0e-4)


def test_scaled_action_learning_rate_linear():
    lr = scaled_action_learning_rate(
        per_device_batch=64,
        reference_batch=16,
        reference_lr=1.5e-4,
        scale="linear",
    )
    assert lr == pytest.approx(6.0e-4)


def test_warmup_lr_ramp():
    import torch

    param = torch.nn.Parameter(torch.zeros(1))
    optim = torch.optim.SGD([param], lr=0.0)
    base = [1.0e-3]
    apply_warmup_lr(optim, step=0, warmup_steps=100, base_lrs=base)
    assert optim.param_groups[0]["lr"] == pytest.approx(1.0e-5)
    apply_warmup_lr(optim, step=99, warmup_steps=100, base_lrs=base)
    assert optim.param_groups[0]["lr"] == pytest.approx(1.0e-3)


def test_build_cosine_scheduler():
    import torch
    from phi0.training.lr_schedule import build_lr_scheduler

    param = torch.nn.Parameter(torch.zeros(1))
    optim = torch.optim.AdamW([param], lr=1e-4)
    sched = build_lr_scheduler(
        optim, scheduler_type="cosine", max_steps=100, warmup_steps=10
    )
    assert sched is not None
    lrs = []
    for _ in range(100):
        optim.step()
        sched.step()
        lrs.append(optim.param_groups[0]["lr"])
    assert lrs[0] < 1e-4
    assert lrs[9] == pytest.approx(1e-4, rel=0.05)
    assert lrs[-1] < lrs[50]


def test_async_gpu_prep_pipeline_cpu_fallback():
    calls = {"n": 0}

    def prepare_gpu(cpu_payload):
        calls["n"] += 1
        return {"id": cpu_payload["id"], "ready": True}

    cpu_batches = [{"id": 0}, {"id": 1}, {"id": 2}]
    pipeline = AsyncGpuPrepPipeline(
        iter(cpu_batches),
        prepare_gpu,
        device=__import__("torch").device("cpu"),
        queue_depth=2,
    )
    out = list(pipeline)
    assert len(out) == 3
    assert calls["n"] == 3
    assert out[-1]["id"] == 2


def test_gpu_batch_pipeline_cpu_fallback():
    calls = {"n": 0}

    def prepare_gpu(cpu_payload):
        calls["n"] += 1
        return {"id": cpu_payload["id"], "ready": True}

    cpu_batches = [{"id": 0}, {"id": 1}, {"id": 2}]
    pipeline = iter_training_batches(
        iter([]),
        prepare_cpu=lambda b: b,
        prepare_gpu=prepare_gpu,
        device=__import__("torch").device("cpu"),
        cpu_prefetch=0,
        gpu_pipeline=True,
    )
    # Build pipeline manually from cpu list
    pipeline = GpuBatchPipeline(iter(cpu_batches), prepare_gpu, device=__import__("torch").device("cpu"))
    out = list(pipeline)
    assert len(out) == 3
    assert calls["n"] == 3
    assert out[-1]["id"] == 2
