"""Distributed training helpers."""

from __future__ import annotations

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from phi0.checkpoint_utils import unwrap_training_module
from phi0.distributed import DistributedContext, init_distributed, unwrap_ddp_module


def test_unwrap_training_module_strips_ddp():
    inner = nn.Linear(4, 2)
    wrapped = DDP.__new__(DDP)
    object.__setattr__(wrapped, "module", inner)
    assert unwrap_training_module(wrapped) is inner


def test_init_distributed_single_process():
    ctx = init_distributed()
    assert ctx.enabled is False
    assert ctx.world_size == 1
    assert ctx.is_main is True


def test_unwrap_ddp_module_matches_training_helper():
    inner = nn.Linear(3, 1)
    wrapped = DDP.__new__(DDP)
    object.__setattr__(wrapped, "module", inner)
    assert unwrap_ddp_module(wrapped) is inner
    assert unwrap_training_module(wrapped) is inner


def test_cleanup_distributed_noop_when_not_initialized():
    from unittest.mock import patch

    from phi0.distributed import cleanup_distributed

    with patch("phi0.distributed.dist.is_initialized", return_value=False):
        cleanup_distributed()  # should not raise


def test_distributed_context_device():
    ctx = DistributedContext(enabled=True, rank=2, world_size=4, local_rank=2)
    assert ctx.device == "cuda:2"
