"""VLA-Adapter-style torchrun / DDP helpers (PartialState + DDP wrap)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)

# Lazily created so importing phi0 does not require an active process group.
_PARTIAL_STATE = None


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> str:
        if torch.cuda.is_available():
            return f"cuda:{self.local_rank}"
        return "cpu"


def _partial_state():
    """Return HuggingFace Accelerate PartialState (same entry point as VLA-Adapter)."""
    global _PARTIAL_STATE
    if _PARTIAL_STATE is None:
        from accelerate import PartialState

        _PARTIAL_STATE = PartialState()
    return _PARTIAL_STATE


def distributed_context_from_cfg(cfg) -> DistributedContext:
    """Resolve distributed settings from Hydra config + torchrun env."""
    explicit = cfg.get("distributed")
    if explicit is not None and not bool(explicit):
        return DistributedContext(enabled=False, rank=0, world_size=1, local_rank=0)

    state = _partial_state()
    if state.num_processes <= 1:
        return DistributedContext(enabled=False, rank=0, world_size=1, local_rank=0)

    return DistributedContext(
        enabled=True,
        rank=int(state.process_index),
        world_size=int(state.num_processes),
        local_rank=int(state.local_process_index),
    )


def setup_process_device(dist_ctx: DistributedContext) -> None:
    """Match VLA-Adapter: bind each rank to its local GPU and clear cache."""
    if not torch.cuda.is_available():
        return
    torch.cuda.set_device(dist_ctx.local_rank)
    if dist_ctx.enabled:
        torch.cuda.empty_cache()


def init_distributed() -> DistributedContext:
    """Backward-compatible helper for tests / manual calls."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(enabled=False, rank=0, world_size=1, local_rank=0)
    return distributed_context_from_cfg({"distributed": True})


def barrier(dist_ctx: Optional[DistributedContext] = None) -> None:
    if dist.is_initialized():
        dist.barrier()


def cleanup_distributed(dist_ctx: Optional[DistributedContext] = None) -> None:
    """Tear down torch.distributed process group (avoids NCCL exit warning)."""
    if not dist.is_initialized():
        return
    if dist_ctx is None or dist_ctx.enabled:
        try:
            dist.barrier()
        except Exception:
            logger.debug("barrier during distributed cleanup failed", exc_info=True)
    dist.destroy_process_group()


def unwrap_ddp_module(module: nn.Module) -> nn.Module:
    """Return inner module when wrapped by DDP or ``torch.compile``."""
    from phi0.checkpoint_utils import unwrap_compiled_module

    module = unwrap_compiled_module(module)
    if isinstance(module, DDP):
        return module.module
    return module


def wrap_ddp(
    module: nn.Module,
    *,
    device_id: int,
    find_unused: bool = True,
) -> DDP:
    """Wrap ``module`` with DDP (VLA-Adapter ``wrap_ddp`` defaults)."""
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA.")
    return DDP(
        module,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=find_unused,
        gradient_as_bucket_view=True,
    )


def configure_logging_for_rank(dist_ctx: DistributedContext) -> None:
    """Only rank 0 emits INFO logs during multi-GPU training."""
    if dist_ctx.is_main:
        return
    logging.getLogger().setLevel(logging.WARNING)
    for name in ("phi0", "phi0.runtime", "phi0.models", "phi0.data"):
        logging.getLogger(name).setLevel(logging.WARNING)
