"""Memory-aware DataLoader settings for LIBERO / DDP training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

LIBERO_DATASET_NAMES = frozenset({"libero_spatial", "libero", "libero_rlds"})


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _data_get(cfg: Any, key: str, default: Any = None) -> Any:
    data = _cfg_get(cfg, "data", {})
    if hasattr(data, "get"):
        return data.get(key, default)
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


@dataclass(frozen=True)
class DataLoaderSettings:
    num_workers: int
    prefetch_factor: int
    cache_native_frames: bool
    notes: tuple[str, ...] = ()


def is_libero_dataset(cfg: Any) -> bool:
    name = str(_data_get(cfg, "dataset", "")).strip().lower()
    return name in LIBERO_DATASET_NAMES


def _world_size(dist_ctx: Any) -> int:
    if dist_ctx is not None and getattr(dist_ctx, "enabled", False):
        return max(1, int(dist_ctx.world_size))
    return 1


def estimate_libero_dataset_copies(*, world_size: int, num_workers: int) -> int:
    """Rank process + DataLoader workers each hold one dataset instance."""
    return int(world_size) * (1 + max(0, int(num_workers)))


def resolve_dataloader_settings(cfg: Any, *, dist_ctx: Any = None) -> DataLoaderSettings:
    """Balance throughput vs RAM for LIBERO RLDS + optional native-frame cache."""
    notes: List[str] = []
    world_size = _world_size(dist_ctx)

    requested_workers = max(0, int(_cfg_get(cfg, "num_workers", 0)))
    max_total_copies = max(1, int(_cfg_get(cfg, "dataloader_max_dataset_copies", 24)))
    per_rank_cap = max(0, int(_cfg_get(cfg, "dataloader_max_workers_per_rank", 2)))

    worker_budget = max(0, max_total_copies - world_size)
    if world_size > 1 and requested_workers > 0:
        per_rank = min(requested_workers, per_rank_cap, worker_budget // world_size)
    else:
        per_rank = requested_workers

    num_workers = max(0, per_rank)
    if num_workers != requested_workers:
        notes.append(
            f"num_workers {requested_workers}->{num_workers} "
            f"(world_size={world_size}, max_copies={max_total_copies}, per_rank_cap={per_rank_cap})"
        )

    prefetch_factor = max(1, int(_cfg_get(cfg, "prefetch_factor", 2)))

    cache_native_frames = False
    if is_libero_dataset(cfg):
        requested_cache = bool(_data_get(cfg, "libero_cache_native_frames", False))
        auto_disable = bool(_data_get(cfg, "libero_cache_auto_disable", True))
        max_cache_copies = max(1, int(_data_get(cfg, "libero_cache_max_copies", 8)))
        copies = estimate_libero_dataset_copies(
            world_size=world_size, num_workers=num_workers
        )
        cache_native_frames = requested_cache
        if cache_native_frames and auto_disable and copies > max_cache_copies:
            cache_native_frames = False
            notes.append(
                "libero_cache_native_frames disabled "
                f"({copies} dataset copies > max_cache_copies={max_cache_copies}); "
                "episodes stay uint8-only (~20 GiB/copy dual-cam)"
            )

    return DataLoaderSettings(
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        cache_native_frames=cache_native_frames,
        notes=tuple(notes),
    )


def log_dataloader_settings(settings: DataLoaderSettings, *, logger: Any, dist_ctx: Any = None) -> None:
    if dist_ctx is not None and not getattr(dist_ctx, "is_main", True):
        return
    world_size = _world_size(dist_ctx)
    copies = estimate_libero_dataset_copies(
        world_size=world_size, num_workers=settings.num_workers
    )
    logger.info(
        "DataLoader policy: num_workers=%d prefetch_factor=%d "
        "libero_cache_native_frames=%s (~%d dataset copies)",
        settings.num_workers,
        settings.prefetch_factor,
        settings.cache_native_frames,
        copies,
    )
    for note in settings.notes:
        logger.info("DataLoader policy: %s", note)
