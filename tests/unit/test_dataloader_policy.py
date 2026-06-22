"""Tests for memory-aware DataLoader policy."""

from __future__ import annotations

from types import SimpleNamespace

from phi0.data.dataloader_policy import (
    estimate_libero_dataset_copies,
    resolve_dataloader_settings,
)


def _cfg(**overrides):
    data = {
        "dataset": "libero_spatial",
        "libero_cache_native_frames": False,
        "libero_cache_auto_disable": True,
        "libero_cache_max_copies": 8,
    }
    base = {
        "num_workers": 8,
        "prefetch_factor": 4,
        "dataloader_max_dataset_copies": 24,
        "dataloader_max_workers_per_rank": 2,
        "data": data,
    }
    base.update(overrides)
    if "data" in overrides:
        data.update(overrides["data"])
        base["data"] = data
    return SimpleNamespace(**base)


def _dist(world_size: int):
    return SimpleNamespace(enabled=True, world_size=world_size, is_main=True)


def test_ddp8_caps_workers_to_two_per_rank():
    settings = resolve_dataloader_settings(_cfg(), dist_ctx=_dist(8))
    assert settings.num_workers == 2
    assert estimate_libero_dataset_copies(world_size=8, num_workers=2) == 24


def test_cache_disabled_when_too_many_copies_even_if_requested():
    cfg = _cfg(data={"libero_cache_native_frames": True})
    settings = resolve_dataloader_settings(cfg, dist_ctx=_dist(8))
    assert settings.cache_native_frames is False
    assert any("libero_cache_native_frames disabled" in n for n in settings.notes)


def test_small_single_gpu_can_enable_cache():
    cfg = _cfg(num_workers=2, data={"libero_cache_native_frames": True})
    settings = resolve_dataloader_settings(cfg, dist_ctx=None)
    assert settings.num_workers == 2
    assert settings.cache_native_frames is True
