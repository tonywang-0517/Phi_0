"""Unit tests for pick-tissue manifest -> unified episode index."""

from __future__ import annotations

from phi0.data.pick_tissue_episode_map import (
    manifest_ep_to_dst_ep,
    manifest_ep_to_unified_episode_index,
)


def test_manifest_ep2_session_20260625_160943():
    manifest = "/mnt/data2/wpy/workspace/Isaac-GR00T/data/data.json"
    valid = "/mnt/data2/wpy/workspace/Isaac-GR00T/data/pick_tissue_valid"
    dst = manifest_ep_to_dst_ep(manifest, "2026-06-25-16-09-43", 2)
    assert dst == 524
    uni = manifest_ep_to_unified_episode_index(manifest, valid, "2026-06-25-16-09-43", 2)
    assert uni == 448
