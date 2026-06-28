"""Map pick-tissue manifest (session, src_ep) -> unified LeRobot episode_index."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


def _load_manifest_episodes(manifest_path: Path) -> list[tuple[str, int]]:
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    out: list[tuple[str, int]] = []
    for session_id, info in manifest.items():
        for ep_idx in info.get("valid", []):
            out.append((str(session_id), int(ep_idx)))
    return out


@lru_cache(maxsize=4)
def _sorted_valid_parquets(valid_root: str) -> tuple[Path, ...]:
    root = Path(valid_root)
    return tuple(sorted((root / "data").rglob("episode_*.parquet")))


def manifest_ep_to_dst_ep(manifest_path: str | Path, session_id: str, src_ep: int) -> int:
    """``prepare_pick_tissue_dataset`` destination episode index (parquet filename)."""
    key = (str(session_id), int(src_ep))
    for dst_ep, ep in enumerate(_load_manifest_episodes(Path(manifest_path))):
        if ep == key:
            return dst_ep
    raise KeyError(f"manifest episode not found: {session_id} ep {src_ep}")


def dst_ep_to_unified_episode_index(valid_root: str | Path, dst_ep: int) -> int:
    """Unified rebuild index = sorted parquet position (not filename number when gaps exist)."""
    valid_root = Path(valid_root)
    target = valid_root / "data" / "chunk-000" / f"episode_{int(dst_ep):06d}.parquet"
    files = _sorted_valid_parquets(str(valid_root))
    for i, pq in enumerate(files):
        if pq.resolve() == target.resolve():
            return i
    raise FileNotFoundError(f"{target} not in sorted valid parquets ({len(files)} files)")


def manifest_ep_to_unified_episode_index(
    manifest_path: str | Path,
    valid_root: str | Path,
    session_id: str,
    src_ep: int,
) -> int:
    dst = manifest_ep_to_dst_ep(manifest_path, session_id, src_ep)
    return dst_ep_to_unified_episode_index(valid_root, dst)
