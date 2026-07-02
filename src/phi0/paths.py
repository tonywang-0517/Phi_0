"""Local workspace roots (override with PHI0_WORKSPACE)."""

from __future__ import annotations

import os
from pathlib import Path

# ponytail: one env var, default /home/user — not the cluster /mnt/data* layout.
_DEFAULT_WORKSPACE = Path("/home/user")

# Training checkpoints may embed these; remap on load for local inference.
_CLUSTER_WORKSPACE_PREFIXES = (
    "/mnt/data2/wpy/workspace",
)


def workspace_root() -> Path:
    return Path(os.environ.get("PHI0_WORKSPACE", str(_DEFAULT_WORKSPACE))).expanduser().resolve()


def remap_workspace_path(path: str | Path) -> str:
    """Rewrite cluster workspace prefixes to ``PHI0_WORKSPACE``."""
    s = str(path).replace("\\", "/")
    for prefix in _CLUSTER_WORKSPACE_PREFIXES:
        if s.startswith(prefix):
            rel = s[len(prefix) :].lstrip("/")
            return str(workspace_root() / rel) if rel else str(workspace_root())
    return s
