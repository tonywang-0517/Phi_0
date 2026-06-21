from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class VLAObservation:
    """Canonical observation format aligned with VLA-Adapter eval helpers."""

    full_image: np.ndarray
    wrist_image: np.ndarray | None
    state: np.ndarray
    raw: dict[str, Any]


@dataclass
class VLAActionChunk:
    """Open-loop action chunk in [T, 7] (dx, dy, dz, droll, dpitch, dyaw, grip)."""

    actions: np.ndarray

