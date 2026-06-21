"""Canonical benchmark data paths for LIBERO / CALVIN."""

from __future__ import annotations

import os
from pathlib import Path

PHI0_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("PHI0_DATA_ROOT", PHI0_ROOT / "data"))
LIBERO_RLDS_ROOT = DATA_ROOT / "libero" / "modified_libero_rlds"
CALVIN_RLDS_ROOT = DATA_ROOT / "calvin" / "calvin_abc_rlds"
CALVIN_HOME = Path(os.environ.get("CALVIN_HOME", DATA_ROOT / "calvin"))
THIRD_PARTY_CALVIN = PHI0_ROOT / "third_party" / "calvin"

LIBERO_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)


def libero_rlds_dir(suite: str) -> Path:
    name = suite if suite.endswith("_no_noops") else f"{suite}_no_noops"
    return LIBERO_RLDS_ROOT / name / "1.0.0"


def calvin_eval_root() -> Path:
    """Root passed to CALVIN eval (contains calvin_models/ + dataset/)."""
    if (CALVIN_HOME / "calvin_models").is_dir():
        return CALVIN_HOME
    if (THIRD_PARTY_CALVIN / "calvin_models").is_dir():
        return THIRD_PARTY_CALVIN
    return CALVIN_HOME


def calvin_validation_dir() -> Path:
    return calvin_eval_root() / "dataset" / "task_ABC_D" / "validation"


def calvin_models_dir() -> Path:
    return calvin_eval_root() / "calvin_models"
