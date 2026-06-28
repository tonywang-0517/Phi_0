"""Shared fixtures for deploy / ZMQ / GT pipeline unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from deploy_test_data import write_synthetic_xperience_hdf5

_GR00T_ROOT = Path(__file__).resolve().parents[3] / "GR00T-WholeBodyControl"
if _GR00T_ROOT.is_dir() and str(_GR00T_ROOT) not in sys.path:
    sys.path.insert(0, str(_GR00T_ROOT))


@pytest.fixture
def synthetic_hdf5(tmp_path: Path) -> Path:
    path = tmp_path / "annotation.hdf5"
    write_synthetic_xperience_hdf5(path)
    return path


@pytest.fixture
def skeleton_constants():
    from phi0.viz.smplh_fk import load_skeleton_constants

    try:
        return load_skeleton_constants()
    except FileNotFoundError:
        pytest.skip("SMPL-H skeleton constants not available")
