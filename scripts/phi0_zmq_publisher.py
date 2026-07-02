#!/usr/bin/env python3
"""Launcher for Phi-0 -> HGPT ZMQ qpos publisher (experiments migration)."""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parent.parent / "experiments/phi0_hgpt_zmq/phi0_zmq_publisher.py"),
    run_name="__main__",
)
