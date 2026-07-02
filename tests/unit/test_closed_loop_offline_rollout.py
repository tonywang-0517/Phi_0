"""Offline closed-loop rollout schedule helpers."""

from __future__ import annotations

import numpy as np

from phi0_sonic_closed_loop_offline_rollout import (
    build_inference_schedule,
    resolve_max_frames,
)


def test_build_inference_schedule_stable_sort():
    ctrl = np.asarray([24, 0, 0, 47], dtype=np.int32)
    assert build_inference_schedule(ctrl) == [(0, 1), (0, 2), (24, 0), (47, 3)]


def test_resolve_max_frames_from_reference():
    n = resolve_max_frames(
        control_idx=np.asarray([0, 24], dtype=np.int32),
        chunk_h=32,
        max_frames=0,
        reference_output=None,
    )
    assert n == 24 + 32
