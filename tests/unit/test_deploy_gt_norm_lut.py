"""Unit tests for deploy GT norm LUT index planning."""

from __future__ import annotations

from phi0.deploy.gt_io import deploy_gt_norm_lut_indices
from phi0.inference.deploy_align import deploy_history_control_indices


def test_deploy_gt_norm_lut_indices_matches_history_loop():
    proprio_w, chunk_h, history_w = 4, 32, 4
    num_frames = 100
    expected: set[int] = set()
    for seg_start in range(0, num_frames, chunk_h):
        deploy_c = proprio_w + seg_start
        expected.update(deploy_history_control_indices(deploy_c, history_w))
    got = set(
        deploy_gt_norm_lut_indices(
            num_frames=num_frames,
            proprio_w=proprio_w,
            chunk_h=chunk_h,
            history_w=history_w,
        )
    )
    assert got == expected
    assert len(got) < proprio_w + num_frames


def test_deploy_gt_norm_lut_indices_empty_without_history():
    assert (
        deploy_gt_norm_lut_indices(
            num_frames=400,
            proprio_w=4,
            chunk_h=32,
            history_w=0,
        )
        == []
    )
