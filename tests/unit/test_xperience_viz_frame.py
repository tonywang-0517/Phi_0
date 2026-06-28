"""Tests for per-frame FK → keypoints Sim(3) alignment."""

from __future__ import annotations

import numpy as np
import pytest

from phi0.viz.xperience_viz_frame import (
    Sim3,
    align_fk_joints_to_keypoints_frame,
    apply_sim3,
    fit_sim3_procrustes,
    keypoints_joint0_from_root,
)


def test_sim3_roundtrip():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 3))
    sim = Sim3(
        scale=1.23,
        rotation=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
        translation=np.array([0.1, -0.2, 0.3], dtype=np.float32),
    )
    dst = sim.apply(src)
    est = fit_sim3_procrustes(src, dst)
    pred = est.apply(src)
    assert np.allclose(pred, dst, atol=1e-5)


def test_keypoints_joint0_is_root_quat_xyz():
    root = np.array([0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    assert np.allclose(keypoints_joint0_from_root(root), root[4:7])


def test_align_preserves_target_joint0():
    fk = np.random.randn(52, 3).astype(np.float32)
    target = np.random.randn(52, 3).astype(np.float32)
    aligned, _ = align_fk_joints_to_keypoints_frame(fk, target)
    assert np.allclose(aligned[0], target[0])
    assert aligned.shape == (52, 3)


@pytest.mark.skipif(
    not __import__("pathlib").Path(
        "/mnt/data2/wpy/workspace/Isaac-GR00T/demo_data/xperience-10m-sample/annotation.hdf5"
    ).is_file(),
    reason="demo HDF5 not available",
)
def test_demo_hdf5_fk_to_keypoints_under_5cm():
    import h5py

    from phi0.data.xperience_unified_gt import reference_joints_world_from_hdf5_quat
    from phi0.viz.smplh_fk import load_skeleton_constants
    from phi0.viz.xperience_viz_frame import procrustes_joint_errors

    hdf5 = "/mnt/data2/wpy/workspace/Isaac-GR00T/demo_data/xperience-10m-sample/annotation.hdf5"
    constants = load_skeleton_constants()
    with h5py.File(hdf5, "r") as f:
        for t in (0, 16, 45, 100):
            fk = reference_joints_world_from_hdf5_quat(f, t, constants=constants)
            kp = f["full_body_mocap/keypoints"][t][:]
            err = procrustes_joint_errors(fk, kp)
            assert float(err.mean()) < 0.05, f"t={t} mean err {err.mean():.3f}m"
