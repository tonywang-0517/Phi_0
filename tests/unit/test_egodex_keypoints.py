"""Unit tests for EgoDex quat cache -> keypoints conversion."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from phi0.data.egodex import EgoDexDataset
from phi0.data.egodex_keypoints import (
    pack_egodex_keypoints_d_raw,
    quat_dim_available_to_keypoint_dim_available,
)
from phi0.schema.action_schema import KEYPOINTS_FLAT_DIM


def _find_processed_hdf5() -> Path | None:
    candidates = [
        Path("/mnt/data1/wpy/workspace/Isaac-GR00T/demo_data/egodex/test/add_remove_lid/0_smplh.hdf5"),
        Path("/mnt/data2/wpy/workspace/Isaac-GR00T/demo_data/egodex/test/add_remove_lid/0_smplh.hdf5"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def test_quat_mask_maps_to_keypoint_xyz_blocks():
    dim = np.zeros(256, dtype=bool)
    dim[7:11] = True  # body_quats[0] -> joint 1
    dim[91:95] = True  # left hand local 0 -> joint 22
    kp_mask = quat_dim_available_to_keypoint_dim_available(dim)
    assert kp_mask.shape == (KEYPOINTS_FLAT_DIM,)
    assert kp_mask[3:6].all()
    assert kp_mask[22 * 3 : 22 * 3 + 3].all()
    assert not kp_mask[6:9].any()


def test_pack_egodex_keypoints_from_real_cache():
    path = _find_processed_hdf5()
    if path is None:
        return
    with h5py.File(path, "r") as f:
        d_quat = f["d_raw"][0]
        dim = f["dim_available_frame"][0].astype(bool)
    d_raw, dim_avail = pack_egodex_keypoints_d_raw(d_quat, dim)
    kp_supervised = dim_avail[:KEYPOINTS_FLAT_DIM].sum()
    assert kp_supervised > 0
    assert kp_supervised < KEYPOINTS_FLAT_DIM
    assert np.linalg.norm(d_raw[:KEYPOINTS_FLAT_DIM]) > 0


def test_egodex_dataset_exposes_sparse_keypoints():
    path = _find_processed_hdf5()
    if path is None:
        return
    raw = path.with_name("0.hdf5")
    if not raw.is_file():
        return
    ds = EgoDexDataset(hdf5_path=raw, processed_hdf5=path, max_frames=2, cache_video=False)
    sample = ds[0]
    avail = (~sample["action_dim_is_pad"]).sum().item()
    assert 0 < avail < KEYPOINTS_FLAT_DIM
    assert sample["action"][0, :KEYPOINTS_FLAT_DIM].abs().sum() > 0
