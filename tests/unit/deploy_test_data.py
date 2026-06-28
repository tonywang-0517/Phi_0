"""Synthetic HDF5 / unified rows for deploy pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def write_synthetic_xperience_hdf5(path: Path, *, num_frames: int = 32) -> None:
    caption = json.dumps({"config": {"Main Task": "synthetic test task"}})

    def _identity_quats(n: int) -> np.ndarray:
        q = np.zeros((n, 4), dtype=np.float32)
        q[:, 0] = 1.0
        return q

    with h5py.File(path, "w") as f:
        roots = np.zeros((num_frames, 7), dtype=np.float32)
        for t in range(num_frames):
            roots[t, :3] = np.array([0.1 * t, -0.05 * t, 0.9 + 0.01 * t], dtype=np.float32)
            roots[t, 3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        f.create_dataset("full_body_mocap/Ts_world_root", data=roots)
        f.create_dataset(
            "full_body_mocap/body_quats", data=_identity_quats(21)[None].repeat(num_frames, 0)
        )
        f.create_dataset(
            "full_body_mocap/left_hand_quats", data=_identity_quats(15)[None].repeat(num_frames, 0)
        )
        f.create_dataset(
            "full_body_mocap/right_hand_quats", data=_identity_quats(15)[None].repeat(num_frames, 0)
        )
        f.create_dataset("full_body_mocap/betas", data=np.zeros((num_frames, 16), dtype=np.float32))
        contacts = np.zeros((num_frames, 21), dtype=np.float32)
        contacts[:, 5] = 1.0
        f.create_dataset("full_body_mocap/contacts", data=contacts)
        mano = np.random.RandomState(0).randn(num_frames, 21, 3).astype(np.float32) * 0.02
        f.create_dataset("hand_mocap/left_joints_3d", data=mano)
        f.create_dataset("hand_mocap/right_joints_3d", data=mano + 0.1)
        f.create_dataset("caption", data=np.array(caption.encode("utf-8")))
