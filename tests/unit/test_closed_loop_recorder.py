"""Closed-loop recorder persistence."""

from __future__ import annotations

import numpy as np

from phi0_sonic_closed_loop_zmq import ClosedLoopRecorder, ObsSnapshot


def test_closed_loop_recorder_saves_inference_elapsed(tmp_path):
    rec = ClosedLoopRecorder(
        prompt="pick tissue",
        camera_source="gt",
        control_fps=50.0,
        checkpoint="ckpt.pt",
    )
    ego = np.zeros((48, 64, 3), dtype=np.uint8)
    rec.record_observation(
        ObsSnapshot(control_idx=0, ego_hwc=ego, wrist_hwc=None, timestamp=1.0),
        inference_elapsed_s=0.42,
    )
    rec.record_observation(
        ObsSnapshot(control_idx=24, ego_hwc=ego, wrist_hwc=None, timestamp=2.0),
        inference_elapsed_s=0.88,
    )
    obs_path = tmp_path / "observations.npz"
    out_path = tmp_path / "outputs.npz"
    rec.save(obs_path, out_path)

    data = np.load(obs_path)
    assert data["inference_elapsed_s"].shape == (2,)
    assert np.allclose(data["inference_elapsed_s"], [0.42, 0.88])


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        test_closed_loop_recorder_saves_inference_elapsed(Path(d))
    print("ok")
