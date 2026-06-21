"""Unit tests for robot 7D normalization, delta stats, loss/deploy paths."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phi0.benchmark.rlds_adapters import libero_rlds_action_to_train, libero_rlds_state_to_eef_7d
from phi0.data.action_stats import (
    compute_action_stats_from_datasets,
    load_or_validate_stats,
    save_action_stats,
)
from phi0.data.processor import Phi0Processor
from phi0.data.robot_action_norm import (
    STATS_SEMANTICS_ABSOLUTE,
    STATS_SEMANTICS_DELTA,
    STATS_SEMANTICS_PROPRIO,
    denormalize_robot7d,
    normalize_robot7d,
    validate_stats_for_cfg,
)
from phi0.data.sequence import SequenceDataset
from phi0.models.phi0 import Phi0
from phi0.runtime import _normalize_libero_proprio_delta_batch


class _FakeLiberoFrameDataset:
    DATASET_NAME = "libero_spatial"

    def __init__(self, frames: list[dict]) -> None:
        self._frames = frames

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> dict:
        return self._frames[idx]


def _frame(state: np.ndarray, action: np.ndarray) -> dict:
    return {
        "dataset": "libero_spatial",
        "idx": 0,
        "task": "task",
        "robot_proprio_7d": torch.from_numpy(libero_rlds_state_to_eef_7d(state)).view(1, 7),
        "robot_delta_7d": torch.from_numpy(libero_rlds_action_to_train(action)).view(1, 7),
        "action": torch.zeros(1, 256),
        "action_dim_is_pad": torch.zeros(256, dtype=torch.bool),
        "images": {"ego_view": torch.zeros(1, 3, 8, 8)},
    }


def _delta_data_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        libero_delta_eef=True,
        libero_absolute_eef=False,
        action_norm_mode="bounds_q99",
        proprio_norm_mode="z-score",
    )


def _absolute_data_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        libero_delta_eef=False,
        libero_absolute_eef=True,
        action_norm_mode="z-score",
        proprio_norm_mode="z-score",
    )


def _make_synthetic_delta_frames(n: int = 64) -> _FakeLiberoFrameDataset:
    rng = np.random.default_rng(0)
    frames = []
    for i in range(n):
        state = np.array([i * 0.01, 0.1, 1.0, 0.0, 0.0, 0.0, 0.04, -0.04], dtype=np.float32)
        action = rng.normal(0, 0.05, size=(7,)).astype(np.float32)
        action[6] = float(i % 2)
        frames.append(_frame(state, action))
    return _FakeLiberoFrameDataset(frames)


def _bounds_stats() -> dict:
    stats = compute_action_stats_from_datasets(
        [_make_synthetic_delta_frames()],
        robot_action_semantics=STATS_SEMANTICS_DELTA,
        norm_mode="bounds_q99",
        normalize_gripper=False,
    )
    return stats


def _zscore_stats() -> dict:
    mean = torch.zeros(256)
    std = torch.ones(256)
    mean[:7] = torch.tensor([0.1, 0.2, 1.0, 0.0, 0.0, 0.0, 0.5])
    std[:7] = torch.tensor([0.05, 0.05, 0.1, 0.01, 0.01, 0.01, 0.5])
    from phi0.data.robot_action_norm import stats_dict_from_tensors

    return stats_dict_from_tensors(
        mean=mean,
        std=std,
        q01=mean - 0.1,
        q99=mean + 0.1,
        robot_action_semantics=STATS_SEMANTICS_PROPRIO,
        norm_mode="z-score",
        num_frames=100,
        normalize_gripper=True,
    )


def test_bounds_q99_roundtrip():
    stats = _bounds_stats()
    lo = torch.tensor(stats["q01"][:6], dtype=torch.float32)
    hi = torch.tensor(stats["q99"][:6], dtype=torch.float32)
    raw = lo + 0.5 * (hi - lo)
    raw = torch.cat([raw, torch.tensor([1.0])]).view(1, 7)
    norm = normalize_robot7d(raw, stats, normalize_gripper=False)
    assert norm[0, 6] == raw[0, 6]
    recon = denormalize_robot7d(norm, stats, denormalize_gripper=False)
    assert torch.allclose(recon[:, :6], raw[:, :6], atol=1e-4)
    assert recon[0, 6] == 1.0


def test_zscore_roundtrip():
    stats = _zscore_stats()
    raw = torch.tensor([[0.15, 0.25, 1.05, 0.01, -0.01, 0.02, 0.8]], dtype=torch.float32)
    norm = normalize_robot7d(raw, stats, normalize_gripper=True)
    recon = denormalize_robot7d(norm, stats, denormalize_gripper=True)
    assert torch.allclose(recon, raw, atol=1e-5)


def test_delta_stats_only_supervise_dims_0_to_5():
    stats = _bounds_stats()
    mask = stats["supervised_mask"]
    assert mask[:6] == [True] * 6
    assert mask[6] is False


def test_validate_stats_rejects_absolute_for_delta_cfg():
    stats = _zscore_stats()
    stats["robot_action_semantics"] = STATS_SEMANTICS_ABSOLUTE
    with pytest.raises(ValueError, match="semantics mismatch"):
        validate_stats_for_cfg(_delta_data_cfg(), stats, proprio=False)


def test_validate_stats_rejects_wrong_norm_mode():
    stats = _bounds_stats()
    stats["norm_mode"] = "z-score"
    with pytest.raises(ValueError, match="norm_mode mismatch"):
        validate_stats_for_cfg(_delta_data_cfg(), stats, proprio=False)


def test_load_or_validate_stats_ok(tmp_path: Path):
    stats = _bounds_stats()
    path = tmp_path / "delta_stats.json"
    save_action_stats(stats, path)
    loaded = load_or_validate_stats(path, _delta_data_cfg(), proprio=False)
    assert loaded is not None
    assert loaded["robot_action_semantics"] == STATS_SEMANTICS_DELTA


def test_ensure_stats_recompute_on_mismatch(tmp_path: Path):
    bad = _zscore_stats()
    bad["robot_action_semantics"] = STATS_SEMANTICS_ABSOLUTE
    path = tmp_path / "bad_delta_stats.json"
    save_action_stats(bad, path)
    with pytest.raises(ValueError):
        load_or_validate_stats(path, _delta_data_cfg(), proprio=False)


def test_processor_delta_denorm_future():
    proc = Phi0Processor(normalize=True)
    proc.register_stats_from_dict(_bounds_stats())
    pred = torch.zeros(1, 8, 7)
    pred[..., 0] = 0.5
    pred[..., 6] = 1.0
    d7 = proc.denormalize_robot7d_future(pred)
    assert d7.shape == (1, 8, 7)
    assert d7[0, 0, 6] == 1.0


def test_normalize_libero_proprio_delta_batch():
    proc = Phi0Processor(normalize=True)
    proc.register_stats_from_dict(_bounds_stats())
    proc.register_proprio_stats_from_dict(_zscore_stats())
    batch = {
        "robot_proprio_7d": torch.randn(2, 5, 7),
        "robot_future_delta_7d": torch.randn(2, 8, 7),
    }
    batch["robot_future_delta_7d"][..., 6] = (torch.arange(16) % 2).float().view(2, 8)
    normed, merged, future = _normalize_libero_proprio_delta_batch(proc, batch)
    assert normed.shape == (2, 13, 7)
    assert merged.shape == (2, 13, 7)
    assert future.shape == (2, 8, 7)
    assert torch.allclose(normed[:, 5:, 6], future[..., 6])


def test_robot_decoder_loss_zero_at_target():
    model = SimpleNamespace(robot_action_loss_type="l1")
    model._compute_robot_action_decoder_loss = Phi0._compute_robot_action_decoder_loss.__get__(
        model, type(model)
    )
    target = torch.tensor([[[0.5, -0.2, 0.1, 0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32)
    loss = model._compute_robot_action_decoder_loss(target, target, action_is_pad=None)
    assert loss.item() < 1e-6


def test_robot_decoder_loss_l1_nonzero():
    model = SimpleNamespace(robot_action_loss_type="l1")
    model._compute_robot_action_decoder_loss = Phi0._compute_robot_action_decoder_loss.__get__(
        model, type(model)
    )
    target = torch.zeros(1, 1, 7)
    pred_norm = torch.zeros(1, 1, 7)
    pred_norm[..., 0] = 1.0
    loss = model._compute_robot_action_decoder_loss(pred_norm, target, action_is_pad=None)
    assert loss.item() > 0.01


def test_sequence_split_proprio_delta_chunk():
    states = [np.array([i, 0, 1, 0, 0, 0, 0.04, -0.04], dtype=np.float32) for i in range(13)]
    actions = [np.array([0.01, 0, 0, 0, 0, 0, 1.0 - (i % 2)], dtype=np.float32) for i in range(13)]
    ds = _FakeLiberoFrameDataset([_frame(s, a) for s, a in zip(states, actions)])
    seq = SequenceDataset(
        ds,
        seq_len=13,
        stride=1,
        future_action_steps=8,
        native_fps={"libero_spatial": 20.0},
    )
    item = seq.sample_at_start(0)
    assert item["robot_proprio_7d"].shape == (5, 7)
    assert item["robot_future_delta_7d"].shape == (8, 7)
    expected_first = torch.from_numpy(libero_rlds_action_to_train(actions[4])).float()
    assert torch.allclose(item["robot_future_delta_7d"][0], expected_first)


def test_compute_proprio_vs_delta_field_selection():
    frame = _frame(
        np.array([1.0, 2.0, 3.0, 0, 0, 0, 0.04, -0.04], dtype=np.float32),
        np.array([0.01, 0, 0, 0, 0, 0, 1.0], dtype=np.float32),
    )
    ds = [_FakeLiberoFrameDataset([frame])]
    delta_stats = compute_action_stats_from_datasets(
        ds,
        robot_action_semantics=STATS_SEMANTICS_DELTA,
        norm_mode="bounds_q99",
        normalize_gripper=False,
        stats_field="robot_delta_7d",
    )
    proprio_stats = compute_action_stats_from_datasets(
        ds,
        robot_action_semantics=STATS_SEMANTICS_PROPRIO,
        norm_mode="z-score",
        normalize_gripper=True,
        stats_field="robot_proprio_7d",
    )
    assert delta_stats["mean"][0] == pytest.approx(0.01, abs=1e-4)
    assert proprio_stats["mean"][0] == pytest.approx(1.0, abs=1e-4)

    proc = Phi0Processor(normalize=True)
    proc.register_stats_from_dict(_bounds_stats())
    raw = torch.tensor([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    norm = proc.normalize_robot7d_tensor(raw.unsqueeze(0), proprio=False, normalize_gripper=False)
    assert norm[0, 6] == 0.0
    raw[6] = 1.0
    norm2 = proc.normalize_robot7d_tensor(raw.unsqueeze(0), proprio=False, normalize_gripper=False)
    assert norm2[0, 6] == 1.0
