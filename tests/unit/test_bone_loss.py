"""Unit tests for bone-length structure loss."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from phi0.schema.action_schema import D_RAW, KEYPOINTS_FLAT_DIM
from phi0.losses.bone import bone_direction_loss, bone_length_loss, hand_bone_length_loss, hand_keypoints_mse_loss
from phi0.losses.bone import HAND_BONE_CHILD_INDICES


def _make_keypoints_batch(values: np.ndarray) -> torch.Tensor:
    out = torch.zeros(1, 1, D_RAW)
    out[..., :KEYPOINTS_FLAT_DIM] = torch.from_numpy(values.reshape(-1))
    return out


def test_bone_loss_zero_when_lengths_match():
    rng = np.random.RandomState(1)
    kp = rng.randn(52, 3).astype(np.float32)
    pred = _make_keypoints_batch(kp)
    target = pred.clone()
    loss = bone_length_loss(pred, target)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_bone_loss_positive_when_stretched():
    kp_pred = np.zeros((52, 3), dtype=np.float32)
    kp_tgt = np.zeros((52, 3), dtype=np.float32)
    kp_pred[1, 0] = 2.0
    kp_tgt[1, 0] = 1.0
    pred = _make_keypoints_batch(kp_pred)
    target = _make_keypoints_batch(kp_tgt)
    loss = bone_length_loss(pred, target)
    assert loss.item() > 0.0


def test_bone_loss_skips_unsupervised_dims():
    kp = np.zeros((52, 3), dtype=np.float32)
    kp[1, 0] = 1.0
    pred = _make_keypoints_batch(kp)
    target = torch.zeros_like(pred)
    dim_pad = torch.ones(D_RAW, dtype=torch.bool)
    loss = bone_length_loss(pred, target, action_dim_is_pad=dim_pad)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_bone_loss_uses_denorm_for_lengths():
    kp_pred = np.zeros((52, 3), dtype=np.float32)
    kp_tgt = np.zeros((52, 3), dtype=np.float32)
    kp_pred[1, 0] = 2.0
    kp_tgt[1, 0] = 1.0
    pred = _make_keypoints_batch(kp_pred)
    target = _make_keypoints_batch(kp_tgt)
    mean = torch.zeros(D_RAW)
    std = torch.ones(D_RAW)
    std[3] = 0.1
    loss_norm = bone_length_loss(pred, target, norm_mean=mean, norm_std=std)
    pred_raw = pred.clone()
    pred_raw[..., 3] = pred[..., 3] * 0.1
    target_raw = target.clone()
    target_raw[..., 3] = target[..., 3] * 0.1
    loss_raw = bone_length_loss(pred_raw, target_raw)
    assert loss_norm.item() == pytest.approx(loss_raw.item(), rel=1e-4)


def test_bone_direction_zero_when_parallel():
    kp = np.zeros((52, 3), dtype=np.float32)
    kp[1, 0] = 1.0
    pred = _make_keypoints_batch(kp)
    target = pred.clone()
    loss = bone_direction_loss(pred, target)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_hand_bone_loss_only_hand_edges():
    kp_pred = np.zeros((52, 3), dtype=np.float32)
    kp_tgt = np.zeros((52, 3), dtype=np.float32)
    kp_pred[45, 0] = 2.0
    kp_tgt[45, 0] = 1.0
    pred = _make_keypoints_batch(kp_pred)
    target = _make_keypoints_batch(kp_tgt)
    assert hand_bone_length_loss(pred, target).item() > 0.0

    kp_body = kp_pred.copy()
    kp_body[45, 0] = 0.0
    kp_body[1, 0] = 2.0
    kp_tgt_body = kp_tgt.copy()
    kp_tgt_body[45, 0] = 0.0
    kp_tgt_body[1, 0] = 1.0
    pred_b = _make_keypoints_batch(kp_body)
    target_b = _make_keypoints_batch(kp_tgt_body)
    assert hand_bone_length_loss(pred_b, target_b).item() == pytest.approx(0.0, abs=1e-6)
    assert bone_length_loss(pred_b, target_b).item() > 0.0


def test_hand_bone_child_indices_cover_fingers():
    assert len(HAND_BONE_CHILD_INDICES) == 30
    assert HAND_BONE_CHILD_INDICES[0] == 22
    assert HAND_BONE_CHILD_INDICES[-1] == 51


def test_hand_keypoints_mse_zero_when_matching():
    kp = np.zeros((52, 3), dtype=np.float32)
    kp[30, 0] = 1.5
    pred = _make_keypoints_batch(kp)
    target = pred.clone()
    assert hand_keypoints_mse_loss(pred, target).item() == pytest.approx(0.0, abs=1e-6)


def test_hand_keypoints_mse_only_hand_dims():
    kp_pred = np.zeros((52, 3), dtype=np.float32)
    kp_tgt = np.zeros((52, 3), dtype=np.float32)
    kp_pred[30, 0] = 2.0
    kp_tgt[30, 0] = 1.0
    pred = _make_keypoints_batch(kp_pred)
    target = _make_keypoints_batch(kp_tgt)
    assert hand_keypoints_mse_loss(pred, target).item() > 0.0

    kp_body = kp_pred.copy()
    kp_body[30, 0] = 0.0
    kp_body[1, 0] = 2.0
    kp_tgt_body = kp_tgt.copy()
    kp_tgt_body[30, 0] = 0.0
    kp_tgt_body[1, 0] = 1.0
    pred_b = _make_keypoints_batch(kp_body)
    target_b = _make_keypoints_batch(kp_tgt_body)
    assert hand_keypoints_mse_loss(pred_b, target_b).item() == pytest.approx(0.0, abs=1e-6)


def test_hand_keypoints_mse_skips_unsupervised_dims():
    kp = np.zeros((52, 3), dtype=np.float32)
    kp[30, 0] = 2.0
    pred = _make_keypoints_batch(kp)
    target = torch.zeros_like(pred)
    dim_pad = torch.ones(D_RAW, dtype=torch.bool)
    loss = hand_keypoints_mse_loss(pred, target, action_dim_is_pad=dim_pad)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
