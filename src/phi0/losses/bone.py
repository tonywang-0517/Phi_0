"""Bone structure losses for keypoints action representation."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch

from phi0.schema.action_schema import KEYPOINTS_FLAT_DIM, get_action_schema
from phi0.viz.skeleton import SMPLH_PARENTS

# SMPL-H 52-joint layout: left hand 22–36, right hand 37–51 (child indices with bone edges).
HAND_JOINT_FIRST = 22
HAND_BONE_CHILD_INDICES: tuple[int, ...] = tuple(range(HAND_JOINT_FIRST, 52))


def _maybe_denorm_keypoints(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    norm_mean: torch.Tensor | None,
    norm_std: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract 52×3 keypoints; optionally denormalize per-dim z-score for metric space."""
    schema = get_action_schema()
    s, e = schema.slices["keypoints_52"]
    pred_kp = pred[..., s:e].reshape(*pred.shape[:-1], 52, 3)
    tgt_kp = target[..., s:e].reshape(*target.shape[:-1], 52, 3)
    if norm_mean is None or norm_std is None:
        return pred_kp, tgt_kp
    mean = norm_mean[s:e].to(device=pred.device, dtype=pred.dtype).view(1, 1, 52, 3)
    std = norm_std[s:e].to(device=pred.device, dtype=pred.dtype).view(1, 1, 52, 3)
    return pred_kp * std + mean, tgt_kp * std + mean


def _edge_valid_mask(
    action_dim_is_pad: torch.Tensor | None,
    parent: int,
    child: int,
    s: int,
    e: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if action_dim_is_pad is None:
        return None
    pad = action_dim_is_pad.to(device=device)
    if pad.ndim == 1:
        kp_pad = pad[s:e].reshape(52, 3)
        p_pad = kp_pad[parent]
        c_pad = kp_pad[child]
    elif pad.ndim == 2:
        p_pad = pad[:, s + parent * 3 : s + parent * 3 + 3]
        c_pad = pad[:, s + child * 3 : s + child * 3 + 3]
    else:
        p_pad = pad[..., s + parent * 3 : s + parent * 3 + 3]
        c_pad = pad[..., s + child * 3 : s + child * 3 + 3]
    return (~p_pad).all(dim=-1) & (~c_pad).all(dim=-1)


def _apply_frame_valid(
    loss: torch.Tensor,
    action_is_pad: torch.Tensor | None,
    action_dim_is_pad: torch.Tensor | None,
    s: int,
    e: int,
) -> torch.Tensor:
    valid = torch.ones_like(loss)
    if action_dim_is_pad is not None:
        pad = action_dim_is_pad[..., s:e].to(device=loss.device)
        if pad.ndim == 1:
            kp_supervised = (~pad).any().to(dtype=loss.dtype)
            valid = valid * kp_supervised
        elif pad.ndim == 2:
            kp_supervised = (~pad).any(dim=-1).to(dtype=loss.dtype)
            valid = valid * kp_supervised.unsqueeze(-1)
        else:
            kp_supervised = (~pad).any(dim=-1).to(dtype=loss.dtype)
            valid = valid * kp_supervised
    if action_is_pad is not None:
        valid = valid * (~action_is_pad).to(dtype=loss.dtype)
    denom = valid.sum().clamp(min=1.0)
    return (loss * valid).sum() / denom


def bone_length_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    action_is_pad: torch.Tensor | None = None,
    action_dim_is_pad: torch.Tensor | None = None,
    norm_mean: torch.Tensor | None = None,
    norm_std: torch.Tensor | None = None,
    child_indices: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """
    Soft SMPL-H skeleton constraint: match bone lengths to GT reference lengths.

    When ``norm_mean/std`` are set, keypoints are denormalized before length comparison
    (physical meters; required under per-dim z-score training).

    ``child_indices``: if set, only penalize edges for these child joint ids (SMPL-H).
    """
    return _bone_length_loss(
        pred,
        target,
        action_is_pad=action_is_pad,
        action_dim_is_pad=action_dim_is_pad,
        norm_mean=norm_mean,
        norm_std=norm_std,
        child_indices=child_indices,
    )


def hand_bone_length_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    action_is_pad: torch.Tensor | None = None,
    action_dim_is_pad: torch.Tensor | None = None,
    norm_mean: torch.Tensor | None = None,
    norm_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extra bone-length loss on finger / hand chains only (child joints 22–51)."""
    return _bone_length_loss(
        pred,
        target,
        action_is_pad=action_is_pad,
        action_dim_is_pad=action_dim_is_pad,
        norm_mean=norm_mean,
        norm_std=norm_std,
        child_indices=HAND_BONE_CHILD_INDICES,
    )


def hand_keypoints_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    action_is_pad: torch.Tensor | None = None,
    action_dim_is_pad: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extra masked MSE on both hands (SMPL-H keypoints joints 22–51)."""
    schema = get_action_schema()
    s, _ = schema.slices["keypoints_52"]
    hand_s = s + HAND_JOINT_FIRST * 3
    hand_e = s + 52 * 3

    pred_hand = pred[..., hand_s:hand_e]
    tgt_hand = target[..., hand_s:hand_e]
    loss = (pred_hand.float() - tgt_hand.float()).pow(2)

    dim_valid = None
    if action_dim_is_pad is not None:
        pad = action_dim_is_pad.to(device=loss.device)
        if pad.ndim == 1:
            dim_valid = (~pad[hand_s:hand_e]).to(dtype=loss.dtype).view(1, 1, -1)
        elif pad.ndim == 2:
            dim_valid = (~pad[:, hand_s:hand_e]).to(dtype=loss.dtype).unsqueeze(0)
        else:
            dim_valid = (~pad[..., hand_s:hand_e]).to(dtype=loss.dtype)
        loss = loss * dim_valid

    if action_is_pad is not None:
        token_valid = (~action_is_pad).to(device=loss.device, dtype=loss.dtype).unsqueeze(-1)
        loss = loss * token_valid

    if dim_valid is not None and action_is_pad is not None:
        token_valid = (~action_is_pad).float().unsqueeze(-1)
        denom = (dim_valid * token_valid).sum().clamp(min=1.0)
        return loss.sum() / denom
    if dim_valid is not None:
        denom = dim_valid.sum().clamp(min=1.0)
        return loss.sum() / denom
    if action_is_pad is not None:
        token_valid = (~action_is_pad).float().unsqueeze(-1)
        denom = token_valid.sum().clamp(min=1.0)
        return loss.sum() / denom
    return loss.mean()


def _bone_length_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    action_is_pad: torch.Tensor | None = None,
    action_dim_is_pad: torch.Tensor | None = None,
    norm_mean: torch.Tensor | None = None,
    norm_std: torch.Tensor | None = None,
    child_indices: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    schema = get_action_schema()
    s, e = schema.slices["keypoints_52"]
    assert e - s == KEYPOINTS_FLAT_DIM

    parents = SMPLH_PARENTS
    pred_kp, tgt_kp = _maybe_denorm_keypoints(
        pred, target, norm_mean=norm_mean, norm_std=norm_std
    )

    children: Iterable[int] = range(1, 52) if child_indices is None else child_indices

    bone_losses: list[torch.Tensor] = []
    bone_valid: list[torch.Tensor] = []
    for child in children:
        child = int(child)
        parent = int(parents[child])
        pred_edge = pred_kp[..., child, :] - pred_kp[..., parent, :]
        tgt_edge = tgt_kp[..., child, :] - tgt_kp[..., parent, :]
        pred_len = pred_edge.pow(2).sum(dim=-1).sqrt().clamp(min=1e-8)
        tgt_len = tgt_edge.pow(2).sum(dim=-1).sqrt().clamp(min=1e-8)
        bone_losses.append((pred_len - tgt_len).pow(2))

        edge_valid = _edge_valid_mask(
            action_dim_is_pad, parent, child, s, e, pred.device, pred_len.dtype
        )
        if edge_valid is None:
            edge_valid = torch.ones_like(pred_len)
        else:
            edge_valid = edge_valid.to(dtype=pred_len.dtype)
        bone_valid.append(edge_valid)

    loss = torch.stack(bone_losses, dim=0)
    edge_valid = torch.stack(bone_valid, dim=0).to(dtype=loss.dtype)
    loss = (loss * edge_valid).sum(dim=0) / edge_valid.sum(dim=0).clamp(min=1.0)
    return _apply_frame_valid(loss, action_is_pad, action_dim_is_pad, s, e)


def bone_direction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    action_is_pad: torch.Tensor | None = None,
    action_dim_is_pad: torch.Tensor | None = None,
    norm_mean: torch.Tensor | None = None,
    norm_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match SMPL-H bone directions (1 - cosine similarity) in physical keypoint space."""
    schema = get_action_schema()
    s, e = schema.slices["keypoints_52"]
    parents = SMPLH_PARENTS
    pred_kp, tgt_kp = _maybe_denorm_keypoints(
        pred, target, norm_mean=norm_mean, norm_std=norm_std
    )

    dir_losses: list[torch.Tensor] = []
    bone_valid: list[torch.Tensor] = []
    for child in range(1, 52):
        parent = int(parents[child])
        pred_edge = pred_kp[..., child, :] - pred_kp[..., parent, :]
        tgt_edge = tgt_kp[..., child, :] - tgt_kp[..., parent, :]
        tgt_len = tgt_edge.pow(2).sum(dim=-1).sqrt()
        pred_len = pred_edge.pow(2).sum(dim=-1).sqrt()
        pred_u = pred_edge / pred_len.unsqueeze(-1).clamp(min=1e-8)
        tgt_u = tgt_edge / tgt_len.unsqueeze(-1).clamp(min=1e-8)
        cos = (pred_u * tgt_u).sum(dim=-1).clamp(-1.0, 1.0)
        dir_losses.append(1.0 - cos)

        edge_valid = _edge_valid_mask(
            action_dim_is_pad, parent, child, s, e, pred.device, pred_edge.dtype
        )
        if edge_valid is None:
            edge_valid = torch.ones_like(dir_losses[-1])
        else:
            edge_valid = edge_valid.to(dtype=dir_losses[-1].dtype)
        edge_valid = edge_valid * (tgt_len > 1e-6).to(dtype=edge_valid.dtype)
        bone_valid.append(edge_valid)

    loss = torch.stack(dir_losses, dim=0)
    edge_valid = torch.stack(bone_valid, dim=0).to(dtype=loss.dtype)
    loss = (loss * edge_valid).sum(dim=0) / edge_valid.sum(dim=0).clamp(min=1.0)
    return _apply_frame_valid(loss, action_is_pad, action_dim_is_pad, s, e)
