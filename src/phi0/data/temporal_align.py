"""Unified control timeline: resample native datasets + DiT4DiT-style video subsampling."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_DATASET_NATIVE_FPS: Dict[str, float] = {
    "xperience": 20.0,
    "egodex": 30.0,
}


def native_span_frames(seq_len: int, control_fps: float, native_fps: float) -> int:
    """Native frames needed to cover ``seq_len`` control steps at ``control_fps``."""
    if seq_len <= 1:
        return 1
    if control_fps <= 0 or native_fps <= 0:
        raise ValueError(f"fps must be positive, got control={control_fps} native={native_fps}")
    return int(round((seq_len - 1) * native_fps / control_fps)) + 1


def max_native_span_frames(
    seq_len: int,
    control_fps: float,
    native_fps_map: Mapping[str, float] | None = None,
) -> int:
    fps_map = native_fps_map or DEFAULT_DATASET_NATIVE_FPS
    return max(native_span_frames(seq_len, control_fps, fps) for fps in fps_map.values())


def control_to_native_indices(src_len: int, dst_len: int) -> np.ndarray:
    """Map each control step to a native source index (inclusive endpoints)."""
    if dst_len <= 0:
        return np.zeros(0, dtype=np.int64)
    if src_len <= 1 or dst_len == 1:
        return np.zeros(dst_len, dtype=np.int64)
    return np.round(np.linspace(0, src_len - 1, dst_len)).astype(np.int64)


def video_sample_control_indices(seq_len: int, action_video_freq_ratio: int) -> List[int]:
    """DiT4DiT-style pixel subsample on the control timeline (0, ratio, 2*ratio, ...)."""
    ratio = max(1, int(action_video_freq_ratio))
    return list(range(0, int(seq_len), ratio))


def v2w_cond_control_span_steps(
    cond_pixel_frames: int = 5,
    action_video_freq_ratio: int = 2,
) -> int:
    """Control-step span covered by ``cond_pixel_frames`` subsampled frames (inclusive endpoints)."""
    ratio = max(1, int(action_video_freq_ratio))
    cond_px = max(1, int(cond_pixel_frames))
    return (cond_px - 1) * ratio


def deploy_v2w_cond_video_control_indices(
    current_step: int,
    *,
    cond_pixel_frames: int = 5,
    action_video_freq_ratio: int = 2,
) -> List[int]:
    """Official V2W cond window: ``cond_pixel_frames`` subsampled indices ending at ``current_step``."""
    ratio = max(1, int(action_video_freq_ratio))
    cond_px = max(1, int(cond_pixel_frames))
    span = v2w_cond_control_span_steps(cond_px, ratio)
    clip_start = max(0, int(current_step) - span)
    rel_len = int(current_step) - clip_start + 1
    rel = video_sample_control_indices(rel_len, ratio)
    # Always include the live frame at ``current_step`` (may fall off the ratio grid).
    last_rel = rel_len - 1
    if not rel or rel[-1] != last_rel:
        rel = sorted(set(rel + [last_rel]))
    if len(rel) > cond_px:
        rel = rel[-cond_px:]
    if len(rel) < cond_px:
        rel = [rel[0]] * (cond_px - len(rel)) + rel
    return [clip_start + int(i) for i in rel]


def proprio_current_control_step(past_action_window_size: int) -> int:
    """Inclusive control index of the proprio-prefix 'current' frame within a training clip."""
    w = int(past_action_window_size)
    if w <= 0:
        raise ValueError(f"past_action_window_size must be positive, got {w}")
    return w - 1


def observation_subsampled_frame_index(
    past_action_window_size: int,
    subsampled_control_indices: Sequence[int],
) -> int:
    """Index into subsampled pixel ``[B,T,C,H,W]`` aligned with proprio current step."""
    if len(subsampled_control_indices) == 0:
        raise ValueError("subsampled_control_indices must be non-empty")
    anchor = proprio_current_control_step(past_action_window_size)
    return int(
        subsampled_positions_for_control_indices(
            [anchor], subsampled_control_indices
        )[0]
    )


def training_v2w_cond_control_indices(
    *,
    past_action_window_size: int,
    action_video_freq_ratio: int = 2,
    cond_pixel_frames: int = 5,
) -> List[int]:
    """V2W cond indices ending at proprio current — same rule as deploy ``past_only``."""
    current = proprio_current_control_step(past_action_window_size)
    return deploy_v2w_cond_video_control_indices(
        current,
        action_video_freq_ratio=action_video_freq_ratio,
        cond_pixel_frames=cond_pixel_frames,
    )


def subsampled_positions_for_control_indices(
    control_indices: Sequence[int],
    subsampled_control_indices: Sequence[int],
) -> List[int]:
    """Map control timeline indices to positions along a subsampled video tensor (T axis)."""
    pos = {int(c): i for i, c in enumerate(subsampled_control_indices)}
    subs = [int(c) for c in subsampled_control_indices]
    if not subs:
        raise ValueError("subsampled_control_indices must be non-empty")
    out: List[int] = []
    for c in control_indices:
        c = int(c)
        if c in pos:
            out.append(pos[c])
            continue
        prior = [x for x in subs if x <= c]
        key = max(prior) if prior else subs[0]
        out.append(pos[key])
    return out


def gather_subsampled_video_bcthw(
    video_bcthw: torch.Tensor,
    tensor_indices: Sequence[int],
) -> torch.Tensor:
    """Gather ``[B,3,T,H,W]`` along T using subsampled positions."""
    if video_bcthw.ndim != 5:
        raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video_bcthw.shape)}")
    idx = torch.tensor(list(tensor_indices), device=video_bcthw.device, dtype=torch.long)
    return video_bcthw.index_select(2, idx).contiguous()


def gather_subsampled_pad_mask(
    pad_bt: torch.Tensor,
    tensor_indices: Sequence[int],
) -> torch.Tensor:
    """Gather per-frame pad flags ``[B,T]`` along T."""
    if pad_bt.ndim != 2:
        raise ValueError(f"pad must be [B,T], got {tuple(pad_bt.shape)}")
    idx = torch.tensor(list(tensor_indices), device=pad_bt.device, dtype=torch.long)
    return pad_bt.index_select(1, idx).contiguous()


def select_proprio_aligned_tower_video(
    video_bcthw: torch.Tensor,
    *,
    seq_len: int,
    past_action_window_size: int,
    action_video_freq_ratio: int = 2,
    cond_pixel_frames: int = 5,
) -> tuple[torch.Tensor, List[int]]:
    """Proprio-aligned V2W window from a full subsampled training clip (matches deploy)."""
    subsampled = video_sample_control_indices(int(seq_len), int(action_video_freq_ratio))
    cond_ctrl = training_v2w_cond_control_indices(
        past_action_window_size=past_action_window_size,
        action_video_freq_ratio=action_video_freq_ratio,
        cond_pixel_frames=cond_pixel_frames,
    )
    t_idx = subsampled_positions_for_control_indices(cond_ctrl, subsampled)
    return gather_subsampled_video_bcthw(video_bcthw, t_idx), cond_ctrl


def video_history_control_span_steps(control_fps: float, video_history_seconds: float) -> int:
    """Control-step delta covering ``video_history_seconds`` (e.g. 20 Hz × 1 s → 20)."""
    if control_fps <= 0 or video_history_seconds <= 0:
        raise ValueError(
            f"control_fps and video_history_seconds must be positive, "
            f"got control_fps={control_fps} video_history_seconds={video_history_seconds}"
        )
    return max(1, int(round(float(control_fps) * float(video_history_seconds))))


def min_seq_len_for_video_history(
    control_fps: float,
    video_history_seconds: float,
    *,
    min_action_steps: int = 1,
) -> int:
    """Minimum ``seq_len`` so a training clip spans at least ``video_history_seconds``."""
    span = video_history_control_span_steps(control_fps, video_history_seconds)
    return max(int(min_action_steps), span + 1)


def dit4dit_train_num_frames_out(seq_len: int, action_video_freq_ratio: int) -> int:
    """Training / eval pixel frame count (DiT4DiT ``train_num_frames_out`` formula)."""
    return len(video_sample_control_indices(seq_len, action_video_freq_ratio))


def video_cond_pixel_frames_for_training(num_latent_conditional_frames: int = 2) -> int:
    """Cosmos Video2World: pixel cond length from ``num_latent_conditional_frames`` (1 or 2)."""
    return video2world_cond_pixel_frames(num_latent_conditional_frames)


def video2world_cond_pixel_frames(num_latent_conditional_frames: int) -> int:
    """Official Cosmos: ``4 * (num_latent_conditional_frames - 1) + 1`` pixel frames."""
    n = int(num_latent_conditional_frames)
    if n not in (1, 2):
        raise ValueError(f"num_latent_conditional_frames must be 1 or 2, got {n}")
    return 4 * (n - 1) + 1


def video2world_gt_reference_uint8(
    gt_prefix_thw3: np.ndarray,
    cond_pixel_frames: int,
    *,
    gt_future_thw3: np.ndarray | None = None,
) -> np.ndarray:
    """GT timeline aligned to Video2World output.

  ``pred[i]`` for ``i < cond`` matches ``gt_prefix[-cond+i]``.
  ``pred[i]`` for ``i >= cond`` matches ``gt_future[i-cond]`` when provided.
    """
    if gt_prefix_thw3.ndim != 4 or gt_prefix_thw3.shape[-1] != 3:
        raise ValueError(f"gt_prefix must be [T,H,W,3], got {gt_prefix_thw3.shape}")
    t_out = int(gt_prefix_thw3.shape[0])
    cond_px = int(cond_pixel_frames)
    if cond_px < 1 or cond_px > t_out:
        raise ValueError(f"cond_pixel_frames={cond_px} invalid for T={t_out}")
    n_future = t_out - cond_px
    out = np.empty_like(gt_prefix_thw3)
    out[:cond_px] = gt_prefix_thw3[-cond_px:]
    if n_future <= 0:
        return out
    if gt_future_thw3 is not None:
        fut = np.asarray(gt_future_thw3, dtype=np.uint8)
        if fut.shape[0] < n_future:
            raise ValueError(
                f"gt_future has {fut.shape[0]} frames but need {n_future} continuation frames"
            )
        out[cond_px:] = fut[:n_future]
    else:
        out[cond_px:] = gt_prefix_thw3[-1]
    return out


def video2world_future_gt_span(
    *,
    num_frames_out: int,
    cond_pixel_frames: int,
    action_video_freq_ratio: int,
) -> tuple[int, int]:
    """Return ``(future_video_frames, extra_control_steps)`` for extended GT loading."""
    future_px = max(0, int(num_frames_out) - int(cond_pixel_frames))
    extra_ctrl = future_px * max(1, int(action_video_freq_ratio))
    return future_px, extra_ctrl


def video2world_mae_metrics(
    pred_thw3: np.ndarray,
    gt_prefix_thw3: np.ndarray,
    cond_pixel_frames: int,
    *,
    gt_future_thw3: np.ndarray | None = None,
) -> dict[str, float]:
    """MAE for Video2World against aligned GT (cond tail + optional real future)."""
    gt_ref = video2world_gt_reference_uint8(
        gt_prefix_thw3,
        cond_pixel_frames,
        gt_future_thw3=gt_future_thw3,
    )
    pred = pred_thw3.astype(np.float32)
    gt = gt_ref.astype(np.float32)
    scale = 255.0
    cond_px = int(cond_pixel_frames)
    cond_mae = float(np.abs(pred[:cond_px] - gt[:cond_px]).mean() / scale)
    full_mae = float(np.abs(pred - gt).mean() / scale)
    future_mae = float(np.abs(pred[cond_px:] - gt[cond_px:]).mean() / scale) if cond_px < pred.shape[0] else 0.0
    chrono_full = float(np.abs(pred - gt_prefix_thw3.astype(np.float32)).mean() / scale)
    return {
        "cond_mae": cond_mae,
        "aligned_full_mae": full_mae,
        "aligned_future_mae": future_mae,
        "chrono_full_mae": chrono_full,
    }


def build_video2world_prepare_clip(
    video_bcthw: torch.Tensor,
    *,
    num_frames_out: int,
    num_latent_conditional_frames: int = 2,
) -> tuple[torch.Tensor, int]:
    """Official Video2World prep: tail cond pixels, pad to ``num_frames_out`` with last frame."""
    if video_bcthw.ndim != 5 or video_bcthw.shape[1] != 3:
        raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video_bcthw.shape)}")
    num_frames_in = video2world_cond_pixel_frames(num_latent_conditional_frames)
    t_in = int(video_bcthw.shape[2])
    if t_in < num_frames_in:
        raise ValueError(
            f"video T={t_in} shorter than Video2World cond={num_frames_in} "
            f"(num_latent_conditional_frames={num_latent_conditional_frames})"
        )
    tail = video_bcthw[:, :, -num_frames_in:, :, :]
    out_t = int(num_frames_out)
    if tail.shape[2] >= out_t:
        return tail[:, :, :out_t].contiguous(), num_frames_in
    n_pad = out_t - int(tail.shape[2])
    last = tail[:, :, -1:, :, :]
    pad = last.repeat(1, 1, n_pad, 1, 1)
    return torch.cat([tail, pad], dim=2).contiguous(), num_frames_in


def split_video_cond_future(
    video_bcthw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """DiT4DiT ``build_cosmos_inputs``: frame 0 -> cond, frames 1..T-1 -> future supervision."""
    if video_bcthw.ndim != 5 or video_bcthw.shape[1] != 3:
        raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video_bcthw.shape)}")
    t = int(video_bcthw.shape[2])
    if t < 1:
        raise ValueError("video must have at least one frame")
    cond = video_bcthw[:, :, :1]
    future = video_bcthw[:, :, 1:] if t > 1 else None
    return cond, future


def resample_action_sequence(action: torch.Tensor, src_len: int, dst_len: int) -> torch.Tensor:
    """Linear resample ``[src_len, D]`` -> ``[dst_len, D]``."""
    if src_len == dst_len:
        return action
    if src_len <= 0 or dst_len <= 0:
        raise ValueError(f"invalid resample lengths src={src_len} dst={dst_len}")
    x = action.reshape(src_len, -1).unsqueeze(0).permute(0, 2, 1).float()
    y = F.interpolate(x, size=dst_len, mode="linear", align_corners=True)
    return y.squeeze(0).permute(1, 0).to(dtype=action.dtype, device=action.device)


def resample_bool_sequence(flags: torch.Tensor, src_len: int, dst_len: int) -> torch.Tensor:
    """Nearest-neighbor resample for pad / validity flags ``[src_len, ...]``."""
    if src_len == dst_len:
        return flags
    idx = control_to_native_indices(src_len, dst_len)
    if flags.ndim == 1:
        return flags[idx]
    return flags[idx]


def resample_image_sequence(images: torch.Tensor, src_len: int, dst_len: int) -> torch.Tensor:
    """Temporal linear resample ``[src_len, C, H, W]`` -> ``[dst_len, C, H, W]``."""
    if src_len == dst_len:
        return images
    x = images.reshape(src_len, -1).unsqueeze(0).permute(0, 2, 1).float()
    y = F.interpolate(x, size=dst_len, mode="linear", align_corners=True)
    c, h, w = images.shape[1:]
    return y.squeeze(0).permute(1, 0).reshape(dst_len, c, h, w).to(
        dtype=images.dtype, device=images.device
    )


def resolve_native_fps(dataset_name: str, native_fps_map: Mapping[str, float]) -> float:
    if dataset_name not in native_fps_map:
        raise KeyError(f"Unknown dataset {dataset_name!r}; set data.dataset_native_fps")
    return float(native_fps_map[dataset_name])
