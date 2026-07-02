"""Reusable multi-chunk deploy inference helpers shared across ZMQ publishers."""

from __future__ import annotations

import numpy as np
import torch

from phi0.deploy.gt_io import is_pick_tissue_unified_cfg
from phi0.inference.deploy_align import deploy_history_control_indices
from phi0.inference.session import (
    ActionInferenceSession,
    resolve_deploy_action_chunk_size,
)


def _history_window(model) -> int:
    if getattr(model, "uses_history_action_input", lambda: False)():
        return int(getattr(model, "action_history_window", 0) or 0)
    return int(getattr(model, "past_action_window_size", 4) or 4)


@torch.no_grad()
def _predict_motion_deploy(
    model,
    processor,
    inputs: dict,
    *,
    num_frames: int,
    proprio_w: int,
    gt_norm_lut: dict[int, torch.Tensor],
) -> np.ndarray:
    session = ActionInferenceSession(model, processor=processor, use_gt_history=True)
    session.prefill_from_clip_inputs(inputs)
    history_w = _history_window(model)
    chunk_h = resolve_deploy_action_chunk_size(model)
    device = model.device
    if hasattr(gt_norm_lut, "pin_device"):
        gt_norm_lut.pin_device(device)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16
    chunks: list[np.ndarray] = []
    for seg_start in range(0, num_frames, chunk_h):
        deploy_c = proprio_w + seg_start
        if history_w > 0:
            hist_idxs = deploy_history_control_indices(deploy_c, history_w)
            hist = torch.stack([gt_norm_lut[c] for c in hist_idxs], dim=0).to(
                device, non_blocking=True
            )
            session.set_history_gt(hist)
        chunk_len = min(chunk_h, num_frames - seg_start)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = session.predict(chunk_len, denormalize=True)
        chunks.append(pred.float().detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _resolve_eval_dataset(cfg, base):
    """Return (dataset, collate_fn) for inference; pick-tissue uses clip dataset directly."""
    from phi0.data.sequence import SequenceDataset, sequence_dataset_from_cfg

    if is_pick_tissue_unified_cfg(cfg.data):
        from phi0.data.pick_tissue_unified import PickTissueUnifiedClipDataset

        return base, PickTissueUnifiedClipDataset.collate_fn
    seq_ds = sequence_dataset_from_cfg(base, cfg.data)
    return seq_ds, SequenceDataset.collate_fn
