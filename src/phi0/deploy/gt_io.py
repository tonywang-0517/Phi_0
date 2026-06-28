"""Unified GT I/O for Phi-0 → ZMQ publisher (Xperience HDF5 or pick-tissue LeRobot)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import h5py
import numpy as np
import torch

from phi0.deploy.pick_tissue_gt import (
    PickTissueEpisodeSpan,
    PickTissueGtReader,
    control_index_to_global_frame,
    episode_span_for_clip_item,
    reader_from_data_cfg,
)
from phi0.deploy.gmr_retarget import (
    translate_human_data_sequence,
    unified_chunk_to_gmr_human_data_list,
)
from phi0.data.processor import Phi0Processor
from phi0.data.xperience_unified_gt import read_xperience_root_trans_world
from phi0.schema.unified_action_schema import (
    pack_from_xperience_hdf5_frame,
    unpack_root_quat_wxyz,
)


def is_pick_tissue_unified_cfg(data_cfg: Mapping[str, Any]) -> bool:
    return str(data_cfg.get("dataset", "")).strip().lower() == "pick_tissue_unified"


def control_index_to_native(
    native_start: int,
    control_idx: int,
    *,
    native_fps: float,
    control_fps: float,
) -> int:
    return int(native_start) + int(
        round(float(control_idx) * float(native_fps) / float(control_fps))
    )


def _normalize_d_raw(processor: Phi0Processor, d_raw: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(d_raw, dtype=np.float32)).float().unsqueeze(0)
    return processor._normalize_action(t)


def deploy_gt_norm_lut_indices(
    *,
    num_frames: int,
    proprio_w: int,
    chunk_h: int,
    history_w: int,
) -> list[int]:
    """Control indices read by multi-chunk deploy GT history (not full 0..num_frames)."""
    from phi0.inference.deploy_align import deploy_history_control_indices

    need: set[int] = set()
    if int(history_w) <= 0:
        return []
    for seg_start in range(0, int(num_frames), int(chunk_h)):
        deploy_c = int(proprio_w) + int(seg_start)
        need.update(deploy_history_control_indices(deploy_c, int(history_w)))
    return sorted(need)


class LazyDeployGtNormLut:
    """On-demand GT proprio LUT for deploy (avoids O(num_frames) parquet reads)."""

    def __init__(
        self,
        backend: DeployGtBackend,
        processor: Phi0Processor,
        control_indices: list[int],
    ) -> None:
        self._backend = backend
        self._processor = processor
        self._cache: dict[int, torch.Tensor] = {}
        self._expected = {int(c) for c in control_indices}
        self._pin_device: torch.device | None = None

    def pin_device(self, device: torch.device | str) -> None:
        """Move cached proprio rows to GPU (inference hot path)."""
        dev = torch.device(device)
        self._pin_device = dev
        for c in list(self._cache):
            self._cache[c] = self._cache[c].to(dev, non_blocking=True)

    def __getitem__(self, control_idx: int) -> torch.Tensor:
        c = int(control_idx)
        if c not in self._cache:
            d_raw, _ = self._backend.pack_deploy_frame(
                control_idx=c,
                state_control_idx=c,
            )
            t = _normalize_d_raw(self._processor, d_raw).reshape(-1)
            if self._pin_device is not None:
                t = t.to(self._pin_device, non_blocking=True)
            self._cache[c] = t
        return self._cache[c]

    def __len__(self) -> int:
        return len(self._expected)

    def loaded_count(self) -> int:
        return len(self._cache)


def build_lazy_deploy_gt_norm_lut(
    backend: DeployGtBackend,
    processor: Phi0Processor,
    *,
    num_frames: int,
    proprio_w: int,
    chunk_h: int,
    history_w: int,
) -> LazyDeployGtNormLut:
    indices = deploy_gt_norm_lut_indices(
        num_frames=num_frames,
        proprio_w=proprio_w,
        chunk_h=chunk_h,
        history_w=history_w,
    )
    return LazyDeployGtNormLut(backend, processor, indices)


@dataclass(frozen=True)
class EvalClipContext:
    is_pick_tissue: bool
    native_start: int
    native_fps: float
    control_fps: float
    hdf5_path: str | None = None
    pick_span: PickTissueEpisodeSpan | None = None
    pick_reader: PickTissueGtReader | None = None


class DeployGtBackend(Protocol):
    def pack_deploy_frame(
        self, *, control_idx: int, state_control_idx: int
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def load_gt_norm_lut(
        self, processor: Phi0Processor, *, max_control: int
    ) -> dict[int, torch.Tensor]: ...

    def load_gt_unified_sequence(
        self,
        *,
        num_frames: int,
        proprio_w: int,
        chunk_h: int,
    ) -> np.ndarray: ...


@dataclass
class XperienceHdf5GtBackend:
    hdf5_path: str
    native_start: int
    native_fps: float
    control_fps: float

    def _n_max(self) -> int:
        with h5py.File(self.hdf5_path, "r") as f:
            return int(f["full_body_mocap/body_quats"].shape[0])

    def pack_deploy_frame(
        self, *, control_idx: int, state_control_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        native_t = control_index_to_native(
            self.native_start,
            control_idx,
            native_fps=self.native_fps,
            control_fps=self.control_fps,
        )
        native_state = control_index_to_native(
            self.native_start,
            state_control_idx,
            native_fps=self.native_fps,
            control_fps=self.control_fps,
        )
        n_max = self._n_max()
        native_t = min(native_t, n_max - 1)
        native_state = min(native_state, n_max - 1)
        with h5py.File(self.hdf5_path, "r") as f:
            d_raw = pack_from_xperience_hdf5_frame(f, native_t, state_t=native_state).astype(
                np.float32
            )
            anchor = read_xperience_root_trans_world(f, native_state)
        return d_raw, anchor

    def load_gt_norm_lut(
        self, processor: Phi0Processor, *, max_control: int
    ) -> dict[int, torch.Tensor]:
        lut: dict[int, torch.Tensor] = {}
        with h5py.File(self.hdf5_path, "r") as f:
            n_max = int(f["full_body_mocap/body_quats"].shape[0])
            for control_idx in range(max_control + 1):
                native_t = control_index_to_native(
                    self.native_start,
                    control_idx,
                    native_fps=self.native_fps,
                    control_fps=self.control_fps,
                )
                native_t = min(native_t, n_max - 1)
                d_raw = pack_from_xperience_hdf5_frame(f, native_t, state_t=native_t)
                lut[control_idx] = _normalize_d_raw(processor, d_raw).reshape(-1)
        return lut

    def load_gt_unified_sequence(
        self,
        *,
        num_frames: int,
        proprio_w: int,
        chunk_h: int,
    ) -> np.ndarray:
        actions: list[np.ndarray] = []
        with h5py.File(self.hdf5_path, "r") as f:
            n_max = int(f["full_body_mocap/body_quats"].shape[0])
            for i in range(num_frames):
                seg_start = (i // chunk_h) * chunk_h
                control_i = proprio_w + i
                control_state = proprio_w + seg_start
                native_t = control_index_to_native(
                    self.native_start,
                    control_i,
                    native_fps=self.native_fps,
                    control_fps=self.control_fps,
                )
                native_state = control_index_to_native(
                    self.native_start,
                    control_state,
                    native_fps=self.native_fps,
                    control_fps=self.control_fps,
                )
                native_t = min(native_t, n_max - 1)
                native_state = min(native_state, n_max - 1)
                actions.append(
                    pack_from_xperience_hdf5_frame(f, native_t, state_t=native_state).astype(
                        np.float32
                    )
                )
        return np.stack(actions, axis=0)


@dataclass
class PickTissueGtBackend:
    reader: PickTissueGtReader
    span: PickTissueEpisodeSpan
    native_fps: float
    control_fps: float

    def pack_deploy_frame(
        self, *, control_idx: int, state_control_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.reader.pack_deploy_frame(
            self.span,
            control_idx=control_idx,
            state_control_idx=state_control_idx,
            native_fps=self.native_fps,
            control_fps=self.control_fps,
        )

    def load_gt_norm_lut(
        self, processor: Phi0Processor, *, max_control: int
    ) -> dict[int, torch.Tensor]:
        lut: dict[int, torch.Tensor] = {}
        for control_idx in range(max_control + 1):
            d_raw, _ = self.pack_deploy_frame(
                control_idx=control_idx, state_control_idx=control_idx
            )
            lut[control_idx] = _normalize_d_raw(processor, d_raw).reshape(-1)
        return lut

    def load_gt_unified_sequence(
        self,
        *,
        num_frames: int,
        proprio_w: int,
        chunk_h: int,
    ) -> np.ndarray:
        actions: list[np.ndarray] = []
        for i in range(num_frames):
            seg_start = (i // chunk_h) * chunk_h
            d_raw, _ = self.pack_deploy_frame(
                control_idx=proprio_w + i,
                state_control_idx=proprio_w + seg_start,
            )
            actions.append(d_raw)
        return np.stack(actions, axis=0)


def build_eval_clip_context(
    data_cfg: Mapping[str, Any],
    clip_item: Mapping[str, Any],
    *,
    hdf5_path: str,
    native_start: int | None = None,
    native_fps: float,
    control_fps: float,
) -> EvalClipContext:
    if is_pick_tissue_unified_cfg(data_cfg):
        reader = reader_from_data_cfg(data_cfg)
        span = episode_span_for_clip_item(reader, clip_item)
        return EvalClipContext(
            is_pick_tissue=True,
            native_start=int(clip_item["idx"]),
            native_fps=float(reader.native_fps),
            control_fps=float(control_fps),
            pick_span=span,
            pick_reader=reader,
        )
    return EvalClipContext(
        is_pick_tissue=False,
        native_start=int(native_start if native_start is not None else clip_item.get("idx", 0)),
        native_fps=float(native_fps),
        control_fps=float(control_fps),
        hdf5_path=str(hdf5_path),
    )


def build_gt_backend(ctx: EvalClipContext) -> DeployGtBackend:
    if ctx.is_pick_tissue:
        if ctx.pick_reader is None or ctx.pick_span is None:
            raise ValueError("pick-tissue context missing reader/span")
        return PickTissueGtBackend(
            reader=ctx.pick_reader,
            span=ctx.pick_span,
            native_fps=ctx.native_fps,
            control_fps=ctx.control_fps,
        )
    if not ctx.hdf5_path:
        raise ValueError("xperience context missing hdf5_path")
    return XperienceHdf5GtBackend(
        hdf5_path=ctx.hdf5_path,
        native_start=ctx.native_start,
        native_fps=ctx.native_fps,
        control_fps=ctx.control_fps,
    )


def denorm_to_human_frames(
    action_denorm: np.ndarray,
    backend: DeployGtBackend,
    *,
    proprio_w: int,
    chunk_h: int,
    constants: dict[str, np.ndarray],
    motion_deploy: bool,
) -> tuple[list[dict], np.ndarray]:
    all_frames: list[dict] = []
    root_quats: list[np.ndarray] = []
    for seg_start in range(0, action_denorm.shape[0], chunk_h):
        seg_len = min(chunk_h, action_denorm.shape[0] - seg_start)
        chunk = action_denorm[seg_start : seg_start + seg_len]
        control_state = (proprio_w + seg_start) if motion_deploy else 0
        d_raw, anchor_root = backend.pack_deploy_frame(
            control_idx=control_state,
            state_control_idx=control_state,
        )
        del d_raw  # anchor only; chunk carries per-frame pose
        hf = unified_chunk_to_gmr_human_data_list(
            chunk,
            state_root_trans_world=anchor_root,
            betas=None,
            constants=constants,
        )
        all_frames.extend(hf)
        for i in range(seg_len):
            root_quats.append(unpack_root_quat_wxyz(chunk[i]).astype(np.float32))
    return translate_human_data_sequence(all_frames), np.stack(root_quats, axis=0)


def pick_tissue_clip_action_matches_deploy(
    reader: PickTissueGtReader,
    span: PickTissueEpisodeSpan,
    clip_actions: np.ndarray,
    *,
    control_fps: float,
    anchor_control: int = 0,
    atol: float = 1e-4,
) -> None:
    """Assert training clip ``action`` rows match deploy repack at the same control indices."""
    actions = np.asarray(clip_actions, dtype=np.float32)
    native_fps = float(reader.native_fps)
    for i in range(actions.shape[0]):
        expected = reader.read_repacked_action(
            span,
            control_idx=i,
            anchor_control=anchor_control,
            native_fps=native_fps,
            control_fps=control_fps,
        )
        if not np.allclose(actions[i], expected, atol=atol, rtol=0.0):
            err = float(np.max(np.abs(actions[i] - expected)))
            raise AssertionError(
                f"clip action row {i} differs from deploy repack (max abs err={err:.6g})"
            )
