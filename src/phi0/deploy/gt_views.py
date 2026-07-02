"""GT ego / wrist panels for pick-tissue ZMQ eval videos."""

from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np

from phi0.deploy.pick_tissue_gt import (
    PickTissueEpisodeSpan,
    PickTissueGtReader,
    control_index_to_global_frame,
)
from phi0.deploy.pick_tissue_gt_images import (
    PickTissuePredecodedReader,
    letterbox_rgb,
    reader_from_meta,
)

PANEL_W = 320
PANEL_H = 180


def upsample_len_20_to_50(num_pub: int) -> int:
    if num_pub < 2:
        return num_pub
    return int(round((num_pub - 1) * 2.5) + 1)


def motion_timeline_len(num_ctrl: int, *, control_fps: float, tracker_freq: int) -> int:
    """Length after resampling publisher control_hz trajectory to tracker_hz."""
    if num_ctrl < 2:
        return num_ctrl
    if abs(float(control_fps) - float(tracker_freq)) < 1e-3:
        return num_ctrl
    ratio = float(tracker_freq) / float(control_fps)
    if abs(ratio - 2.5) < 1e-3:
        return upsample_len_20_to_50(num_ctrl)
    return int(round((num_ctrl - 1) * ratio) + 1)


def upsample_len_ctrl_to_tracker(
    num_pub: int, *, control_fps: float = 20.0, tracker_freq: int = 50
) -> int:
    return motion_timeline_len(num_pub, control_fps=control_fps, tracker_freq=tracker_freq)


def track_step_to_pub_idx(
    track_step: int,
    *,
    num_pub: int,
    len_qpos_50: int,
    stand_n: int,
    blend_n: int,
) -> int:
    """Map tracker timeline step → publisher motion frame index."""
    if num_pub <= 1:
        return 0
    if track_step < stand_n:
        return 0
    motion_idx_50 = track_step - stand_n + blend_n
    motion_idx_50 = min(max(motion_idx_50, 0), len_qpos_50 - 1)
    if len_qpos_50 <= 1:
        return 0
    pub = int(round(motion_idx_50 * (num_pub - 1) / (len_qpos_50 - 1)))
    return min(num_pub - 1, max(0, pub))


def load_pub_gt_panels(
    reader: PickTissueGtReader | PickTissuePredecodedReader,
    span: PickTissueEpisodeSpan,
    *,
    num_pub: int,
    proprio_w: int,
    control_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ego + wrist RGB for each publisher frame (num_pub, H, W, 3)."""
    native_fps = float(reader.native_fps)
    ego_rows: list[np.ndarray] = []
    wrist_rows: list[np.ndarray] = []
    for i in range(num_pub):
        global_f = control_index_to_global_frame(
            span.frame_start,
            proprio_w + i,
            native_fps=native_fps,
            control_fps=control_fps,
        )
        ego, wrist = reader.read_ego_wrist_pair(global_f, span)
        ego_rows.append(ego)
        wrist_rows.append(wrist)
    return np.stack(ego_rows, axis=0), np.stack(wrist_rows, axis=0)


def build_track_panel_indices(
    traj_len: int,
    *,
    num_pub: int,
    stand_seconds: float,
    blend_seconds: float,
    freq: int,
    control_fps: float = 20.0,
) -> np.ndarray:
    stand_n = max(0, int(round(float(stand_seconds) * int(freq))))
    blend_n = max(0, int(round(float(blend_seconds) * int(freq))))
    len_tracker = upsample_len_ctrl_to_tracker(num_pub, control_fps=control_fps, tracker_freq=freq)
    return np.array(
        [
            track_step_to_pub_idx(
                t,
                num_pub=num_pub,
                len_qpos_50=len_tracker,
                stand_n=stand_n,
                blend_n=blend_n,
            )
            for t in range(traj_len)
        ],
        dtype=np.int32,
    )


def composite_tracker_gt_views(
    tracker_rgb: np.ndarray,
    ego_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    *,
    panel_w: int = PANEL_W,
    panel_h: int = PANEL_H,
) -> np.ndarray:
    tracker = np.asarray(tracker_rgb)[..., :3]
    th, tw = int(panel_h), int(panel_w)

    def _fit_panel(img: np.ndarray) -> np.ndarray:
        rgb = np.asarray(img)[..., :3]
        if rgb.shape[0] == th and rgb.shape[1] == tw:
            return rgb
        return letterbox_rgb(rgb, (th, tw))

    ego = _fit_panel(ego_rgb)
    wrist = _fit_panel(wrist_rgb)
    bottom = np.concatenate([ego, wrist], axis=1)
    out = np.concatenate([tracker, bottom], axis=0)
    y = tracker.shape[0] + 22
    cv2.putText(out, "ego GT", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        out,
        "wrist GT",
        (panel_w + 8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def composite_tracker_sequence(
    tracker_images: list[np.ndarray],
    ego_by_pub: np.ndarray,
    wrist_by_pub: np.ndarray,
    panel_indices: np.ndarray,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for t, tracker in enumerate(tracker_images):
        idx = int(panel_indices[t])
        out.append(
            composite_tracker_gt_views(
                tracker,
                ego_by_pub[idx],
                wrist_by_pub[idx],
            )
        )
    return out


def meta_supports_gt_views(meta: Mapping[str, Any]) -> bool:
    ds = str(meta.get("dataset", "")).strip().lower()
    if ds != "pick_tissue_unified":
        return False
    return meta.get("episode_idx") is not None and meta.get("proprio_w") is not None


def load_gt_views_from_meta(
    meta: Mapping[str, Any],
    *,
    traj_len: int,
    stand_seconds: float,
    blend_seconds: float,
    freq: int,
    view_fit: str = "letterbox_raw",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reader = reader_from_meta(meta, view_fit=view_fit)  # type: ignore[arg-type]
    span = reader.episode_span(int(meta["episode_idx"]))
    num_pub = int(meta["num_frames"])
    proprio_w = int(meta["proprio_w"])
    control_fps = float(meta.get("control_fps", 50.0))
    ego_by_pub, wrist_by_pub = load_pub_gt_panels(
        reader,
        span,
        num_pub=num_pub,
        proprio_w=proprio_w,
        control_fps=control_fps,
    )
    panel_indices = build_track_panel_indices(
        traj_len,
        num_pub=num_pub,
        stand_seconds=stand_seconds,
        blend_seconds=blend_seconds,
        freq=freq,
        control_fps=control_fps,
    )
    return ego_by_pub, wrist_by_pub, panel_indices
