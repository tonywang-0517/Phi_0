"""Sequential MP4 preload for training (avoids per-frame cv2 open/seek)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def preload_mp4_frames(
    video_path: str | Path | None,
    image_size: Tuple[int, int],
    *,
    max_frames: Optional[int] = None,
) -> Optional[List[torch.Tensor]]:
    """Read MP4 once into a list of [C,H,W] float tensors in [0,1]."""
    if video_path is None:
        return None
    path = Path(video_path)
    if not path.is_file():
        return None

    import cv2

    h, w = image_size
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.warning("VideoFrameCache: failed to open %s", path)
        return None

    frames: List[torch.Tensor] = []
    try:
        while True:
            if max_frames is not None and len(frames) >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            frames.append(torch.from_numpy(frame).permute(2, 0, 1).contiguous().float() / 255.0)
    finally:
        cap.release()

    if not frames:
        logger.warning("VideoFrameCache: no frames read from %s", path)
        return None

    logger.info("VideoFrameCache: preloaded %d frames from %s", len(frames), path.name)
    return frames
