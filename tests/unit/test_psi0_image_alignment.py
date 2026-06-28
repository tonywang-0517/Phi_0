"""Psi0 vs Phi-0 vision pipeline alignment checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from phi0.models.vlm.preprocess import make_psi0_vlm_image_transform, preprocess_frame_for_vlm


def test_psi0_transform_matches_torchvision_compose():
    vlm_size = (180, 320)
    transform = make_psi0_vlm_image_transform(vlm_size, img_aug=False, training=False)
    native = np.zeros((480, 640, 3), dtype=np.uint8)
    native[400:, :, 1] = 200  # bottom band
    out = np.asarray(transform(Image.fromarray(native)))
    assert out.shape == (180, 320, 3)
    assert out[170, 160, 1] == 200


def test_preprocess_always_applies_transform():
    vlm_size = (180, 320)
    transform = make_psi0_vlm_image_transform(vlm_size, img_aug=False, training=False)
    frame = __import__("torch").rand(3, 180, 320)
    pil = preprocess_frame_for_vlm(frame, transform, vlm_image_size=vlm_size)
    assert pil.size == (320, 180)


@pytest.mark.skipif(
    not Path(
        "/mnt/data2/wpy/workspace/Isaac-GR00T/data/pick_tissue_xperience_unified"
    ).is_dir(),
    reason="pick-tissue dataset missing",
)
def test_native_predecode_matches_psi0_transform_on_ep0():
    import cv2
    from phi0.data.predecoded_video import decode_mp4_to_uint8_thwc
    from phi0.data.psi0_image import read_lerobot_video_hw

    root = Path("/mnt/data2/wpy/workspace/Isaac-GR00T/data/pick_tissue_xperience_unified")
    key = "observation.images.ego_view"
    mp4 = root / "videos/chunk-000" / key / "episode_000000.mp4"
    native_hw = read_lerobot_video_hw(root, key)
    native = decode_mp4_to_uint8_thwc(mp4, None, fps=50.0, expected_length=None, backend="cv2")
    assert native.shape[1:3] == native_hw

    transform = make_psi0_vlm_image_transform((180, 320), img_aug=False, training=False)
    psi0_out = np.asarray(transform(Image.fromarray(native[0])))

    # Old bilinear predecode path (legacy) differs from Psi0 NEAREST on native.
    bilinear = cv2.resize(native[0], (320, 180), interpolation=cv2.INTER_LINEAR)
    assert float(np.abs(psi0_out.astype(float) - bilinear.astype(float)).mean()) > 0.4
