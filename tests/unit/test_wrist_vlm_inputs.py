"""Unit tests for wrist camera -> QwenVL multi-image path."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from phi0.data.processor import Phi0Processor


def test_processor_stacks_dual_cameras_when_use_wrist_view():
    processor = Phi0Processor(use_wrist_view=True)
    batch = {
        "task": ["pick up the bowl"],
        "action": torch.zeros(1, 1, 256),
        "image_is_pad": torch.zeros(1, 1, dtype=torch.bool),
        "action_is_pad": torch.zeros(1, 9, dtype=torch.bool),
        "action_dim_is_pad": torch.zeros(1, 256, dtype=torch.bool),
        "idx": torch.tensor([0]),
        "images": {
            "ego_view": torch.rand(1, 1, 3, 64, 64),
            "wrist_view": torch.rand(1, 1, 3, 64, 64),
        },
    }
    sample = processor.preprocess(batch)
    assert sample["pixel_values"].shape == (1, 2, 1, 3, 64, 64)


def test_build_vlm_inputs_from_pixel_batch_dual_views():
    from phi0.models.vlm.preprocess import build_vlm_inputs_from_pixel_batch

    processor = MagicMock()
    transform = MagicMock(side_effect=lambda pil: pil)
    pixel = torch.rand(2, 1, 3, 32, 32)
    wrist = torch.rand(2, 1, 3, 32, 32)

    with patch(
        "phi0.models.vlm.preprocess.preprocess_frame_for_vlm",
        side_effect=lambda frame, _transform: frame,
    ) as mock_prep, patch(
        "phi0.models.vlm.preprocess.build_vlm_chat_inputs",
        return_value={"input_ids": torch.ones(2, 8)},
    ) as mock_chat:
        out = build_vlm_inputs_from_pixel_batch(
            processor,
            pixel,
            ["task a", "task b"],
            vlm_image_size=(180, 320),
            transform=transform,
            wrist_pixel=wrist,
        )

    assert "input_ids" in out
    images_arg = mock_chat.call_args.args[1]
    assert len(images_arg) == 2
    assert len(images_arg[0]) == 2
    assert len(images_arg[1]) == 2
    assert mock_prep.call_count == 4


def test_collate_vlm_inputs_dual_image_batch2():
    from phi0.models.vlm.preprocess import collate_vlm_inputs

    items = [
        {
            "input_ids": torch.arange(10).unsqueeze(0),
            "pixel_values": torch.randn(616, 1536),
            "image_grid_thw": torch.tensor([[1, 14, 22], [1, 14, 22]]),
            "mm_token_type_ids": torch.zeros(1, 10, dtype=torch.long),
        },
        {
            "input_ids": torch.arange(10, 20).unsqueeze(0),
            "pixel_values": torch.randn(616, 1536),
            "image_grid_thw": torch.tensor([[1, 14, 22], [1, 14, 22]]),
            "mm_token_type_ids": torch.ones(1, 10, dtype=torch.long),
        },
    ]
    out = collate_vlm_inputs(items, pad_token_id=0)
    assert out["input_ids"].shape[0] == 2
    assert out["image_grid_thw"].shape == (4, 3)
    assert out["pixel_values"].shape == (2, 616, 1536)
    assert out["mm_token_type_ids"].shape == (2, 10)


def test_mono_camera_dataset_skips_wrist_view():
    from phi0.data.libero_rlds import LiberoRldsFrameDataset

    ds = LiberoRldsFrameDataset(
        suite="libero_spatial",
        max_episodes=1,
        max_shards=1,
        mono_camera=True,
        cache_native_frames=False,
    )
    sample = ds[0]
    assert "wrist_view" not in sample["images"]
