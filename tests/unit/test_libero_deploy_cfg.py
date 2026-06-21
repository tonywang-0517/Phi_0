"""LIBERO deploy configuration, proprio encoding, and sim action postprocess."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch


def test_resolve_libero_deploy_flags_from_train_cfg_when_unset():
    from phi0.benchmark.libero_deploy import resolve_libero_deploy_flags
    from phi0.benchmark.policy import Phi0VLAPolicyConfig

    train_cfg = SimpleNamespace(
        data={
            "libero_delta_eef": True,
            "libero_proprio_absolute": True,
            "libero_absolute_eef": False,
        }
    )
    policy_cfg = Phi0VLAPolicyConfig(
        checkpoint="ckpt.pt",
        config_dir="/tmp/cfg",
        config_name="train_libero_spatial_vlm_only_35k_ddp4",
    )

    flags = resolve_libero_deploy_flags(policy_cfg, train_cfg)

    assert flags.delta_eef is True
    assert flags.proprio_absolute is True
    assert flags.absolute_eef is False
    assert flags.use_proprio_stats(MagicMock(proprio_mean=torch.zeros(7))) is True


def test_resolve_libero_deploy_flags_explicit_policy_override():
    from phi0.benchmark.libero_deploy import resolve_libero_deploy_flags
    from phi0.benchmark.policy import Phi0VLAPolicyConfig

    train_cfg = SimpleNamespace(
        data={
            "libero_delta_eef": True,
            "libero_proprio_absolute": True,
            "libero_absolute_eef": False,
        }
    )
    policy_cfg = Phi0VLAPolicyConfig(
        checkpoint="ckpt.pt",
        config_dir="/tmp/cfg",
        libero_delta_eef=False,
        libero_absolute_eef=True,
        libero_proprio_absolute=False,
    )

    flags = resolve_libero_deploy_flags(policy_cfg, train_cfg)

    assert flags.delta_eef is False
    assert flags.proprio_absolute is False
    assert flags.absolute_eef is True


def test_postprocess_libero_delta_chunk_inverts_gripper():
    from phi0.benchmark.libero_deploy import LiberoDeployFlags, postprocess_libero_robot7d_chunk

    flags = LiberoDeployFlags(delta_eef=True, proprio_absolute=True, absolute_eef=False)
    d7 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    out = postprocess_libero_robot7d_chunk(d7, flags, invert_openvla_gripper=True)
    assert out.shape == (1, 7)
    assert out[0, 6] in (-1.0, 1.0)


def test_policy_init_applies_train_cfg_libero_flags():
    from phi0.benchmark.policy import Phi0VLAPolicy, Phi0VLAPolicyConfig

    train_cfg = MagicMock()
    train_cfg.data = {
        "libero_delta_eef": True,
        "libero_proprio_absolute": True,
        "libero_absolute_eef": False,
        "seq_len": 9,
        "action_video_freq_ratio": 2,
        "control_fps": 20.0,
    }
    train_cfg.device = "cpu"
    train_cfg.get.side_effect = lambda key, default=None: {
        "smoke_action_only": False,
    }.get(key, default)
    payload = {"cfg": {"data": dict(train_cfg.data)}}

    mock_model = MagicMock()
    mock_model.past_action_window_size = 1
    mock_model.device = torch.device("cpu")
    mock_model.torch_dtype = torch.float32
    mock_model.uses_robot7d_action.return_value = True

    mock_processor = MagicMock()
    mock_processor.proprio_mean = torch.zeros(7)

    with patch("phi0.benchmark.policy.initialize_config_dir"), patch(
        "phi0.benchmark.policy.compose", return_value=train_cfg
    ), patch("phi0.benchmark.policy.torch.load", return_value=payload), patch(
        "phi0.benchmark.policy.resolve_inference_device", return_value="cpu"
    ), patch("phi0.benchmark.policy.activate_cuda_device"), patch(
        "phi0.benchmark.policy.create_phi0", return_value=mock_model
    ), patch("phi0.benchmark.policy.build_processor", return_value=mock_processor), patch(
        "phi0.benchmark.policy.apply_processor_stats_from_checkpoint"
    ), patch("phi0.benchmark.policy.sync_model_action_norm"), patch(
        "phi0.benchmark.policy.merge_saved_cfg", side_effect=lambda a, b: a
    ):
        policy = Phi0VLAPolicy(
            Phi0VLAPolicyConfig(
                checkpoint="ckpt.pt",
                config_dir="/tmp/cfg",
                config_name="train_libero_spatial_vlm_only_35k_ddp4",
                device="cpu",
                min_free_gb=0.0,
            )
        )

    assert policy._libero_delta_eef is True
    assert policy._libero_proprio_absolute is True
    assert policy._libero_absolute_eef is False
    assert policy._libero_flags.delta_eef is True


def test_normalize_proprio_uses_proprio_stats_when_delta_deploy():
    from phi0.benchmark.libero_deploy import LiberoDeployFlags, normalize_libero_proprio_eef_7d

    flags = LiberoDeployFlags(delta_eef=True, proprio_absolute=True, absolute_eef=False)
    processor = MagicMock()
    processor.proprio_mean = torch.ones(7)
    processor.normalize_robot7d_tensor.return_value = torch.zeros(1, 1, 7)
    model = MagicMock()
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.uses_robot7d_action.return_value = True

    eef = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    normalize_libero_proprio_eef_7d(processor, model, eef, flags)

    processor.normalize_robot7d_tensor.assert_called_once()
    assert processor.normalize_robot7d_tensor.call_args.kwargs["proprio"] is True


def test_video_bcthw_to_pixel_batch():
    from phi0.models.vlm.preprocess import video_bcthw_to_pixel_batch

    video = torch.zeros(2, 3, 1, 4, 4) * 2.0 - 1.0
    pixel = video_bcthw_to_pixel_batch(video)
    assert pixel.shape == (2, 1, 3, 4, 4)
    assert torch.allclose(pixel, torch.zeros_like(pixel))


def test_build_deploy_vlm_inputs_uses_processor_transform():
    from phi0.models.vlm.preprocess import build_deploy_vlm_inputs_from_pixels

    phi0_processor = MagicMock()
    phi0_processor.vlm_image_size = (180, 320)
    transform = MagicMock()
    phi0_processor.vlm_image_transform.return_value = transform
    vlm_processor = MagicMock()

    pixel = torch.rand(1, 1, 3, 256, 256)
    with patch(
        "phi0.models.vlm.preprocess.build_vlm_inputs_from_pixel_batch",
        return_value={"input_ids": torch.ones(1, 4)},
    ) as mock_build:
        out = build_deploy_vlm_inputs_from_pixels(
            vlm_processor,
            phi0_processor,
            pixel,
            ["pick up the bowl"],
            model_max_length=512,
        )

    assert "input_ids" in out
    phi0_processor.vlm_image_transform.assert_called_once()
    kwargs = mock_build.call_args.kwargs
    assert kwargs["img_aug"] is False
    assert kwargs["training"] is False
    assert kwargs["transform"] is transform


def test_build_training_vlm_inputs_respects_processor_train_mode():
    from phi0.models.vlm.preprocess import build_training_vlm_inputs_from_pixels

    phi0_processor = MagicMock()
    phi0_processor.vlm_image_size = (180, 320)
    phi0_processor.vlm_img_aug = True
    phi0_processor._is_train = True
    transform = MagicMock()
    phi0_processor.vlm_image_transform.return_value = transform
    vlm_processor = MagicMock()
    pixel = torch.rand(1, 1, 3, 256, 256)

    with patch(
        "phi0.models.vlm.preprocess.build_vlm_inputs_from_pixel_batch",
        return_value={"input_ids": torch.ones(1, 4)},
    ) as mock_build:
        build_training_vlm_inputs_from_pixels(
            vlm_processor, phi0_processor, pixel, ["task"], model_max_length=512
        )

    kwargs = mock_build.call_args.kwargs
    assert kwargs["img_aug"] is True
    assert kwargs["training"] is True
    assert kwargs["transform"] is transform
