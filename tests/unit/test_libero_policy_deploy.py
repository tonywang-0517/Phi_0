"""LIBERO delta deploy: gripper processed once (no double invert)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import torch


def test_libero_absolute_robot7d_uses_absolute_action_path():
    from phi0.benchmark.libero_deploy import LiberoDeployFlags
    from phi0.benchmark.policy import Phi0VLAPolicy

    policy = Phi0VLAPolicy.__new__(Phi0VLAPolicy)
    policy.cfg = MagicMock(invert_openvla_gripper=False)
    policy.action_mode = "robot7d"
    policy._libero_flags = LiberoDeployFlags(
        delta_eef=False, proprio_absolute=True, absolute_eef=True
    )
    policy.default_open_loop = 8
    policy.model = MagicMock()
    policy.model.uses_robot7d_action.return_value = True
    policy.processor = MagicMock()
    policy.processor.denormalize_robot7d_future.return_value = torch.tensor(
        [[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 8,
        dtype=torch.float32,
    )

    absolute_calls = {"n": 0}
    delta_calls = {"n": 0}

    def counting_postprocess(d7, flags, *, invert_openvla_gripper):
        if flags.absolute_eef and not flags.delta_eef:
            absolute_calls["n"] += 1
        elif flags.delta_eef:
            delta_calls["n"] += 1
        return np.asarray(d7, dtype=np.float32)

    with patch(
        "phi0.benchmark.policy.postprocess_libero_robot7d_chunk",
        side_effect=counting_postprocess,
    ), patch.object(policy, "predict_phi0_chunk", return_value=torch.zeros(8, 256)):
        policy.step({}, "task", 0, benchmark="libero")

    assert absolute_calls["n"] == 1
    assert delta_calls["n"] == 0


def test_libero_delta_robot7d_processes_gripper_once():
    from phi0.benchmark.libero_deploy import LiberoDeployFlags
    from phi0.benchmark.policy import Phi0VLAPolicy

    policy = Phi0VLAPolicy.__new__(Phi0VLAPolicy)
    policy.cfg = MagicMock(invert_openvla_gripper=False)
    policy.action_mode = "robot7d"
    policy._libero_flags = LiberoDeployFlags(
        delta_eef=True, proprio_absolute=True, absolute_eef=False
    )
    policy.default_open_loop = 8
    policy.model = MagicMock()
    policy.model.uses_robot7d_action.return_value = True
    policy.processor = MagicMock()
    policy.processor.denormalize_robot7d_future.return_value = torch.tensor(
        [[0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 8,
        dtype=torch.float32,
    )

    call_count = {"n": 0}

    def counting_postprocess(d7, flags, *, invert_openvla_gripper):
        call_count["n"] += 1
        out = np.asarray(d7, dtype=np.float32).copy()
        out[..., 6] = -1.0
        return out

    with patch(
        "phi0.benchmark.policy.postprocess_libero_robot7d_chunk",
        side_effect=counting_postprocess,
    ), patch.object(policy, "predict_phi0_chunk", return_value=torch.zeros(8, 256)):
        actions = policy.step({}, "task", 0, benchmark="libero")

    assert call_count["n"] == 1
    assert len(actions) == 8
    assert actions[0].shape == (7,)
