"""CALVIN play-table env wrapper (vendored from VLA-Adapter)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Tuple, Union

import gym
import numpy as np
import torch

from calvin_env.envs.play_table_env import get_env
from calvin_env.utils.utils import EglDeviceNotFoundError, get_egl_device_id

logger = logging.getLogger(__name__)


class CalvinEnvWrapperRaw(gym.Wrapper):
    def __init__(self, abs_datasets_dir, observation_space, device, show_gui=False, **kwargs):
        env = get_env(abs_datasets_dir, show_gui=show_gui, obs_space=observation_space, **kwargs)
        super().__init__(env)
        self.observation_space_keys = observation_space
        self.device = device
        self.relative_actions = "rel_actions" in self.observation_space_keys["actions"]
        logger.info("Initialized PlayTableEnv for device %s", self.device)

    @staticmethod
    def set_egl_device(device):
        if "EGL_VISIBLE_DEVICES" in os.environ:
            logger.warning("Environment variable EGL_VISIBLE_DEVICES is already set.")
        cuda_id = torch.cuda.current_device()
        try:
            egl_id = get_egl_device_id(cuda_id)
        except EglDeviceNotFoundError:
            logger.warning("EGL device not found; falling back to EGL_VISIBLE_DEVICES=0")
            egl_id = 0
        os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
        logger.info("EGL_DEVICE_ID %s <==> CUDA_DEVICE_ID %s", egl_id, cuda_id)

    def step(self, action_tensor: torch.Tensor):
        if self.relative_actions:
            action = action_tensor
            assert len(action) == 7
        else:
            if action_tensor.shape[-1] == 7:
                slice_ids = [3, 6]
            elif action_tensor.shape[-1] == 8:
                slice_ids = [3, 7]
            else:
                raise NotImplementedError("actions must have length 7 or 8")
            action = np.split(action_tensor, slice_ids)
        o, r, d, i = self.env.step(action)
        return o, r, d, i

    def reset(
        self,
        reset_info: Dict[str, Any] = None,
        batch_idx: int = 0,
        seq_idx: int = 0,
        scene_obs: Any = None,
        robot_obs: Any = None,
    ):
        if reset_info is not None:
            obs = self.env.reset(
                robot_obs=reset_info["robot_obs"][batch_idx, seq_idx],
                scene_obs=reset_info["scene_obs"][batch_idx, seq_idx],
            )
        elif scene_obs is not None or robot_obs is not None:
            obs = self.env.reset(scene_obs=scene_obs, robot_obs=robot_obs)
        else:
            obs = self.env.reset()
        return obs

    def get_info(self):
        return self.env.get_info()

    def get_obs(self):
        return self.env.get_obs()

    def action_space(self):
        return self.env.action_space

    def observation_space(self):
        return self.env.observation_space
