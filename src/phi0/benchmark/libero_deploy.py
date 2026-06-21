"""Shared LIBERO deploy configuration and action I/O helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np
import torch

from phi0.benchmark.adapters import process_libero_absolute_eef_action, process_vla_action


class _PolicyCfg(Protocol):
    libero_delta_eef: bool | None
    libero_proprio_absolute: bool | None
    libero_absolute_eef: bool | None


def _data_get(data_cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(data_cfg, Mapping):
        return data_cfg.get(key, default)
    return getattr(data_cfg, key, default)


@dataclass(frozen=True)
class LiberoDeployFlags:
    delta_eef: bool
    proprio_absolute: bool
    absolute_eef: bool

    def use_proprio_stats(self, processor: Any) -> bool:
        return self.delta_eef and hasattr(processor, "proprio_mean")


def resolve_libero_deploy_flags(
    policy_cfg: _PolicyCfg,
    train_cfg: Any,
) -> LiberoDeployFlags:
    """Resolve LIBERO deploy flags; explicit policy_cfg overrides train_cfg.data."""
    data = getattr(train_cfg, "data", train_cfg)

    if policy_cfg.libero_delta_eef is not None:
        delta_eef = bool(policy_cfg.libero_delta_eef)
    else:
        delta_eef = bool(_data_get(data, "libero_delta_eef", False))

    if policy_cfg.libero_proprio_absolute is not None:
        proprio_absolute = bool(policy_cfg.libero_proprio_absolute)
    else:
        proprio_absolute = bool(
            _data_get(
                data,
                "libero_proprio_absolute",
                _data_get(data, "libero_absolute_eef", True),
            )
        )

    if policy_cfg.libero_absolute_eef is not None:
        absolute_eef = bool(policy_cfg.libero_absolute_eef)
    else:
        absolute_eef = bool(_data_get(data, "libero_absolute_eef", not delta_eef))

    return LiberoDeployFlags(
        delta_eef=delta_eef,
        proprio_absolute=proprio_absolute,
        absolute_eef=absolute_eef,
    )


def normalize_libero_proprio_eef_7d(
    processor: Any,
    model: Any,
    eef_7d: np.ndarray,
    flags: LiberoDeployFlags,
) -> torch.Tensor:
    """Sim absolute EEF [7] -> normalized proprio token on model device."""
    t7 = torch.from_numpy(np.asarray(eef_7d, dtype=np.float32)).view(1, 1, 7)
    with torch.no_grad():
        if model.uses_robot7d_action():
            enc = processor.normalize_robot7d_tensor(
                t7.to(device=model.device),
                proprio=flags.use_proprio_stats(processor),
            ).cpu()
        else:
            from phi0.schema.draw_schema import D_RAW

            enc = torch.zeros(1, 1, D_RAW, dtype=torch.float32)
            enc[0, 0, :7] = t7[0, 0]
            if processor.normalize:
                enc = processor._normalize_action(enc)
    return enc[0, 0].to(device=model.device, dtype=model.torch_dtype)


def postprocess_libero_robot7d_chunk(
    d7: np.ndarray,
    flags: LiberoDeployFlags,
    *,
    invert_openvla_gripper: bool,
) -> np.ndarray:
    """Denormalized robot7d chunk -> LIBERO sim actions [T, 7]."""
    chunk = np.asarray(d7, dtype=np.float32)
    invert = bool(invert_openvla_gripper)

    if flags.absolute_eef and not flags.delta_eef:
        chunk = chunk.copy()
        chunk[:, 6] = np.clip(chunk[:, 6], 0.0, 1.0)
        return process_libero_absolute_eef_action(chunk, invert_openvla_gripper=invert)

    if flags.delta_eef:
        chunk = np.clip(chunk, -1.0, 1.0).astype(np.float32)
        chunk[:, 6] = np.clip(chunk[:, 6], 0.0, 1.0)
        return process_vla_action(chunk, invert_openvla_gripper=invert)

    chunk = np.clip(chunk, -1.0, 1.0).astype(np.float32)
    chunk[:, 6] = np.clip(chunk[:, 6], 0.0, 1.0)
    return chunk
