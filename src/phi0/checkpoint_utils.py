"""Checkpoint config merge and state_dict load logging."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def merge_saved_cfg(cfg: DictConfig, saved: Optional[Dict[str, Any]]) -> DictConfig:
    """Merge full checkpoint ``cfg`` into Hydra-composed config (saved wins on conflict)."""
    if not saved:
        return cfg
    saved_cfg = OmegaConf.create(saved)
    # Drop keys removed from current schema (e.g. deprecated latent_cache_path).
    if OmegaConf.is_config(cfg):
        _prune_unknown_keys(saved_cfg, cfg)
    return OmegaConf.merge(cfg, saved_cfg)


def _prune_unknown_keys(src: DictConfig, template: DictConfig) -> None:
    """Remove ``src`` keys that no longer exist in ``template`` (struct-safe merge)."""
    if not OmegaConf.is_config(src) or not OmegaConf.is_config(template):
        return
    for key in list(src.keys()):
        if key not in template:
            del src[key]
        else:
            child_src = src[key]
            child_tpl = template[key]
            if OmegaConf.is_config(child_src) and OmegaConf.is_config(child_tpl):
                _prune_unknown_keys(child_src, child_tpl)


def load_model_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    strict: bool = False,
    source: str = "",
) -> None:
    """Load weights and log missing/unexpected keys (never silent with strict=False)."""
    incompatible = model.load_state_dict(state_dict, strict=strict)
    prefix = f"{source}: " if source else ""
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing:
        logger.warning("%sMissing keys (%d): %s", prefix, len(missing), missing[:20])
        if len(missing) > 20:
            logger.warning("%s... and %d more missing keys", prefix, len(missing) - 20)
    if unexpected:
        logger.warning("%sUnexpected keys (%d): %s", prefix, len(unexpected), unexpected[:20])
        if len(unexpected) > 20:
            logger.warning("%s... and %d more unexpected keys", prefix, len(unexpected) - 20)
    if not missing and not unexpected:
        logger.info("%sState dict loaded with no missing/unexpected keys", prefix.rstrip(": "))


def checkpoint_paths(
    output_dir: Union[str, Any],
    checkpoint_name: str,
    step: int,
) -> tuple[Any, Any, Any]:
    """Return (step_path, latest_path, legacy_alias_path) for saving."""
    from pathlib import Path

    out = Path(str(output_dir))
    name = str(checkpoint_name or "phi0")
    step_path = out / f"{name}_step{step}.pt"
    latest_path = out / f"{name}_latest.pt"
    legacy_alias = out / "phi0_smoke.pt"
    return step_path, latest_path, legacy_alias


def extract_action_expert_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Strip ``action_expert.*`` weights for compact checkpoints."""
    prefix = "action_expert."
    return {k[len(prefix) :]: v for k, v in model.state_dict().items() if k.startswith(prefix)}


def load_action_expert_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor], *, source: str = "") -> None:
    if not hasattr(model, "action_expert"):
        raise AttributeError("Model has no action_expert module.")
    remapped = dict(state_dict)
    if "input_proj.weight" in remapped and "action_encoder.weight" not in remapped:
        remapped["action_encoder.weight"] = remapped.pop("input_proj.weight")
        if "input_proj.bias" in remapped:
            remapped["action_encoder.bias"] = remapped.pop("input_proj.bias")
    incompatible = model.action_expert.load_state_dict(remapped, strict=False)
    prefix = f"{source}: " if source else ""
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        logger.warning(
            "%saction_expert load: missing=%d unexpected=%d",
            prefix,
            len(missing),
            len(unexpected),
        )
    else:
        logger.info("%saction_expert loaded (%d tensors)", prefix.rstrip(": "), len(state_dict))
