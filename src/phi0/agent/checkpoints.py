"""Per-skill Phi0 checkpoint registry (swap paths when new skills are trained)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PHI0_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SkillCheckpointSpec:
    skill: str
    checkpoint: str
    config_name: str
    fallback_checkpoint: str | None = None


DEFAULT_SKILL_CHECKPOINTS: dict[str, SkillCheckpointSpec] = {
    "pick_tissues": SkillCheckpointSpec(
        skill="pick_tissues",
        checkpoint=(
            "experiments/pick_tissue_xperience_unified_3k_ddp4_fast/"
            "pick_tissue_xperience_unified_act_latest.pt"
        ),
        config_name="train_pick_tissue_xperience_unified_ddp4_3k",
    ),
    "throw_rubbish": SkillCheckpointSpec(
        skill="throw_rubbish",
        checkpoint=(
            "experiments/throw_rubbish_xperience_unified/"
            "throw_rubbish_xperience_unified_act_latest.pt"
        ),
        config_name="train_pick_tissue_xperience_unified_ddp4_3k",
        fallback_checkpoint=(
            "experiments/pick_tissue_xperience_unified_3k_ddp4_fast/"
            "pick_tissue_xperience_unified_act_latest.pt"
        ),
    ),
}


def resolve_skill_checkpoint(
    spec: SkillCheckpointSpec,
    *,
    root: Path | None = None,
) -> tuple[Path, bool]:
    base = root or PHI0_ROOT
    primary = (base / spec.checkpoint).resolve()
    if primary.is_file():
        return primary, False
    if spec.fallback_checkpoint:
        fallback = (base / spec.fallback_checkpoint).resolve()
        if fallback.is_file():
            logger.warning(
                "skill %s ckpt missing at %s; using fallback %s",
                spec.skill,
                primary,
                fallback,
            )
            return fallback, True
    raise FileNotFoundError(
        f"checkpoint for skill {spec.skill!r} not found: {primary}"
        + (
            f" (fallback {spec.fallback_checkpoint} also missing)"
            if spec.fallback_checkpoint
            else ""
        )
    )


def skill_checkpoint_overrides(
    *,
    pick_tissues: str | None = None,
    throw_rubbish: str | None = None,
    config_name: str | None = None,
) -> dict[str, SkillCheckpointSpec]:
    out = dict(DEFAULT_SKILL_CHECKPOINTS)
    if pick_tissues:
        spec = out["pick_tissues"]
        out["pick_tissues"] = SkillCheckpointSpec(
            skill=spec.skill,
            checkpoint=pick_tissues,
            config_name=config_name or spec.config_name,
            fallback_checkpoint=spec.fallback_checkpoint,
        )
    if throw_rubbish:
        spec = out["throw_rubbish"]
        out["throw_rubbish"] = SkillCheckpointSpec(
            skill=spec.skill,
            checkpoint=throw_rubbish,
            config_name=config_name or spec.config_name,
            fallback_checkpoint=spec.fallback_checkpoint,
        )
    elif config_name:
        for key, spec in out.items():
            out[key] = SkillCheckpointSpec(
                skill=spec.skill,
                checkpoint=spec.checkpoint,
                config_name=config_name,
                fallback_checkpoint=spec.fallback_checkpoint,
            )
    return out
