"""LangChain tools bridging agent decisions to Phi0."""

from __future__ import annotations

from langchain_core.tools import tool

from phi0.agent.executor import Phi0SkillRouter

_ROUTER: Phi0SkillRouter | None = None
_EGO_IMAGE = None
_WRIST_IMAGE = None
_DRY_RUN: bool = False


def bind_runtime(
    router: Phi0SkillRouter | None,
    *,
    ego_image,
    wrist_image=None,
    dry_run: bool = False,
) -> None:
    global _ROUTER, _EGO_IMAGE, _WRIST_IMAGE, _DRY_RUN
    _ROUTER = router
    _EGO_IMAGE = ego_image
    _WRIST_IMAGE = wrist_image
    _DRY_RUN = bool(dry_run)


def _run_action_skill(skill: str) -> str:
    if _EGO_IMAGE is None:
        raise RuntimeError("call bind_runtime(..., ego_image=...) before invoking tools")
    if _DRY_RUN:
        from phi0.agent.executor import SKILL_TO_PHI0_INSTRUCTION

        ckpt = ""
        if _ROUTER is not None:
            try:
                ckpt = str(_ROUTER.checkpoint_for_skill(skill))
            except (KeyError, FileNotFoundError):
                ckpt = ""
        return str(
            {
                "skill": skill,
                "phi0_instruction": SKILL_TO_PHI0_INSTRUCTION.get(skill, ""),
                "status": "dry_run",
                "checkpoint": ckpt,
                "message": "dry-run：未执行 Phi0 predict。",
            }
        )
    if _ROUTER is None:
        raise RuntimeError("Phi0SkillRouter not configured")
    return str(_ROUTER.run_skill(skill, _EGO_IMAGE, wrist_image=_WRIST_IMAGE).to_dict())


@tool
def pick_tissues() -> str:
    """捡起纸巾（桌上、沙发上、地面等处的纸巾）。"""
    return _run_action_skill("pick_tissues")


@tool
def throw_rubbish() -> str:
    """把垃圾扔进垃圾桶。"""
    return _run_action_skill("throw_rubbish")


@tool
def stay() -> str:
    """保持原地不动，不执行操作（等待或纯对话）。"""
    return str(
        {
            "skill": "stay",
            "phi0_instruction": "",
            "status": "ok",
            "message": "保持不动，未调用 Phi0。",
        }
    )


ROBOT_TOOLS = [pick_tissues, throw_rubbish, stay]
