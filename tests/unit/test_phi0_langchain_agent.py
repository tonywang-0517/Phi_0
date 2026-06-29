"""Unit tests for Phi0 LangChain agent (no GPU)."""

from __future__ import annotations

import ast
from pathlib import Path

from phi0.agent.checkpoints import DEFAULT_SKILL_CHECKPOINTS, resolve_skill_checkpoint
from phi0.agent.prompts import ROBOT_SYSTEM_PROMPT_ZH, build_agent_user_turn
from phi0.agent.qwen_vl_chat import _parse_tool_calls, _strip_tool_markup
from phi0.agent.robot_agent import _output_without_follow_up
from langchain_core.messages import AIMessage


def test_system_prompt_lists_three_skills():
    assert "pick_tissues" in ROBOT_SYSTEM_PROMPT_ZH
    assert "throw_rubbish" in ROBOT_SYSTEM_PROMPT_ZH
    assert "stay" in ROBOT_SYSTEM_PROMPT_ZH


def test_throw_rubbish_fallback_to_pick_ckpt():
    root = Path(__file__).resolve().parents[2]
    ckpt, used_fallback = resolve_skill_checkpoint(
        DEFAULT_SKILL_CHECKPOINTS["throw_rubbish"],
        root=root,
    )
    assert ckpt.is_file()
    assert used_fallback is True


def test_parse_qwen_tool_call_block():
    raw = (
        "好的，我来帮你。\n"
        "<tool_call>\n"
        '{"name": "pick_tissues", "arguments": {}}\n'
        "</tool_call>"
    )
    calls = _parse_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0]["name"] == "pick_tissues"


def test_output_without_follow_up_stay():
    out = _output_without_follow_up(AIMessage(content=""), [{"tool": "stay", "result": "{}"}])
    assert "保持不动" in out


def test_stay_tool_no_phi0():
    from PIL import Image

    from phi0.agent.tools import bind_runtime, stay

    bind_runtime(None, ego_image=Image.new("RGB", (8, 8)), dry_run=False)
    payload = ast.literal_eval(stay.invoke({}))
    assert payload["skill"] == "stay"
    assert "未调用 Phi0" in payload["message"]


def test_pick_dry_run_resolves_checkpoint():
    from PIL import Image

    from phi0.agent.executor import Phi0SkillRouter
    from phi0.agent.tools import bind_runtime, pick_tissues

    router = Phi0SkillRouter()
    bind_runtime(router, ego_image=Image.new("RGB", (8, 8)), dry_run=True)
    payload = ast.literal_eval(pick_tissues.invoke({}))
    assert payload["skill"] == "pick_tissues"
    assert payload["checkpoint"]
