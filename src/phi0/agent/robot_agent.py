"""Assemble LangChain agent: official Qwen3-VL + Phi0 tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from PIL import Image

from phi0.agent.executor import Phi0SkillRouter
from phi0.agent.prompts import ROBOT_SYSTEM_PROMPT_ZH, build_agent_user_turn
from phi0.agent.qwen_vl_chat import ChatQwen3VLLocal
from phi0.agent.tools import ROBOT_TOOLS, bind_runtime
from phi0.models.vlm.tower import GenerateTextConfig


def build_robot_agent(
    llm: ChatQwen3VLLocal | None = None,
    *,
    model_path: str = "Qwen/Qwen3-VL-2B-Instruct",
    device: str = "cuda",
    gen_cfg: GenerateTextConfig | None = None,
    phi0_router: Phi0SkillRouter | None = None,
) -> "RobotAgent":
    if llm is None:
        llm = ChatQwen3VLLocal(
            model_path=model_path,
            device=device,
            gen_cfg=gen_cfg or GenerateTextConfig(max_new_tokens=256, do_sample=False),
        )
    return RobotAgent(llm=llm, phi0_router=phi0_router)


def _output_without_follow_up(ai_msg: AIMessage, tool_steps: list[dict[str, Any]]) -> str:
    if ai_msg.content and str(ai_msg.content).strip():
        return str(ai_msg.content).strip()
    if not tool_steps:
        return ""
    names = [s["tool"] for s in tool_steps]
    if names == ["stay"]:
        return "好的，我会保持不动。有什么需要再告诉我。"
    return f"好的，正在执行：{', '.join(names)}。"


@dataclass
class RobotAgent:
    llm: ChatQwen3VLLocal
    phi0_router: Phi0SkillRouter | None = None

    def _human_message(
        self,
        user_text: str,
        ego_image: Image.Image,
        wrist_image: Image.Image | None,
    ) -> HumanMessage:
        parts: list[dict[str, Any]] = [{"type": "image", "image": ego_image.convert("RGB")}]
        if wrist_image is not None:
            parts.append({"type": "image", "image": wrist_image.convert("RGB")})
        parts.append({"type": "text", "text": user_text})
        return HumanMessage(content=parts)

    def run(
        self,
        user_instruction: str,
        ego_image: Image.Image,
        *,
        wrist_image: Image.Image | None = None,
        chat_history: Sequence[tuple[str, str]] | None = None,
        dry_run: bool = False,
        follow_up_reply: bool = False,
    ) -> dict[str, Any]:
        user_text = build_agent_user_turn(user_instruction)
        bind_runtime(
            self.phi0_router,
            ego_image=ego_image,
            wrist_image=wrist_image,
            dry_run=dry_run or self.phi0_router is None,
        )

        messages: list[Any] = [SystemMessage(content=ROBOT_SYSTEM_PROMPT_ZH)]
        if chat_history:
            for role, text in chat_history:
                if role == "human":
                    messages.append(HumanMessage(content=text))
                elif role == "ai":
                    messages.append(AIMessage(content=text))
        messages.append(self._human_message(user_text, ego_image, wrist_image))

        ai_msg: AIMessage = self.llm.bind_tools(ROBOT_TOOLS).invoke(messages)
        tool_steps: list[dict[str, Any]] = []
        if ai_msg.tool_calls:
            tool_map = {t.name: t for t in ROBOT_TOOLS}
            for tc in ai_msg.tool_calls:
                name = tc["name"]
                tool_fn = tool_map.get(name)
                out = tool_fn.invoke(tc.get("args") or {}) if tool_fn else f"unknown tool {name!r}"
                tool_steps.append({"tool": name, "args": tc.get("args"), "result": out})

        if follow_up_reply and ai_msg.tool_calls:
            from langchain_core.messages import ToolMessage

            messages.append(ai_msg)
            for tc, step in zip(ai_msg.tool_calls, tool_steps):
                messages.append(
                    ToolMessage(content=str(step["result"]), tool_call_id=tc["id"], name=step["tool"])
                )
            output = str(self.llm.invoke(messages).content)
        else:
            output = _output_without_follow_up(ai_msg, tool_steps)

        selected_skill = tool_steps[0]["tool"] if len(tool_steps) == 1 else None
        return {
            "user_instruction": user_instruction,
            "output": output,
            "tool_steps": tool_steps,
            "first_pass": ai_msg.content,
            "selected_skill": selected_skill,
        }
