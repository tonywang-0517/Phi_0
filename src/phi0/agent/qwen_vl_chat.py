"""LangChain ChatModel wrapper for local official Qwen3-VL (multimodal + tool calling)."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Sequence

import torch
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from phi0.models.vlm.preprocess import build_qwenvl_inputs_single
from phi0.models.vlm.tower import GenerateTextConfig, Qwen3VLTower, load_agent_speech_tower


def _message_content_to_parts(content: Any) -> tuple[list[Any], str]:
    """Extract PIL images and text from LangChain message content."""
    images: list[Any] = []
    text_parts: list[str] = []
    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
                elif block.get("type") == "image_url":
                    url = block.get("image_url", {})
                    if isinstance(url, dict):
                        url = url.get("url", "")
                    if isinstance(url, str) and url.startswith("data:"):
                        import base64
                        from io import BytesIO

                        from PIL import Image

                        b64 = url.split(",", 1)[1]
                        images.append(Image.open(BytesIO(base64.b64decode(b64))))
                elif block.get("type") == "image":
                    images.append(block["image"])
    else:
        text_parts.append(str(content))
    return images, "\n".join(p for p in text_parts if p).strip()


def _langchain_messages_to_qwen(
    messages: Sequence[BaseMessage],
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Flatten LangChain history into one user multimodal turn (ponytail: single-turn agent)."""
    system_texts: list[str] = []
    images: list[Any] = []
    user_texts: list[str] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_texts.append(str(msg.content))
        elif isinstance(msg, HumanMessage):
            imgs, txt = _message_content_to_parts(msg.content)
            images.extend(imgs)
            if txt:
                user_texts.append(txt)
        elif isinstance(msg, AIMessage):
            user_texts.append(f"[assistant]\n{msg.content}")
        elif isinstance(msg, ToolMessage):
            user_texts.append(f"[tool:{msg.name}]\n{msg.content}")
    prompt_lines = []
    if system_texts:
        prompt_lines.append("\n".join(system_texts))
    if user_texts:
        prompt_lines.append("\n".join(user_texts))
    instruction = "\n\n".join(prompt_lines).strip()
    return [[{"role": "user", "content": []}]], images if images else []


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse Qwen3-VL <tool_call> JSON blocks."""
    calls: list[dict[str, Any]] = []
    for block in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL):
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        name = payload.get("name") or payload.get("function")
        if isinstance(name, dict):
            name = name.get("name")
        args = payload.get("arguments") or payload.get("parameters") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if name:
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "name": str(name),
                    "args": args if isinstance(args, dict) else {},
                }
            )
    return calls


def _strip_tool_markup(text: str) -> str:
    return re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()


class ChatQwen3VLLocal(BaseChatModel):
    """Local Qwen3-VL-2B-Instruct for LangChain agent (official weights, not Psi0)."""

    model_path: str = Field(default="Qwen/Qwen3-VL-2B-Instruct")
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    gen_cfg: GenerateTextConfig = Field(default_factory=GenerateTextConfig)
    _tower: Qwen3VLTower | None = None
    bound_tools: list[BaseTool] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "chat-qwen3-vl-local"

    def _load_tower(self) -> Qwen3VLTower:
        if self._tower is None:
            dtype = getattr(torch, self.torch_dtype, torch.bfloat16)
            self._tower = load_agent_speech_tower(
                self.model_path,
                device=self.device,
                torch_dtype=dtype,
                attn_implementation="sdpa",
                local_files_only=False,
            )
        return self._tower

    def bind_tools(
        self,
        tools: Sequence[BaseTool | dict | type],
        **kwargs: Any,
    ) -> "ChatQwen3VLLocal":
        cloned = self.model_copy()
        cloned.bound_tools = list(tools)  # type: ignore[arg-type]
        return cloned.bind(**kwargs)

    def _build_vlm_inputs(
        self,
        messages: Sequence[BaseMessage],
    ) -> tuple[dict[str, torch.Tensor], str]:
        tower = self._load_tower()
        _, images = _langchain_messages_to_qwen(messages)
        system_texts = [str(m.content) for m in messages if isinstance(m, SystemMessage)]
        human_texts: list[str] = []
        for m in messages:
            if isinstance(m, HumanMessage):
                _, txt = _message_content_to_parts(m.content)
                if txt:
                    human_texts.append(txt)
        instruction = "\n\n".join(system_texts + human_texts).strip()
        if not images:
            raise ValueError("ChatQwen3VLLocal requires at least one image in messages")

        openai_tools = None
        if self.bound_tools:
            openai_tools = [convert_to_openai_tool(t) for t in self.bound_tools]

        proc = tower.processor
        from phi0.models.vlm.preprocess import _vlm_messages

        vlm_messages = _vlm_messages(images, instruction)
        if openai_tools:
            chat_text = proc.apply_chat_template(
                vlm_messages[0],
                tools=openai_tools,
                add_generation_prompt=True,
                tokenize=False,
            )
            vlm = build_qwenvl_inputs_single(proc, images, instruction, chat_text=chat_text)
        else:
            vlm = build_qwenvl_inputs_single(proc, images, instruction)
        return vlm, instruction

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        tower = self._load_tower()
        vlm, _ = self._build_vlm_inputs(messages)
        on_dev = {
            k: v.to(tower.device) if torch.is_tensor(v) else v
            for k, v in vlm.items()
        }
        gen_cfg = self.gen_cfg
        if self.bound_tools:
            # ponytail: official instruct emits <tool_call> blocks; keep MM token suppress off
            gen_cfg = GenerateTextConfig(
                max_new_tokens=gen_cfg.max_new_tokens,
                do_sample=gen_cfg.do_sample,
                temperature=gen_cfg.temperature,
                top_p=gen_cfg.top_p,
                repetition_penalty=gen_cfg.repetition_penalty,
                suppress_mm_tokens=False,
            )
        texts = tower.generate_text_from_vlm_batch(on_dev, gen_cfg=gen_cfg)
        raw = texts[0] if texts else ""
        tool_calls = _parse_tool_calls(raw) if self.bound_tools else []
        content = _strip_tool_markup(raw) if tool_calls else raw.strip()
        ai = AIMessage(content=content, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=ai)])
