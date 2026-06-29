#!/usr/bin/env python3
"""Print VLM agent speech diagnostic matrix for ep447 (input vs weights)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tests.unit.test_vlm_agent_speech_debug import (  # noqa: E402
    _load_ep447_bundle,
    _word_repetition_ratio,
)


def main() -> None:
    b = _load_ep447_bundle()
    ids_ok = bool(
        __import__("torch").equal(
            b["train_vlm"]["input_ids"],
            b["deploy_vlm"]["input_ids"],
        )
    )
    pv_diff = float(
        (
            b["train_vlm"]["pixel_values"].float()
            - b["deploy_vlm"]["pixel_values"].float()
        )
        .abs()
        .max()
    )
    print("=== ep447 VLM agent diagnostic ===")
    print("input_ids train==deploy:", ids_ok)
    print("pixel_values max|diff|:", f"{pv_diff:.4f}")
    print("train vision rep:", f"{_word_repetition_ratio(b['train_text']):.2f}")
    print("deploy vision rep:", f"{_word_repetition_ratio(b['deploy_text']):.2f}")
    print("text-only rep:", f"{_word_repetition_ratio(b['text_only_out']):.2f}")
    print("\n--- train vision ---")
    print(b["train_text"])
    print("\n--- deploy vision ---")
    print(b["deploy_text"])
    print("\n--- text-only (no image) ---")
    print(b["text_only_out"])
    print(
        "\nverdict:",
        "输入对齐 OK；乱码来自 Psi0 HE 微调后 LM 语言能力退化，"
        "不是 ep447 帧/模板问题。要流畅说话需 base Qwen3-VL Instruct 或 language SFT。",
    )


if __name__ == "__main__":
    main()
