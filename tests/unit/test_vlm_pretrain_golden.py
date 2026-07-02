"""Golden test: Phi_0 pretrain-aligned VLM inputs must match Psi0ModelTransform.

Tests that ``build_pretrain_aligned_vlm_inputs`` / ``build_qwenvl_inputs_single``
produce identical ``input_ids``, ``attention_mask``, and ``image_grid_thw`` tensors
to the Psi0 reference path (``Psi0ModelTransform.build_qwenvl_inputs``) on the same
synthetic PIL images and instruction.

Skip conditions:
  - Processor checkpoint not found on disk (``PHI0_VLM_CKPT`` env or default path)
  - ``qwen_vl_utils`` not installed

Run:
  pytest tests/unit/test_vlm_pretrain_golden.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Checkpoint / processor resolution
# ---------------------------------------------------------------------------

_DEFAULT_CKPT = Path(
    "./checkpoints/psi0"
    "/pre.fast.1by1.2601091803.ckpt.ego200k.he30k"
    "/psi0"
    "/pre.fast.1by1.2601091803.ckpt.ego200k.he30k"
)

_CKPT_PATH = Path(
    os.environ.get("PHI0_VLM_CKPT", str(_DEFAULT_CKPT))
)

_PROCESSOR_AVAILABLE = (_CKPT_PATH / "tokenizer_config.json").is_file()

needs_processor = pytest.mark.skipif(
    not _PROCESSOR_AVAILABLE,
    reason=f"Qwen3-VL processor not found at {_CKPT_PATH}; "
           "set PHI0_VLM_CKPT to override",
)

# ---------------------------------------------------------------------------
# qwen_vl_utils guard
# ---------------------------------------------------------------------------

try:
    from qwen_vl_utils import process_vision_info as _pvinfo  # noqa: F401
    _QWEN_VL_UTILS_OK = True
except ImportError:
    _QWEN_VL_UTILS_OK = False

needs_qwen_vl_utils = pytest.mark.skipif(
    not _QWEN_VL_UTILS_OK,
    reason="qwen_vl_utils not installed",
)


# ---------------------------------------------------------------------------
# Psi0ModelTransform.build_qwenvl_inputs  (inline reference replica)
# ---------------------------------------------------------------------------

def _psi0_build_qwenvl_inputs(
    vlm_processor: Any,
    imgs: List[Image.Image],
    instruction: str,
) -> Dict[str, torch.Tensor]:
    """Exact replication of Psi0ModelTransform.build_qwenvl_inputs (single sample).

    Source: Psi0-main/src/psi/config/transform.py  Psi0ModelTransform.build_qwenvl_inputs
    Differences from the batched version: single sample, no action/state inputs.
    """
    from qwen_vl_utils import process_vision_info

    content: list[dict] = [{"type": "image", "image": img} for img in imgs]
    content.append({"type": "text", "text": instruction})
    user_msg = {"role": "user", "content": content}
    messages = [[user_msg]]  # list-of-one-conversation

    texts = [
        vlm_processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
    return vlm_processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vlm_processor():
    """Load Qwen3VLProcessor once per test-module from local checkpoint."""
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(str(_CKPT_PATH), local_files_only=True)
    return proc


def _synthetic_pil(h: int, w: int, seed: int = 0) -> Image.Image:
    """Reproducible RGB PIL image of size (w, h) pixels."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_tensors_equal(
    phi0: Dict[str, torch.Tensor],
    psi0: Dict[str, torch.Tensor],
    key: str,
) -> None:
    assert key in phi0, f"Phi-0 output missing key '{key}'"
    assert key in psi0, f"Psi0 reference output missing key '{key}'"
    t_phi0 = phi0[key]
    t_psi0 = psi0[key]
    assert t_phi0.shape == t_psi0.shape, (
        f"Shape mismatch for '{key}': Phi-0={tuple(t_phi0.shape)} "
        f"Psi0={tuple(t_psi0.shape)}"
    )
    assert torch.equal(t_phi0, t_psi0), (
        f"Values differ for '{key}':\n"
        f"  Phi-0: {t_phi0}\n"
        f"  Psi0:  {t_psi0}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@needs_processor
@needs_qwen_vl_utils
class TestVlmPretrainGolden:
    """Phi-0 vs Psi0 VLM tokenization parity on synthetic images."""

    @pytest.fixture(autouse=True)
    def _proc(self, vlm_processor):
        self.processor = vlm_processor

    def _compare(
        self,
        images: List[Image.Image],
        instruction: str,
    ) -> None:
        from phi0.models.vlm.contract import build_pretrain_aligned_vlm_inputs
        from phi0.models.vlm.preprocess import (
            build_qwenvl_inputs_single,
            normalize_vlm_instruction,
        )

        normed = normalize_vlm_instruction(instruction)

        # --- Phi-0 paths ---
        phi0_contract = build_pretrain_aligned_vlm_inputs(
            self.processor, images, instruction
        )
        phi0_single = build_qwenvl_inputs_single(
            self.processor, images, normed
        )

        # --- Psi0 reference (already lowercased by repacker in production) ---
        psi0_ref = _psi0_build_qwenvl_inputs(self.processor, images, normed)

        for key in ("input_ids", "attention_mask", "image_grid_thw"):
            _assert_tensors_equal(phi0_contract, psi0_ref, key)
            _assert_tensors_equal(phi0_single,   psi0_ref, key)

        # pixel_values: check shape + dtype match (values depend on processor internals)
        for path_name, phi_out in [("contract", phi0_contract), ("single", phi0_single)]:
            assert "pixel_values" in phi_out, f"{path_name}: missing pixel_values"
            assert phi_out["pixel_values"].shape == psi0_ref["pixel_values"].shape, (
                f"{path_name} pixel_values shape mismatch: "
                f"{tuple(phi_out['pixel_values'].shape)} vs {tuple(psi0_ref['pixel_values'].shape)}"
            )
            assert torch.equal(phi_out["pixel_values"], psi0_ref["pixel_values"]), (
                f"{path_name} pixel_values differ"
            )

    # --- concrete test cases ---

    def test_single_view_180x320(self):
        """Single egocentric view at Phi-0 pretrain VLM size."""
        self._compare(
            images=[_synthetic_pil(180, 320, seed=1)],
            instruction="pick tissue",
        )

    def test_single_view_224x224(self):
        """Single view at Psi0 default 224x224 — verifies tokenizer path, not spatial."""
        self._compare(
            images=[_synthetic_pil(224, 224, seed=2)],
            instruction="place the cup on the table",
        )

    def test_dual_view_180x320(self):
        """Two views (ego + wrist) at Phi-0 size."""
        self._compare(
            images=[_synthetic_pil(180, 320, seed=3), _synthetic_pil(180, 320, seed=4)],
            instruction="pick tissue",
        )

    def test_instruction_uppercase_normalized(self):
        """build_pretrain_aligned_vlm_inputs lowercases; Psi0 repacker also lowercases."""
        self._compare(
            images=[_synthetic_pil(180, 320, seed=5)],
            instruction="PICK TISSUE",  # contract normalizes this before calling processor
        )

    def test_instruction_with_whitespace(self):
        self._compare(
            images=[_synthetic_pil(180, 320, seed=6)],
            instruction="  pick tissue  ",
        )

    def test_image_grid_thw_values(self):
        """Explicit check: image_grid_thw encodes T=1, H_patches, W_patches."""
        from phi0.models.vlm.preprocess import build_qwenvl_inputs_single, normalize_vlm_instruction

        img = _synthetic_pil(180, 320, seed=7)
        out = build_qwenvl_inputs_single(self.processor, [img], "pick tissue")
        thw = out["image_grid_thw"]
        assert thw.ndim == 2 and thw.shape[1] == 3, f"Expected (N,3) thw, got {tuple(thw.shape)}"
        t, h_patches, w_patches = thw[0].tolist()
        assert t == 1, f"Expected T=1 for static image, got {t}"
        # Qwen3-VL merges spatial patches; patches must be > 0
        assert h_patches > 0 and w_patches > 0


# ---------------------------------------------------------------------------
# Standalone smoke test (no pytest, just run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _PROCESSOR_AVAILABLE:
        print(f"SKIP: processor not found at {_CKPT_PATH}")
    elif not _QWEN_VL_UTILS_OK:
        print("SKIP: qwen_vl_utils not installed")
    else:
        from transformers import AutoProcessor
        from phi0.models.vlm.contract import build_pretrain_aligned_vlm_inputs
        from phi0.models.vlm.preprocess import build_qwenvl_inputs_single, normalize_vlm_instruction

        proc = AutoProcessor.from_pretrained(str(_CKPT_PATH), local_files_only=True)
        img = _synthetic_pil(180, 320, seed=0)
        instr = "pick tissue"
        phi0_out = build_pretrain_aligned_vlm_inputs(proc, [img], instr)
        psi0_out = _psi0_build_qwenvl_inputs(proc, [img], normalize_vlm_instruction(instr))
        for k in ("input_ids", "attention_mask", "image_grid_thw", "pixel_values"):
            match = torch.equal(phi0_out[k], psi0_out[k])
            print(f"  {k}: {'OK' if match else 'MISMATCH'} shape={tuple(phi0_out[k].shape)}")
