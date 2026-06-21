"""Cross-attention mode helpers for action DiT blocks."""

from __future__ import annotations

from typing import Literal, Optional

ActionCrossAttnMode = Literal["interleave_vlm", "dual_vlm_vggt", "all_vlm", "self_only"]

_LEGACY_MODE_MAP = {
    "interleave_cosmos": "interleave_vlm",
    "dual_cosmos_vggt": "dual_vlm_vggt",
    "all_cosmos": "all_vlm",
}


def resolve_action_cross_attn_mode(
    action_cross_attn_mode: Optional[str] = None,
    *,
    interleave_self_attention: Optional[bool] = None,
) -> ActionCrossAttnMode:
    """Resolve mode from explicit config or legacy ``interleave_self_attention``."""
    if action_cross_attn_mode is not None:
        key = str(action_cross_attn_mode).strip().lower()
        key = _LEGACY_MODE_MAP.get(key, key)
        if key in {"interleave_vlm", "dual_vlm_vggt", "all_vlm", "self_only"}:
            return key  # type: ignore[return-value]
        raise ValueError(
            f"Unknown action_cross_attn_mode={action_cross_attn_mode!r}; "
            "expected interleave_vlm | dual_vlm_vggt | all_vlm | self_only"
        )
    if interleave_self_attention is False:
        return "all_vlm"
    return "interleave_vlm"


def cross_attn_target(mode: ActionCrossAttnMode, block_idx: int) -> Optional[str]:
    """Return cross-attn context key for a block: vlm, vggt, or None."""
    if mode == "self_only":
        return None
    if mode == "all_vlm":
        return "vlm"
    if mode == "dual_vlm_vggt":
        return "vlm" if block_idx % 2 == 0 else "vggt"
    # interleave_vlm: even layers cross-attend VLM, odd self-only.
    return "vlm" if block_idx % 2 == 0 else None
