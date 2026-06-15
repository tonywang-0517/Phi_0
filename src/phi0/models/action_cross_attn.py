"""Cross-attention mode helpers for action DiT blocks."""

from __future__ import annotations

from typing import Literal, Optional

ActionCrossAttnMode = Literal["interleave_cosmos", "dual_cosmos_vggt", "all_cosmos"]


def resolve_action_cross_attn_mode(
    action_cross_attn_mode: Optional[str] = None,
    *,
    interleave_self_attention: Optional[bool] = None,
) -> ActionCrossAttnMode:
    """Resolve mode from explicit config or legacy ``interleave_self_attention``."""
    if action_cross_attn_mode is not None:
        key = str(action_cross_attn_mode).strip().lower()
        if key in {"interleave_cosmos", "dual_cosmos_vggt", "all_cosmos"}:
            return key  # type: ignore[return-value]
        raise ValueError(
            f"Unknown action_cross_attn_mode={action_cross_attn_mode!r}; "
            "expected interleave_cosmos | dual_cosmos_vggt | all_cosmos"
        )
    if interleave_self_attention is False:
        return "all_cosmos"
    return "interleave_cosmos"


def cross_attn_target(mode: ActionCrossAttnMode, block_idx: int) -> Optional[str]:
    """Return cross-attn context key for a block: cosmos, vggt, or None."""
    if mode == "all_cosmos":
        return "cosmos"
    if mode == "dual_cosmos_vggt":
        return "cosmos" if block_idx % 2 == 0 else "vggt"
    # interleave_cosmos: even layers cross-attend Cosmos, odd self-only.
    return "cosmos" if block_idx % 2 == 0 else None
