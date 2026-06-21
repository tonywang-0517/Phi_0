"""Shared Video2World conditioning sizes (Cosmos / VGGT / action proprio prefix)."""

from __future__ import annotations

from phi0.data.temporal_align import video2world_cond_pixel_frames

# Official Cosmos Predict2.5 multi-frame V2W (num_latent_conditional_frames=2).
DEFAULT_NUM_LATENT_CONDITIONAL_FRAMES = 2


def v2w_cond_pixel_frames(num_latent_conditional_frames: int = DEFAULT_NUM_LATENT_CONDITIONAL_FRAMES) -> int:
    return video2world_cond_pixel_frames(int(num_latent_conditional_frames))


# Action ACT prefix: 4 past proprio steps + 1 current (= same count as V2W pixel cond).
ACTION_PROPRIO_PREFIX_SIZE = v2w_cond_pixel_frames(DEFAULT_NUM_LATENT_CONDITIONAL_FRAMES)


def validate_v2w_triple_align(
    *,
    past_action_window_size: int,
    num_latent_conditional_frames: int,
    seq_len: int,
) -> None:
    """Raise if Cosmos/VGGT/action prefix sizes or seq_len are inconsistent."""
    from phi0.data.temporal_align import training_v2w_cond_control_indices

    cond_px = v2w_cond_pixel_frames(num_latent_conditional_frames)
    w = int(past_action_window_size)
    if w != cond_px:
        raise ValueError(
            f"past_action_window_size={w} must equal official V2W cond pixel frames={cond_px} "
            f"(4 history + 1 current proprio prefix)."
        )
    if int(seq_len) <= w:
        raise ValueError(f"seq_len={seq_len} must exceed proprio prefix size={w}.")
    proprio_ctrl = list(range(w))
    cond_ctrl = training_v2w_cond_control_indices(
        past_action_window_size=w,
        cond_pixel_frames=cond_px,
    )
    if cond_ctrl[-1] != proprio_ctrl[-1]:
        raise ValueError(
            f"V2W cond must end at proprio current step {proprio_ctrl[-1]}, got {cond_ctrl[-1]}"
        )
    if not set(cond_ctrl) & set(proprio_ctrl):
        raise ValueError(
            f"V2W cond control indices {cond_ctrl} must overlap proprio prefix {proprio_ctrl}"
        )
