"""LIBERO eval env helpers (OSC absolute EEF execution)."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Iterator


@contextmanager
def osc_pose_absolute_controller() -> Iterator[None]:
    """Force ``OSC_POSE`` with ``control_delta=false`` for absolute world-frame targets."""
    import robosuite as suite

    orig = suite.load_controller_config

    def _patched(default_controller: str = "OSC_POSE"):
        cfg = dict(orig(default_controller=default_controller))
        if str(cfg.get("type", default_controller)).upper() == "OSC_POSE":
            cfg["control_delta"] = False
        return cfg

    suite.load_controller_config = _patched  # type: ignore[method-assign]
    try:
        yield
    finally:
        suite.load_controller_config = orig  # type: ignore[method-assign]


def make_libero_offscreen_env(
    *,
    bddl_file_name: str,
    camera_heights: int,
    camera_widths: int,
    osc_absolute: bool = True,
    **kwargs: Any,
):
    """Construct ``OffScreenRenderEnv`` with optional OSC absolute pose control."""
    from libero.libero.envs import OffScreenRenderEnv

    ctx = osc_pose_absolute_controller() if osc_absolute else nullcontext()
    with ctx:
        return OffScreenRenderEnv(
            bddl_file_name=bddl_file_name,
            camera_heights=int(camera_heights),
            camera_widths=int(camera_widths),
            **kwargs,
        )
