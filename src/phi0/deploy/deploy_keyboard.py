"""Keyboard -> ZMQ command topic for SONIC deploy (zmq_manager on port 5556)."""

from __future__ import annotations

import logging
import select
import sys
import termios
import threading
import tty
from dataclasses import dataclass, field
from typing import Callable

import zmq

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message

logger = logging.getLogger(__name__)

HELP_TEXT = """
[deploy keyboard] SONIC deploy controls (ZMQ command @ pose port):
  k     start/stop C++ control loop (planner mode; g1_debug @ 5557)
  ]     enter CONTROL + streamed motion (Phi-0 tokens — use after tx frame=1)
  p     enter CONTROL + planner mode
  Enter toggle planner <-> streamed motion (when in CONTROL)
  O/o   emergency STOP
  h/?   show this help
  q     quit closed-loop publisher (Ctrl-C also works)

Recommended: k (control loop) -> wait robot proprio active -> ] or Enter (streaming)
""".strip()


@dataclass
class DeployCommandState:
    planner_mode: bool = False
    control_active: bool = False
    cpp_loop_running: bool = False


def send_deploy_command(
    pub: zmq.Socket,
    *,
    start: bool = False,
    stop: bool = False,
    planner: bool = False,
    send_lock: threading.Lock | None = None,
) -> None:
    msg = build_command_message(start=start, stop=stop, planner=planner)
    if send_lock is not None:
        with send_lock:
            pub.send(msg)
    else:
        pub.send(msg)


def send_start_streamed(pub: zmq.Socket, send_lock: threading.Lock | None = None) -> None:
    send_deploy_command(pub, start=True, stop=False, planner=True, send_lock=send_lock)
    send_deploy_command(pub, start=True, stop=False, planner=False, send_lock=send_lock)


def handle_deploy_key(
    key: str,
    pub: zmq.Socket,
    state: DeployCommandState,
    *,
    send_lock: threading.Lock | None = None,
    on_quit: Callable[[], None] | None = None,
) -> bool:
    """Handle one keypress. Returns False if listener should exit."""
    if key in {"h", "?", "H"}:
        logger.info("\n%s", HELP_TEXT)
        return True
    if key in {"q", "Q", "\x03"}:
        logger.info("[deploy keyboard] quit requested")
        if on_quit is not None:
            on_quit()
        return False
    if key in {"k", "K"}:
        if state.cpp_loop_running:
            send_deploy_command(
                pub,
                start=False,
                stop=True,
                planner=state.planner_mode,
                send_lock=send_lock,
            )
            state.cpp_loop_running = False
            state.control_active = False
            logger.info(
                "[deploy keyboard] stopped C++ control loop (was %s)",
                "planner" if state.planner_mode else "streamed",
            )
        else:
            state.planner_mode = True
            state.control_active = True
            state.cpp_loop_running = True
            send_deploy_command(pub, start=True, stop=False, planner=True, send_lock=send_lock)
            logger.info(
                "[deploy keyboard] started C++ control loop in planner mode (k) — "
                "wait for g1_debug, then ] for streaming"
            )
        return True
    if key == "]":
        state.planner_mode = False
        state.control_active = True
        state.cpp_loop_running = True
        send_start_streamed(pub, send_lock=send_lock)
        logger.info("[deploy keyboard] CONTROL + streamed motion (])")
        return True
    if key in {"p", "P"}:
        state.planner_mode = True
        state.control_active = True
        state.cpp_loop_running = True
        send_deploy_command(pub, start=True, stop=False, planner=True, send_lock=send_lock)
        logger.info("[deploy keyboard] CONTROL + planner mode (p)")
        return True
    if key in {"\n", "\r"}:
        if not state.control_active:
            state.planner_mode = False
            state.control_active = True
            state.cpp_loop_running = True
            send_start_streamed(pub, send_lock=send_lock)
            logger.info("[deploy keyboard] CONTROL + streamed motion (Enter)")
            return True
        state.planner_mode = not state.planner_mode
        send_deploy_command(
            pub,
            start=True,
            stop=False,
            planner=state.planner_mode,
            send_lock=send_lock,
        )
        mode = "planner" if state.planner_mode else "streamed motion"
        logger.info("[deploy keyboard] toggle -> %s (Enter)", mode)
        return True
    if key in {"o", "O"}:
        state.control_active = False
        state.cpp_loop_running = False
        send_deploy_command(pub, start=False, stop=True, planner=False, send_lock=send_lock)
        logger.info("[deploy keyboard] emergency STOP (O)")
        return True
    return True


@dataclass
class DeployKeyboardListener:
    pub: zmq.Socket
    stop_event: threading.Event
    send_lock: threading.Lock = field(default_factory=threading.Lock)
    state: DeployCommandState = field(default_factory=DeployCommandState)
    on_quit: Callable[[], None] | None = None
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def start(self) -> bool:
        if not sys.stdin.isatty():
            logger.info("[deploy keyboard] disabled (stdin is not a TTY)")
            return False
        self._thread = threading.Thread(
            target=self._run,
            name="deploy_keyboard",
            daemon=True,
        )
        self._thread.start()
        logger.info("[deploy keyboard] active — press h for help")
        return True

    def _run(self) -> None:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self.stop_event.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue
                key = sys.stdin.read(1)
                if not key:
                    continue
                if not handle_deploy_key(
                    key,
                    self.pub,
                    self.state,
                    send_lock=self.send_lock,
                    on_quit=self.on_quit,
                ):
                    self.stop_event.set()
                    break
        except Exception:
            logger.exception("[deploy keyboard] listener failed")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
