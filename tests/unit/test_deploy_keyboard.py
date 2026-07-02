"""Deploy keyboard command mapping."""

from __future__ import annotations

import threading
from unittest import mock

import zmq

from phi0.deploy.deploy_keyboard import (
    DeployCommandState,
    handle_deploy_key,
    send_start_streamed,
)


def test_handle_deploy_key_k_toggles_control_loop():
    pub = mock.Mock()
    lock = threading.Lock()
    state = DeployCommandState()
    assert handle_deploy_key("k", pub, state, send_lock=lock)
    assert state.cpp_loop_running
    assert state.planner_mode
    pub.send.assert_called_once()

    assert handle_deploy_key("k", pub, state, send_lock=lock)
    assert not state.cpp_loop_running
    assert not state.control_active
    assert pub.send.call_count == 2


def test_handle_deploy_key_stop_and_start():
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind("inproc://test_deploy_kb")
    sub = ctx.socket(zmq.SUB)
    sub.connect("inproc://test_deploy_kb")
    sub.setsockopt_string(zmq.SUBSCRIBE, "command")

    lock = threading.Lock()
    state = DeployCommandState()
    assert handle_deploy_key("]", pub, state, send_lock=lock)
    assert state.control_active
    assert not state.planner_mode

    assert handle_deploy_key("O", pub, state, send_lock=lock)
    assert not state.control_active

    pub.close(linger=0)
    sub.close(linger=0)


def test_send_start_streamed_sends_two_commands():
    pub = mock.Mock()
    send_start_streamed(pub)
    assert pub.send.call_count == 2
