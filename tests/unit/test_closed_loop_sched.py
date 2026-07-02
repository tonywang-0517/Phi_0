"""Unit tests for closed-loop prefetch scheduling."""

from phi0.inference.closed_loop_sched import (
    chunk_index_when_result_arrives,
    inference_pipeline_idle,
    prefetch_trigger_steps,
    resolve_min_inference_interval,
    should_trigger_chunk_prefetch,
)


def test_resolve_min_inference_interval():
    assert resolve_min_inference_interval(0) is None
    assert resolve_min_inference_interval(2.5) == 0.4


def test_prefetch_steps_for_300ms_budget():
    assert prefetch_trigger_steps(0.3, 50.0, safety_steps=2) == 17


def test_chunk_index_from_control_steps():
    assert chunk_index_when_result_arrives(15, 0, 32) == 15
    assert chunk_index_when_result_arrives(40, 25, 32) == 15
    assert chunk_index_when_result_arrives(100, 0, 32) == 31


def test_prefetch_trigger_when_running_low():
    assert should_trigger_chunk_prefetch(
        cached_chunk_exists=True,
        pipeline_idle=True,
        action_chunk_index=31,
        action_horizon=32,
        prefetch_steps=17,
        min_interval_s=None,
        time_since_last_trigger_s=0.0,
        steps_since_chunk_consumed=1,
    )
    assert not should_trigger_chunk_prefetch(
        cached_chunk_exists=True,
        pipeline_idle=True,
        action_chunk_index=5,
        action_horizon=32,
        prefetch_steps=17,
        min_interval_s=None,
        time_since_last_trigger_s=0.0,
        steps_since_chunk_consumed=1,
    )


def test_prefetch_skips_immediate_retrigger_on_fresh_chunk():
    assert not should_trigger_chunk_prefetch(
        cached_chunk_exists=True,
        pipeline_idle=True,
        action_chunk_index=15,
        action_horizon=32,
        prefetch_steps=17,
        min_interval_s=None,
        time_since_last_trigger_s=0.0,
        steps_since_chunk_consumed=0,
    )
    assert should_trigger_chunk_prefetch(
        cached_chunk_exists=True,
        pipeline_idle=True,
        action_chunk_index=31,
        action_horizon=32,
        prefetch_steps=17,
        min_interval_s=None,
        time_since_last_trigger_s=0.0,
        steps_since_chunk_consumed=0,
    )


def test_pipeline_idle_requires_no_pending_work():
    assert inference_pipeline_idle(worker_busy=False, inference_queue_pending=False, result_queue_pending=False)
    assert not inference_pipeline_idle(worker_busy=True, inference_queue_pending=False, result_queue_pending=False)
    assert not inference_pipeline_idle(worker_busy=False, inference_queue_pending=True, result_queue_pending=False)
    assert not inference_pipeline_idle(worker_busy=False, inference_queue_pending=False, result_queue_pending=True)
