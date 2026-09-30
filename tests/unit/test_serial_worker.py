"""`SerialWorker` runs calls in order, stays bounded, drains on flush, and keeps the context."""

from __future__ import annotations

import contextvars
import threading

import pytest

from batcher._internal.concurrency.serial import SerialWorker

pytestmark = pytest.mark.unit


def test_calls_run_in_submission_order_and_flush_waits_for_them():
    out: list[int] = []
    worker = SerialWorker("test-order")
    for i in range(200):
        assert worker.submit(out.append, i)
    assert worker.flush(10.0)
    assert out == list(range(200))


def test_a_full_queue_refuses_so_the_caller_runs_the_call_itself():
    gate, started = threading.Event(), threading.Event()

    def block() -> None:
        started.set()
        gate.wait(10.0)

    worker = SerialWorker("test-bounded", capacity=2)
    assert worker.submit(block)
    assert started.wait(10.0)  # the thread is busy and the queue is empty
    accepted = [worker.submit(lambda: None) for _ in range(5)]
    # The running call is off the queue, so exactly `capacity` more fit, then it refuses.
    assert accepted[:2] == [True, True] and not any(accepted[2:])
    gate.set()
    assert worker.flush(10.0)


def test_a_call_sees_the_context_it_was_submitted_in():
    var: contextvars.ContextVar[str] = contextvars.ContextVar("var", default="unset")
    seen: list[str] = []
    worker = SerialWorker("test-context")
    token = var.set("submitter")
    try:
        worker.submit(lambda: seen.append(var.get()))
    finally:
        var.reset(token)
    assert worker.flush(10.0)
    assert seen == ["submitter"]


def test_a_failing_call_does_not_stop_the_worker():
    out: list[str] = []
    worker = SerialWorker("test-failure")
    worker.submit(lambda: 1 / 0)
    worker.submit(out.append, "after")
    assert worker.flush(10.0)
    assert out == ["after"]
