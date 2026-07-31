"""A ceiling on how much of a job may be spent retrying.

Per-task retry limits do not bound a job: `max_retries=2` over a hundred thousand tasks
authorizes two hundred thousand retries, and a fleet broken in some way no probe catches will
use every one of them. The observable result is a run that takes hours at a fraction of its
rate and then fails with whatever error happened to be last, long after the first one said
exactly what was wrong.

The two sizing properties below are why the allowance is relative rather than a fixed number.
A fixed budget is either too small for a large job — exhausted in the first second of a bad
patch, failing a run that was merely unlucky — or too large for a small one.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.resilience import RetryBudget

pytestmark = pytest.mark.unit


def test_a_large_job_gets_proportionally_more_retries():
    budget = RetryBudget(fraction=0.1, floor=0)
    budget.record_attempt(1000)
    assert budget.state().allowance == 100
    assert all(budget.try_consume() for _ in range(100))
    assert budget.try_consume() is False


def test_a_small_job_still_gets_the_floor():
    # Ten tasks at a 10% fraction would authorize one retry, so one flaky node fails the run.
    budget = RetryBudget(fraction=0.1, floor=16)
    budget.record_attempt(10)
    assert budget.state().allowance == 16


def test_the_allowance_grows_as_the_job_does():
    budget = RetryBudget(fraction=0.5, floor=0)
    budget.record_attempt(4)
    assert budget.try_consume() and budget.try_consume()
    assert budget.try_consume() is False
    budget.record_attempt(2)
    assert budget.try_consume() is True


def test_a_draw_is_all_or_nothing():
    # A partial draw would let a caller retry some of a batch and abandon the rest, which is
    # neither of the two outcomes it knows how to handle.
    budget = RetryBudget(fraction=0.0, floor=3)
    assert budget.try_consume(4) is False
    assert budget.state().spent == 0
    assert budget.try_consume(3) is True


def test_exhaustion_is_immediate_and_never_waits():
    # A resilience mechanism that blocks turns a broken fleet into a hung job.
    budget = RetryBudget(fraction=0.0, floor=1)
    assert budget.try_consume() is True
    assert budget.try_consume() is False
    assert budget.state().exhausted is True
    assert budget.state().remaining == 0


def test_exhaustion_is_announced_once_not_once_per_refusal():
    from batcher._internal import events

    seen: list[str] = []

    def _sink(event) -> None:
        if event.kind == events.RECOVERY:
            seen.append(event.fields.get("event", ""))

    budget = RetryBudget(fraction=0.0, floor=0, label="aggregate")
    unsubscribe = events.subscribe(_sink)
    try:
        for _ in range(50):
            budget.try_consume()
    finally:
        unsubscribe()
    assert seen.count("budget_exhausted") == 1


def test_reset_returns_the_budget_to_empty():
    budget = RetryBudget(fraction=0.0, floor=1)
    budget.try_consume()
    budget.reset()
    assert budget.state().spent == 0
    assert budget.try_consume() is True
