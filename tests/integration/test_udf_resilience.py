"""`map_batches` transient-failure resilience: retry with backoff, restricted retries,
per-call timeout, and how they compose with the `max_errored_rows` budget.

Retry/timeout wrap the raw `fn` call *outside* the error-budget bisection, so a transient
failure is retried first and only a failure that survives every retry is charged against
the budget. These are the LLM-API / vector-DB / flaky-model resilience knobs.
"""

from __future__ import annotations

import time

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import batcher as bt

pytest.importorskip("batcher._native", reason="native engine not built")


def _add_one(batch: pa.RecordBatch) -> pa.RecordBatch:
    return batch.set_column(0, "x", pc.add(batch.column("x"), 1))


def test_retry_succeeds_after_transient_failures():
    """A `fn` that raises a retryable error N times, then succeeds, produces the right answer
    with `max_retries >= N` and no rows lost."""
    state = {"n": 0}

    def flaky(batch: pa.RecordBatch) -> pa.RecordBatch:
        state["n"] += 1
        if state["n"] < 3:
            raise ConnectionError("429 rate limited")
        return _add_one(batch)

    out = (
        bt.from_pydict({"x": [1, 2, 3]})
        .map_batches(flaky, max_retries=3, retry_backoff=0.0)
        .to_pydict()
    )
    assert out == {"x": [2, 3, 4]}
    assert state["n"] == 3  # failed twice, third attempt won


def test_retry_exhausted_propagates():
    """A `fn` that always raises fails once the retry budget is spent (no silent drop)."""

    def always(batch: pa.RecordBatch) -> pa.RecordBatch:
        raise ConnectionError("service down")

    with pytest.raises(Exception, match="service down"):
        bt.from_pydict({"x": [1]}).map_batches(always, max_retries=2, retry_backoff=0.0).collect()


def test_retry_on_restricts_to_listed_types():
    """`retry_on` limits retries to the given exception types; anything else fails immediately."""
    state = {"n": 0}

    def bug(batch: pa.RecordBatch) -> pa.RecordBatch:
        state["n"] += 1
        raise TypeError("a real bug, not a transient")

    with pytest.raises(TypeError, match="real bug"):
        bt.from_pydict({"x": [1]}).map_batches(
            bug, max_retries=5, retry_backoff=0.0, retry_on=ConnectionError
        ).collect()
    assert state["n"] == 1  # not retried — the type was not in retry_on


def test_retry_on_accepts_a_tuple_of_types():
    state = {"n": 0}

    def flaky(batch: pa.RecordBatch) -> pa.RecordBatch:
        state["n"] += 1
        if state["n"] < 2:
            raise TimeoutError("slow")
        return _add_one(batch)

    out = (
        bt.from_pydict({"x": [5]})
        .map_batches(
            flaky, max_retries=2, retry_backoff=0.0, retry_on=(ConnectionError, TimeoutError)
        )
        .to_pydict()
    )
    assert out == {"x": [6]}
    assert state["n"] == 2


def test_timeout_raises_on_a_hung_call():
    """A call exceeding `timeout` raises `TimeoutError` instead of hanging the query."""

    def slow(batch: pa.RecordBatch) -> pa.RecordBatch:
        time.sleep(5)
        return batch

    t0 = time.perf_counter()
    with pytest.raises(Exception, match="timeout"):
        bt.from_pydict({"x": [1]}).map_batches(slow, timeout=0.2).collect()
    # The query returned promptly rather than waiting out the 5s sleep.
    assert time.perf_counter() - t0 < 3.0


def test_timeout_is_retried_then_recovers():
    """A timed-out call is retried (a slow first attempt, a fast second) and recovers."""
    state = {"n": 0}

    def sometimes_slow(batch: pa.RecordBatch) -> pa.RecordBatch:
        state["n"] += 1
        if state["n"] == 1:
            time.sleep(5)
        return _add_one(batch)

    out = (
        bt.from_pydict({"x": [1, 2]})
        .map_batches(sometimes_slow, timeout=0.3, max_retries=2, retry_backoff=0.0)
        .to_pydict()
    )
    assert out == {"x": [2, 3]}
    assert state["n"] == 2  # first attempt timed out, second succeeded


def test_retry_then_error_budget_composes():
    """Retry (transient) and `max_errored_rows` (permanent) are independent layers: a batch
    retried to exhaustion falls through to the budget and its rows are isolated/dropped."""
    seen: set[int] = set()

    def flaky_per_value(batch: pa.RecordBatch) -> pa.RecordBatch:
        # Every call for a batch containing the value 2 raises — a *permanent* failure for
        # that row, so retries never help and it must be charged to the error budget.
        vals = batch.column("x").to_pylist()
        for v in vals:
            seen.add(v)
        if 2 in vals:
            raise ValueError("row 2 is always bad")
        return _add_one(batch)

    out = (
        bt.from_pydict({"x": [1, 2, 3]})
        .map_batches(
            flaky_per_value,
            max_retries=2,
            retry_backoff=0.0,
            max_errored_rows=10,
            output_columns=["x"],
        )
        .to_pydict()
    )
    # Row 2 dropped after retries were exhausted; 1 and 3 survive and are incremented.
    assert sorted(out["x"]) == [2, 4]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_retries": -1},
        {"timeout": -0.5},
        {"retry_backoff": -1.0},
        {"retry_on": 42},
        {"retry_on": (ConnectionError, "not-a-type")},
    ],
)
def test_invalid_retry_options_raise_planerror(kwargs):
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError):
        bt.from_pydict({"x": [1]}).map_batches(_add_one, **kwargs)
