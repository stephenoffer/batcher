"""StreamingQuery lifecycle + the remaining sink/trigger/output-mode surface.

Covers the parts of the unified `ds.write` streaming API beyond the happy path:
`stop()` mid-stream, `is_active`/`status`/`recent_progress` while running, the
`bt.streams()` registry, `await_termination(timeout)`, the `processing_time` trigger,
the `update` output mode, and the console / `for_each` row sinks.
"""

from __future__ import annotations

import time

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

_SCHEMA = pa.schema([("k", pa.string()), ("v", pa.int64())])


def _slow_stream(n: int = 50, delay: float = 0.02):
    """An unbounded source that yields `n` one-row batches, pacing each by `delay`."""

    def feed():
        for i in range(n):
            time.sleep(delay)
            yield pa.record_batch({"k": ["a"], "v": [i]}, schema=_SCHEMA)

    return bt.from_batches(feed, _SCHEMA, bounded=False)


@pytest.mark.integration
def test_stop_halts_mid_stream():
    q = _slow_stream(n=1000).write.memory("life_stop", trigger=bt.Trigger.processing_time(0))
    time.sleep(0.1)
    assert q.is_active is True
    q.stop()
    assert q.is_active is False
    processed = bt.read_memory("life_stop").count()
    assert 0 < processed < 1000  # stopped early, not drained


@pytest.mark.integration
def test_streams_registry_lists_active_query():
    q = _slow_stream(n=1000).write.memory("life_reg", trigger=bt.Trigger.processing_time(0))
    time.sleep(0.05)
    names = [s.name for s in bt.streams()]
    assert q.name in names
    q.stop()
    assert q.name not in [s.name for s in bt.streams()]


@pytest.mark.integration
def test_status_and_progress_while_running():
    q = _slow_stream(n=1000).write.memory("life_status", trigger=bt.Trigger.processing_time(0))
    time.sleep(0.1)
    status = q.status
    assert status.is_active is True
    assert status.batches_processed >= 1
    progress = q.recent_progress
    assert progress and progress[-1].num_input_rows == 1
    q.stop()


@pytest.mark.integration
def test_await_termination_timeout_returns_false_then_true():
    q = _slow_stream(n=1000).write.memory("life_await", trigger=bt.Trigger.processing_time(0))
    assert q.await_termination(timeout=0.05) is False  # still running
    q.stop()
    assert q.await_termination() is True


@pytest.mark.integration
def test_exception_is_none_on_clean_stop():
    q = _slow_stream(n=1000).write.memory("life_exc", trigger=bt.Trigger.processing_time(0))
    time.sleep(0.05)
    assert q.exception() is None  # healthy while running
    q.stop()
    assert q.exception() is None  # clean stop, no error


@pytest.mark.integration
def test_query_is_a_context_manager():
    with _slow_stream(n=1000).write.memory("life_cm", trigger=bt.Trigger.processing_time(0)) as q:
        assert q.is_active is True
        assert "active" in repr(q)
    # Leaving the `with` block stops the query.
    assert q.is_active is False
    assert "stopped" in repr(q)


@pytest.mark.integration
def test_await_any_termination():
    # No active queries → returns immediately.
    assert bt.await_any_termination(timeout=0.0) is True
    q = _slow_stream(n=1000).write.memory("life_any", trigger=bt.Trigger.processing_time(0))
    # A running query → the poll times out.
    assert bt.await_any_termination(timeout=0.05) is False
    q.stop()
    # After it stops → returns True.
    assert bt.await_any_termination(timeout=1.0) is True


@pytest.mark.integration
def test_processing_time_trigger_drains_bounded_stream():
    # A finite unbounded source drains under processing_time and the query ends.
    q = _slow_stream(n=5, delay=0.0).write.memory(
        "life_pt", trigger=bt.Trigger.processing_time("10 milliseconds")
    )
    q.await_termination()
    assert bt.read_memory("life_pt").count() == 5


def _batches_of(mode: str) -> list[dict]:
    """Run the same two-micro-batch aggregation under `mode` and record what each emitted."""
    seen: list[pa.Table] = []

    def feed():
        yield pa.record_batch({"k": ["a", "b"], "v": [1, 2]}, schema=_SCHEMA)
        yield pa.record_batch({"k": ["a"], "v": [3]}, schema=_SCHEMA)

    query = (
        bt.from_batches(feed, _SCHEMA, bounded=False)
        .group_by("k")
        .agg(total=bt.col("v").sum())
        .write.for_each_batch(
            lambda table, _id: seen.append(table),
            trigger=bt.Trigger.available_now(),
            output_mode=mode,
        )
    )
    query.await_termination()
    return [
        dict(zip(t.to_pydict()["k"], t.to_pydict()["total"], strict=True))
        for t in seen
        if t.num_rows
    ]


@pytest.mark.integration
def test_update_output_mode_emits_only_the_groups_that_changed():
    """Spark's rule, and the only thing that distinguishes `update` from `complete`: `b`
    got no row in the second micro-batch, so its unchanged total is not re-sent. Emitting
    it anyway is `complete` wearing `update`'s name, and on a wide key space it is exactly
    the traffic the mode exists to avoid."""
    assert _batches_of("update") == [{"a": 1, "b": 2}, {"a": 4}]


@pytest.mark.integration
def test_complete_output_mode_still_emits_the_whole_result():
    """The contrast that gives the test above its meaning."""
    assert _batches_of("complete") == [{"a": 1, "b": 2}, {"a": 4, "b": 2}]


@pytest.mark.integration
def test_console_sink(capsys):
    def feed():
        yield pa.record_batch({"k": ["a"], "v": [1]}, schema=_SCHEMA)

    q = bt.from_batches(feed, _SCHEMA, bounded=False).write.console(
        trigger=bt.Trigger.available_now()
    )
    q.await_termination()
    assert "Batch:" in capsys.readouterr().out


@pytest.mark.integration
def test_for_each_row_sink():
    rows: list[dict] = []

    def feed():
        yield pa.record_batch({"k": ["a", "b"], "v": [1, 2]}, schema=_SCHEMA)

    q = bt.from_batches(feed, _SCHEMA, bounded=False).write.for_each(
        rows.append, trigger=bt.Trigger.available_now()
    )
    q.await_termination()
    assert sorted(r["v"] for r in rows) == [1, 2]


@pytest.mark.integration
def test_read_memory_missing_raises():
    with pytest.raises(PlanError, match="no in-memory streaming sink"):
        bt.read_memory("never_written_sink")


@pytest.mark.integration
def test_a_group_whose_value_is_unchanged_is_not_re_sent_even_when_it_gets_rows():
    """ "Changed" is about the value, not about whether a row arrived. Adding zero to a
    sum touches the group and changes nothing, so nothing should go downstream."""
    seen: list[pa.Table] = []

    def feed():
        yield pa.record_batch({"k": ["a"], "v": [5]}, schema=_SCHEMA)
        yield pa.record_batch({"k": ["a"], "v": [0]}, schema=_SCHEMA)

    query = (
        bt.from_batches(feed, _SCHEMA, bounded=False)
        .group_by("k")
        .agg(total=bt.col("v").sum())
        .write.for_each_batch(
            lambda table, _id: seen.append(table),
            trigger=bt.Trigger.available_now(),
            output_mode="update",
        )
    )
    query.await_termination()
    # One call, not two: the second micro-batch produced no changed row, so the sink is
    # not invoked at all rather than handed an empty table.
    assert [t.num_rows for t in seen] == [1]


@pytest.mark.integration
def test_update_mode_on_a_windowed_aggregate_emits_only_the_window_that_moved():
    """The group key is an expression here, which is why the diff is over output *values*
    rather than over "which keys did this batch carry" -- the batch carries `ts`, not the
    window the row lands in."""
    import datetime as dt

    base = dt.datetime(2024, 1, 1)
    schema = pa.schema([("ts", pa.timestamp("us")), ("v", pa.int64())])
    seen: list[pa.Table] = []

    def feed():
        yield pa.record_batch({"ts": [base], "v": [1]}, schema=schema)
        yield pa.record_batch({"ts": [base + dt.timedelta(hours=3)], "v": [2]}, schema=schema)

    query = (
        bt.from_batches(feed, schema, bounded=False)
        .with_watermark("ts", "10m")
        .group_by(w=bt.window(bt.col("ts"), "1h"))
        .agg(total=bt.col("v").sum())
        .write.for_each_batch(
            lambda table, _id: seen.append(table),
            trigger=bt.Trigger.available_now(),
            output_mode="update",
        )
    )
    query.await_termination()
    emitted = [t.to_pydict()["total"] for t in seen if t.num_rows]
    assert emitted == [[1], [2]]
