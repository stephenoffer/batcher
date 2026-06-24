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
    progress = q.recent_progress()
    assert progress and progress[-1].num_input_rows == 1
    q.stop()


@pytest.mark.integration
def test_await_termination_timeout_returns_false_then_true():
    q = _slow_stream(n=1000).write.memory("life_await", trigger=bt.Trigger.processing_time(0))
    assert q.await_termination(timeout=0.05) is False  # still running
    q.stop()
    assert q.await_termination() is True


@pytest.mark.integration
def test_processing_time_trigger_drains_bounded_stream():
    # A finite unbounded source drains under processing_time and the query ends.
    q = _slow_stream(n=5, delay=0.0).write.memory(
        "life_pt", trigger=bt.Trigger.processing_time("10 milliseconds")
    )
    q.await_termination()
    assert bt.read_memory("life_pt").count() == 5


@pytest.mark.integration
def test_update_output_mode_emits_running_result():
    seen: list[pa.Table] = []

    def sink(table: pa.Table, _batch_id: int) -> None:
        seen.append(table)

    def feed():
        yield pa.record_batch({"k": ["a", "b"], "v": [1, 2]}, schema=_SCHEMA)
        yield pa.record_batch({"k": ["a"], "v": [3]}, schema=_SCHEMA)

    q = (
        bt.from_batches(feed, _SCHEMA, bounded=False)
        .group_by("k")
        .agg(total=bt.col("v").sum())
        .write.for_each_batch(sink, trigger=bt.Trigger.available_now(), output_mode="update")
    )
    q.await_termination()
    last = seen[-1].to_pydict()
    final = dict(zip(last["k"], last["total"], strict=True))
    assert final == {"a": 4, "b": 2}


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
