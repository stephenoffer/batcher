"""Late rows and retained state are counted, and reach the progress record.

The documentation used to say, as a known issue, that a late row vanished with "no side
output, no dead-letter, no late-row counter" — so a windowed total that came out short had
no evidence anywhere explaining why. These pin the counter end to end: a real query, a real
watermark, a row that arrives after its window closed, and the number showing up on
`recent_progress` and on the listener that saw the batch.
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

_SCHEMA = pa.schema([("ts", pa.timestamp("us")), ("v", pa.int64())])
_BASE = datetime.datetime(2024, 1, 1)


def _at(minutes: int) -> datetime.datetime:
    return _BASE + datetime.timedelta(minutes=minutes)


def _batch(*minutes: int, value: int = 1) -> pa.RecordBatch:
    return pa.record_batch(
        {"ts": [_at(m) for m in minutes], "v": [value] * len(minutes)}, schema=_SCHEMA
    )


def _windowed_query(feed, name: str):
    ds = bt.from_batches(feed, _SCHEMA, bounded=False).with_watermark("ts", "5 minutes")
    return (
        ds.group_by(w=bt.window(col("ts"), "10 minutes"))
        .agg(total=col("v").sum())
        .write.memory(name, trigger=bt.Trigger.available_now(), output_mode="append")
    )


@pytest.mark.integration
def test_a_row_that_arrives_after_its_window_closed_is_counted_as_late():
    def feed():
        yield _batch(0, 1)  # watermark -> -5m
        yield _batch(30)  # watermark -> 25m, closing the 0-10m window
        yield _batch(1, value=5)  # far below the watermark: dropped

    query = _windowed_query(feed, "late_counted")
    assert query.await_termination(timeout=30) is True

    per_batch = [p.num_late_rows for p in query.recent_progress]
    assert per_batch[-1] == 1, "the late row was dropped with no counter"
    assert sum(per_batch) == 1, "an on-time row was miscounted as late"


@pytest.mark.integration
def test_a_query_with_no_late_rows_reports_zero_throughout():
    def feed():
        yield _batch(0, 1)
        yield _batch(2, 3)

    query = _windowed_query(feed, "no_late")
    assert query.await_termination(timeout=30) is True
    assert all(p.num_late_rows == 0 for p in query.recent_progress)


@pytest.mark.integration
def test_state_metrics_report_retention_eviction_and_the_watermark():
    def feed():
        yield _batch(0, 1)
        yield _batch(30)  # closes and evicts the first window

    query = _windowed_query(feed, "state_metrics")
    assert query.await_termination(timeout=30) is True

    operators = [p.state_operators for p in query.recent_progress]
    assert all(len(ops) == 1 for ops in operators)
    assert all(ops[0].operator_name == "windowed_aggregate" for ops in operators)
    # The second batch's watermark closes the first window, so something is evicted.
    assert operators[-1][0].num_rows_removed >= 1
    assert operators[-1][0].watermark_micros is not None
    assert operators[-1][0].memory_used_bytes > 0
    # The query watermark is the minimum across operators, which with one is that one.
    assert query.recent_progress[-1].event_time_watermark_micros == (
        operators[-1][0].watermark_micros
    )


@pytest.mark.integration
def test_a_stateless_pipeline_reports_no_state_operators():
    def feed():
        yield _batch(0, 1)

    stream = bt.from_batches(feed, _SCHEMA, bounded=False)
    query = stream.filter(col("v") > 0).write.memory(
        "stateless_metrics", trigger=bt.Trigger.available_now()
    )
    assert query.await_termination(timeout=30) is True
    last = query.recent_progress[-1]
    assert last.state_operators == ()
    assert last.num_late_rows == 0
    assert last.event_time_watermark_micros is None


@pytest.mark.integration
def test_progress_names_its_query_and_describes_its_source_and_sink():
    def feed():
        yield _batch(0)

    stream = bt.from_batches(feed, _SCHEMA, bounded=False)
    query = stream.write.memory(
        "described", trigger=bt.Trigger.available_now(), query_name="described-query"
    )
    assert query.await_termination(timeout=30) is True

    last = query.recent_progress[-1]
    assert last.name == "described-query"
    assert len(last.sources) == 1
    assert last.sources[0].num_input_rows == 1
    assert last.sink is not None
    assert "Memory" in last.sink.description


@pytest.mark.integration
def test_a_listener_sees_the_same_batches_the_handle_records():
    seen: list[int] = []

    class _Watcher(bt.StreamingQueryListener):
        def on_query_progress(self, event):
            seen.append(event.progress.batch_id)

    watcher = _Watcher()
    bt.add_streaming_listener(watcher)
    try:

        def feed():
            yield _batch(0)
            yield _batch(1)

        stream = bt.from_batches(feed, _SCHEMA, bounded=False)
        query = stream.write.memory("listener_parity", trigger=bt.Trigger.available_now())
        assert query.await_termination(timeout=30) is True
        assert seen == [p.batch_id for p in query.recent_progress]
    finally:
        bt.remove_streaming_listener(watcher)


@pytest.mark.integration
def test_an_aggregate_reports_the_rows_it_is_retaining():
    def feed():
        yield pa.record_batch({"ts": [_at(0)], "v": [1]}, schema=_SCHEMA)
        yield pa.record_batch({"ts": [_at(1)], "v": [2]}, schema=_SCHEMA)

    stream = bt.from_batches(feed, _SCHEMA, bounded=False)
    query = (
        stream.group_by(k=col("v"))
        .agg(n=col("v").count())
        .write.memory("agg_state", trigger=bt.Trigger.available_now(), output_mode="complete")
    )
    assert query.await_termination(timeout=30) is True

    last = query.recent_progress[-1].state_operators
    assert len(last) == 1 and last[0].operator_name == "aggregate"
    assert last[0].num_rows_total == 2, "two distinct keys are retained"
    assert last[0].num_rows_removed == 0, "an unwatermarked aggregate never evicts"
