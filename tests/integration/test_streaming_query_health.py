"""A streaming query's lifetime totals, listener health, and a bounded stop.

Three gaps a monitoring page used to paper over with advice:

- `recent_progress` keeps only the last `streaming.progress_history` batches, so a sum of
  late rows over it undercounts any query that has run longer than that. `status` now
  carries lifetime totals that do not roll off.
- A listener that raises is logged and skipped so it cannot fail the query, which also left
  a broken monitoring callback invisible to anything watching the query. `status` counts
  those failures.
- `stop()` waited unboundedly for the micro-batch boundary. `stop(timeout=...)` gives up and
  says so, while still letting the batch finish rather than cutting it off.
"""

from __future__ import annotations

import datetime
import threading

import pyarrow as pa
import pytest

import batcher as bt
from batcher import Config, StreamingConfig, col

pytestmark = pytest.mark.integration

_SCHEMA = pa.schema([("ts", pa.timestamp("us")), ("v", pa.int64())])
_BASE = datetime.datetime(2024, 1, 1)


@pytest.fixture(autouse=True)
def _clean_listeners():
    before = bt.streaming_listeners()
    yield
    for listener in bt.streaming_listeners():
        bt.remove_streaming_listener(listener)
    for listener in before:
        bt.add_streaming_listener(listener)


def _batch(*minutes: int, value: int = 1) -> pa.RecordBatch:
    ts = [_BASE + datetime.timedelta(minutes=m) for m in minutes]
    return pa.record_batch({"ts": ts, "v": [value] * len(minutes)}, schema=_SCHEMA)


def test_lifetime_totals_outlive_the_bounded_progress_window():
    cfg = Config().replace(streaming=StreamingConfig(progress_history=2))
    with bt.config_context(cfg):
        stream = bt.read.rate(rows_per_second=5, num_rows=23, pace=False)
        query = stream.write.memory("health_totals", trigger=bt.Trigger.available_now())
        assert query.await_termination(timeout=30) is True

    window = query.recent_progress
    status = query.status
    assert len(window) == 2
    assert status.batches_processed == 5
    # The window saw only its last two batches; the totals saw every row.
    assert sum(p.num_input_rows for p in window) < 23
    assert status.total_input_rows == 23
    assert status.total_output_rows == 23


def test_late_rows_total_counts_batches_that_left_the_window():
    def feed():
        yield _batch(0, 1)  # watermark -> -5m
        yield _batch(30)  # watermark -> 25m, closing the 0-10m window
        yield _batch(1, value=5)  # far below the watermark: dropped
        yield _batch(31)
        yield _batch(32)

    cfg = Config().replace(streaming=StreamingConfig(progress_history=2))
    with bt.config_context(cfg):
        ds = bt.from_batches(feed, _SCHEMA, bounded=False).with_watermark("ts", "5 minutes")
        query = (
            ds.group_by(w=bt.window(col("ts"), "10 minutes"))
            .agg(total=col("v").sum())
            .write.memory("health_late", trigger=bt.Trigger.available_now(), output_mode="append")
        )
        assert query.await_termination(timeout=30) is True

    # The batch that dropped the late row has rolled out of the two-batch window.
    assert sum(p.num_late_rows for p in query.recent_progress) == 0
    assert query.status.total_late_rows == 1


def test_a_failing_listener_is_counted_on_the_query_status():
    class Broken(bt.StreamingQueryListener):
        def on_query_progress(self, event):
            raise RuntimeError("metrics endpoint down")

    bt.add_streaming_listener(Broken())
    stream = bt.read.rate(rows_per_second=5, num_rows=10, pace=False)
    query = stream.write.memory("health_listener", trigger=bt.Trigger.available_now())
    assert query.await_termination(timeout=30) is True

    status = query.status
    assert query.exception() is None, "a listener must never fail the query"
    assert status.batches_processed == 2
    assert status.listener_failures == 2


def test_a_healthy_listener_reports_no_failures():
    class Quiet(bt.StreamingQueryListener):
        pass

    bt.add_streaming_listener(Quiet())
    stream = bt.read.rate(rows_per_second=5, num_rows=10, pace=False)
    query = stream.write.memory("health_quiet", trigger=bt.Trigger.available_now())
    assert query.await_termination(timeout=30) is True
    assert query.status.listener_failures == 0


def test_stop_with_a_timeout_returns_false_while_a_batch_is_held_then_stops():
    entered = threading.Event()
    release = threading.Event()

    class Hold(bt.StreamingQueryListener):
        # Listeners run on the loop thread, so this holds the loop between batches exactly
        # the way a long micro-batch would.
        def on_query_progress(self, event):
            entered.set()
            release.wait(timeout=30)

    bt.add_streaming_listener(Hold())
    stream = bt.read.rate(rows_per_second=5, pace=False)
    query = stream.write.memory(
        "health_stop", trigger=bt.Trigger.processing_time("100 milliseconds")
    )
    try:
        assert entered.wait(timeout=30), "the first micro-batch never completed"
        assert query.stop(timeout=0.2) is False
        assert query.is_active, "a timed-out stop must not report the query stopped"
    finally:
        release.set()
    assert query.await_termination(timeout=30) is True
    assert query.stop(timeout=5) is True
    assert not query.is_active
