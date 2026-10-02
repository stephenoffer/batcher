"""An idle input must not stall a union of streams or a stream-stream join.

Both used to pull from their inputs in turn on one thread, so an input parked in a read
that does not return (a quiet topic) held every other input behind it. These pin the
readiness-based driver: one input blocks forever, and the other's rows still come out.
"""

from __future__ import annotations

import datetime as dt
import threading

import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.terminal.stream.multiplex import multiplex

pytestmark = pytest.mark.integration

_WAIT = 20.0


@pytest.fixture
def release():
    """Unblocks the idle input at teardown so its reader thread can end."""
    event = threading.Event()
    yield event
    event.set()


def _collect_until(stream, want: int, timeout: float = _WAIT) -> list:
    """Pull from `stream` on a helper thread until `want` rows arrive or time runs out."""
    got: list = []
    done = threading.Event()

    def run():
        for batch in stream:
            got.extend(batch.to_pylist())
            if len(got) >= want:
                break
        done.set()

    threading.Thread(target=run, daemon=True).start()
    done.wait(timeout)
    return got


def test_a_union_keeps_emitting_while_one_branch_is_idle(release):
    schema = pa.schema([("v", pa.int64())])

    def busy():
        for i in range(5):
            yield pa.record_batch({"v": [i]}, schema=schema)

    def idle():
        yield pa.record_batch({"v": [100]}, schema=schema)
        release.wait()  # a read that does not return while the test runs

    both = bt.from_batches(busy, schema, bounded=False).union(
        bt.from_batches(idle, schema, bounded=False)
    )
    rows = _collect_until(both.iter_batches(), want=6)
    assert sorted(r["v"] for r in rows) == [0, 1, 2, 3, 4, 100]


def test_a_stream_join_keeps_joining_while_one_side_is_idle(release):
    base = dt.datetime(2024, 1, 1)
    left_schema = pa.schema([("ad", pa.string()), ("shown", pa.timestamp("us"))])
    right_schema = pa.schema([("ad", pa.string()), ("clicked", pa.timestamp("us"))])

    def impressions():
        for i in range(4):
            yield pa.record_batch(
                {"ad": ["a"], "shown": [base + dt.timedelta(seconds=i)]}, schema=left_schema
            )

    def clicks():
        yield pa.record_batch(
            {"ad": ["a"], "clicked": [base + dt.timedelta(seconds=2)]}, schema=right_schema
        )
        release.wait()

    joined = bt.from_batches(impressions, left_schema, bounded=False).join_stream(
        bt.from_batches(clicks, right_schema, bounded=False),
        on="ad",
        left_time="shown",
        right_time="clicked",
        within="1h",
    )
    # Every impression matches the one click; round-robin reads parked on the idle side
    # after the second impression and never joined the last two.
    rows = _collect_until(joined.iter_batches(), want=4)
    assert len(rows) == 4


def test_multiplex_reraises_an_input_failure():
    def broken():
        yield 1
        raise ValueError("input failed")

    with pytest.raises(ValueError, match="input failed"):
        list(multiplex([iter([1, 2]), broken()]))


def test_multiplex_keeps_each_inputs_order_and_ends_when_all_do():
    out = list(multiplex([iter(range(50)), iter(range(100, 130))]))
    first = [v for i, v in out if i == 0]
    second = [v for i, v in out if i == 1]
    assert first == list(range(50))
    assert second == list(range(100, 130))


def test_closing_the_consumer_closes_every_input():
    closed = threading.Event()

    def endless():
        try:
            while True:
                yield 0
        finally:
            closed.set()

    stream = multiplex([endless()])
    next(stream)
    stream.close()
    assert closed.wait(_WAIT), "the reader thread never closed its input"


def _impressions_and_clicks(n_impressions: int = 6):
    base = dt.datetime(2024, 1, 1)
    left_schema = pa.schema([("ad", pa.string()), ("shown", pa.timestamp("us"))])
    right_schema = pa.schema([("ad", pa.string()), ("clicked", pa.timestamp("us"))])

    def impressions():
        for i in range(n_impressions):
            ad = "a" if i == 0 else f"x{i}"
            yield pa.record_batch(
                {"ad": [ad], "shown": [base + dt.timedelta(seconds=i)]}, schema=left_schema
            )

    def clicks():
        yield pa.record_batch(
            {"ad": ["a"], "clicked": [base + dt.timedelta(seconds=1)]}, schema=right_schema
        )

    return (
        bt.from_batches(impressions, left_schema, bounded=False),
        bt.from_batches(clicks, right_schema, bounded=False),
    )


def test_a_driver_stream_reports_rows_read_not_rows_emitted():
    """F391: one match out of seven source rows used to be reported as one row read.

    A driver micro-batch is one *output* batch, so reads are attributed to the batch they
    precede: the rows read after the last match belong to no micro-batch and are not
    counted. What is pinned is that the count is of reads, and so exceeds the output.
    """
    left, right = _impressions_and_clicks()
    joined = left.join_stream(right, on="ad", left_time="shown", right_time="clicked", within="1h")
    query = joined.write.memory("driver_reads", trigger=bt.Trigger.available_now())
    assert query.await_termination(timeout=60) is True
    progress = query.recent_progress
    assert sum(p.num_output_rows for p in progress) == 1
    read = sum(p.num_input_rows for p in progress)
    assert 1 < read <= 7


def test_a_stream_join_reports_its_buffered_state():
    """F392: the join's two buffers are state, and the progress record now says so."""
    left, right = _impressions_and_clicks()
    joined = left.join_stream(right, on="ad", left_time="shown", right_time="clicked", within="1h")
    query = joined.write.memory("driver_state", trigger=bt.Trigger.available_now())
    assert query.await_termination(timeout=60) is True
    ops = [op for p in query.recent_progress for op in p.state_operators]
    assert ops, "no state operator was reported for a buffering join"
    assert {op.operator_name for op in ops} == {"stream_join"}
    assert max(op.num_rows_total for op in ops) > 0
    assert max(op.memory_used_bytes for op in ops) > 0


def test_a_watermark_dedup_and_a_session_window_report_their_state():
    base = dt.datetime(2024, 1, 1)
    schema = pa.schema([("id", pa.int64()), ("ts", pa.timestamp("us"))])

    def events():
        for i in range(4):
            yield pa.record_batch(
                {"id": [i % 2], "ts": [base + dt.timedelta(seconds=i)]}, schema=schema
            )

    stream = bt.from_batches(events, schema, bounded=False)
    shapes = {
        "dedup": stream.drop_duplicates_within_watermark(["id"], event_time="ts", lateness="1h"),
        "session_window": stream.session_window("ts", "1h", n=bt.col("id").count()),
    }
    for name, ds in shapes.items():
        query = ds.write.memory(f"state_{name}", trigger=bt.Trigger.available_now())
        assert query.await_termination(timeout=60) is True
        ops = [op for p in query.recent_progress for op in p.state_operators]
        assert query.recent_progress, f"{name} emitted no micro-batch"
        assert {op.operator_name for op in ops} == {name}, (name, ops)
