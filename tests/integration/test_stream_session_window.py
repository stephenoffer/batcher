"""Session windows over a stream, where the end of a window is not known in advance.

A tumbling window's bounds exist before any row arrives, so a streaming aggregation can
close one the moment the watermark passes its end. A session's do not: every event can
extend the session it lands in, and an event between two sessions merges them. So the
operator has to hold rows until the watermark guarantees nothing can change them, and
these tests are about that guarantee -- when a session is emitted, what happens to a row
that arrives after it was, and what bounds the buffer when the guarantee never comes.

The result-is-correct half is proved against DuckDB in
`tests/differential/test_diff_session_window.py`, at three different micro-batch
granularities. What is here is what an oracle over the whole input cannot see.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher._internal.errors import ResourceError
from batcher.config import Config, MemoryConfig, config_context

_BASE = dt.datetime(2024, 1, 1)
_MIN = dt.timedelta(minutes=1)
_SCHEMA = pa.schema([("k", pa.string()), ("ts", pa.timestamp("us")), ("v", pa.int64())])


def _stream(batches: list[list[tuple[str, int, int]]]):
    """An unbounded source; each inner list is one micro-batch of ``(key, minute, v)``."""

    def feed():
        for rows in batches:
            yield pa.record_batch(
                {
                    "k": [k for k, _, _ in rows],
                    "ts": [_BASE + m * _MIN for _, m, _ in rows],
                    "v": [v for _, _, v in rows],
                },
                schema=_SCHEMA,
            )

    return bt.from_batches(feed, _SCHEMA, bounded=False)


def _sessions(dataset) -> list[tuple]:
    out = []
    for batch in dataset.iter_batches():
        for row in batch.to_pylist():
            out.append((row["k"], row["session_start"], row["session_end"], row["total"]))
    return sorted(out)


def _at(minutes: int) -> dt.datetime:
    return _BASE + minutes * _MIN


@pytest.mark.integration
def test_a_session_is_emitted_once_the_gap_has_passed_with_nothing_arriving():
    stream = _stream([[("a", 0, 1), ("a", 2, 1)], [("a", 30, 1)]])
    got = _sessions(stream.session_window("ts", "5m", partition_by=["k"], total=col("v").sum()))
    assert got == [("a", _at(0), _at(2), 2), ("a", _at(30), _at(30), 1)]


@pytest.mark.integration
def test_a_session_spanning_micro_batches_is_one_session():
    """The batch boundary is a scheduling artifact and must not be a session boundary."""
    stream = _stream([[("a", 0, 1)], [("a", 2, 1)], [("a", 4, 1)], [("a", 60, 1)]])
    got = _sessions(stream.session_window("ts", "5m", partition_by=["k"], total=col("v").sum()))
    assert got == [("a", _at(0), _at(4), 3), ("a", _at(60), _at(60), 1)]


@pytest.mark.integration
def test_an_event_arriving_between_two_open_sessions_merges_them():
    """The property that makes a session window stateful rather than merely windowed: the
    bridging event turns two sessions into one. It is only possible while both are still
    open, which is what the allowed lateness buys -- with none, the watermark would have
    closed the first session before the bridge arrived, and the bridge would be late."""
    stream = _stream([[("a", 0, 1), ("a", 8, 1)], [("a", 4, 1)], [("a", 60, 1)]]).with_watermark(
        "ts", "10m"
    )
    got = _sessions(stream.session_window("ts", "5m", partition_by=["k"], total=col("v").sum()))
    assert got == [("a", _at(0), _at(8), 3), ("a", _at(60), _at(60), 1)]


@pytest.mark.integration
def test_without_lateness_the_bridging_event_is_simply_late():
    """The same three batches with no allowed lateness. The watermark closed the first
    session at the end of batch one, so the bridge cannot reopen it -- and the operator
    drops it rather than emitting a second, contradictory row for a session already sent
    downstream. Spark behaves identically; the remedy is `with_watermark`, above."""
    stream = _stream([[("a", 0, 1), ("a", 8, 1)], [("a", 4, 1)], [("a", 60, 1)]])
    got = _sessions(stream.session_window("ts", "5m", partition_by=["k"], total=col("v").sum()))
    assert got == [("a", _at(0), _at(0), 1), ("a", _at(8), _at(8), 1), ("a", _at(60), _at(60), 1)]


@pytest.mark.integration
def test_the_last_session_of_a_stream_is_emitted_when_the_stream_ends():
    """No watermark ever closes it -- nothing arrives to advance event time past its gap.
    Dropping it would lose the newest session of every key, silently."""
    stream = _stream([[("a", 0, 1)], [("a", 1, 1)]])
    got = _sessions(stream.session_window("ts", "5m", partition_by=["k"], total=col("v").sum()))
    assert got == [("a", _at(0), _at(1), 2)]


@pytest.mark.integration
def test_a_row_older_than_the_watermark_is_dropped_rather_than_reopening_a_session():
    """The straggler's session was already emitted. Letting it in would either emit a
    second, contradictory row for the same session or silently correct one already
    written to a sink -- so it is dropped, which is the promise the watermark made."""
    stream = _stream([[("a", 0, 1)], [("a", 100, 1)], [("a", 1, 999)]])
    got = _sessions(stream.session_window("ts", "5m", partition_by=["k"], total=col("v").sum()))
    assert got == [("a", _at(0), _at(0), 1), ("a", _at(100), _at(100), 1)]


@pytest.mark.integration
def test_allowed_lateness_keeps_the_session_open_long_enough_to_accept_the_straggler():
    """The same straggler, with `with_watermark` buying it room. This is the knob, and it
    is the only difference between the two tests."""
    stream = _stream([[("a", 0, 1)], [("a", 100, 1)], [("a", 1, 999)]]).with_watermark("ts", "3h")
    got = _sessions(stream.session_window("ts", "5m", partition_by=["k"], total=col("v").sum()))
    assert got == [("a", _at(0), _at(1), 1000), ("a", _at(100), _at(100), 1)]


@pytest.mark.integration
def test_keys_are_sessionized_independently():
    """One key going quiet must not close another key's session, and a busy key must not
    hold a quiet one's open."""
    stream = _stream([[("a", 0, 1), ("b", 0, 10)], [("a", 3, 1)], [("a", 60, 1), ("b", 60, 10)]])
    got = _sessions(stream.session_window("ts", "5m", partition_by=["k"], total=col("v").sum()))
    assert got == [
        ("a", _at(0), _at(3), 2),
        ("a", _at(60), _at(60), 1),
        ("b", _at(0), _at(0), 10),
        ("b", _at(60), _at(60), 10),
    ]


@pytest.mark.integration
def test_without_partition_by_the_whole_stream_is_one_session_chain():
    stream = _stream([[("a", 0, 1), ("b", 2, 1)], [("a", 60, 1)]])
    rows = []
    for batch in stream.session_window("ts", "5m", total=col("v").sum()).iter_batches():
        rows.extend(batch.to_pylist())
    got = sorted((r["session_start"], r["session_end"], r["total"]) for r in rows)
    assert got == [(_at(0), _at(2), 2), (_at(60), _at(60), 1)]


@pytest.mark.integration
def test_several_aggregates_per_session():
    stream = _stream([[("a", 0, 3), ("a", 1, 5)], [("a", 60, 1)]])
    rows = []
    plan = stream.session_window(
        "ts", "5m", partition_by=["k"], total=col("v").sum(), n=col("v").count(), top=col("v").max()
    )
    for batch in plan.iter_batches():
        rows.extend(batch.to_pylist())
    first = next(r for r in rows if r["session_start"] == _at(0))
    assert (first["total"], first["n"], first["top"]) == (8, 2, 5)


@pytest.mark.integration
def test_a_sessionized_stream_writes_to_a_sink():
    stream = _stream([[("a", 0, 1), ("a", 2, 1)], [("a", 60, 1)]])
    query = stream.session_window(
        "ts", "5m", partition_by=["k"], total=col("v").sum()
    ).write.memory("session_sink", trigger=bt.Trigger.available_now())
    assert query.await_termination(timeout=60) is True
    written = bt.read_memory("session_sink").to_pydict()
    assert sorted(written["total"]) == [1, 2]


@pytest.mark.integration
def test_a_stalled_clock_raises_instead_of_growing_the_buffer_forever():
    """A session closes on event time, so a source whose event time stops advancing never
    closes one. The remedy is to look at the stalled source, and a named error says so
    where an OOM three hours later does not."""
    batches = [[(f"k{i}", 0, 1) for i in range(2000)] for _ in range(20)]
    tight = Config().replace(memory=MemoryConfig(streaming_state_max_bytes=64 << 10))
    with config_context(tight), pytest.raises(ResourceError, match="never close"):
        _sessions(
            _stream(batches).session_window("ts", "5m", partition_by=["k"], total=col("v").sum())
        )


@pytest.mark.integration
def test_a_bounded_source_still_takes_the_composed_plan():
    """Nothing to wait for, so there is nothing to buffer -- and the streaming node must
    not appear where it would only add latency."""
    from batcher.plan.logical import StreamingSessionWindow

    bounded = bt.from_pydict({"k": ["a"], "ts": [_at(0)], "v": [1]})
    plan = bounded.session_window("ts", "5m", partition_by=["k"], total=col("v").sum())._plan
    assert not isinstance(plan, StreamingSessionWindow)


@pytest.mark.integration
def test_the_streaming_node_refuses_to_be_lowered_to_the_ir():
    """It is executed by the driver. A lowering that quietly produced *something* would be
    a second, wrong definition of what a session is."""
    from batcher.plan.logical import StreamingSessionWindow

    plan = (
        _stream([[("a", 0, 1)]])
        .session_window("ts", "5m", partition_by=["k"], total=col("v").sum())
        ._plan
    )
    assert isinstance(plan, StreamingSessionWindow)
    with pytest.raises(NotImplementedError, match="streaming driver"):
        plan.to_ir()
