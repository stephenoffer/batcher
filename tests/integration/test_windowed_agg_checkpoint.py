"""A windowed streaming aggregation must checkpoint its state — or restart loses windows.

The micro-batch loop commits the source offsets it consumed on **every** epoch, but it snapshots
running state only for a processor that reports itself stateful — and `StreamingRunner.has_state`
duck-types on the presence of `snapshot_state`.

`WindowedAggregateProcessor` defined neither `snapshot_state` nor `restore_state`, so it reported
*stateless*. Offsets advanced; the open windows and the watermark did not. A crash mid-stream
therefore resumed **past** the consumed data with all in-flight windows gone — silently never
emitted, and unrecoverable, because the rows that fed them had been marked consumed. That is data
loss in exactly the query shape the watermark machinery exists to serve, and it shipped: no test
restarted a *windowed* aggregation (`test_streaming_checkpoint.py` restarts only stateless and
plain-aggregate queries).

Both halves of the state matter. Restoring the partials without the watermark would be worse than
restoring nothing: the engine would hold windows it could never close, and would re-admit as
on-time the rows the old watermark had already ruled late.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.integration

_BASE = dt.datetime(2024, 1, 1)
_SCHEMA = pa.schema([("ts", pa.timestamp("us")), ("v", pa.int64())])


def _rb(rows: list[dict]) -> pa.RecordBatch:
    return pa.RecordBatch.from_pylist(rows, schema=_SCHEMA)


def _at(minute: int, v: int) -> dict:
    return {"ts": _BASE + dt.timedelta(minutes=minute), "v": v}


def _windowed_plan():
    """A watermarked, hourly-windowed sum — the append-mode shape."""

    def batches():
        yield _rb([_at(0, 1)])

    ds = bt.from_batches(batches, _SCHEMA, bounded=False)
    return (
        ds.with_watermark("ts", "5m")
        .group_by(w=bt.window(col("ts"), "1h"))
        .agg(total=col("v").sum())
        ._plan
    )


def _processor():
    from batcher.core.streaming_query import make_processor

    return make_processor(_windowed_plan(), "append", None)


def test_a_windowed_aggregation_reports_itself_stateful():
    """The duck-typed check the micro-batch loop actually makes before snapshotting."""
    from batcher.core.streaming_query import WindowedAggregateProcessor

    proc = _processor()
    assert isinstance(proc, WindowedAggregateProcessor)
    assert getattr(proc, "snapshot_state", None) is not None, (
        "has_state() duck-types on snapshot_state; without it the loop commits offsets but "
        "never writes state, and a crash loses every open window"
    )
    assert getattr(proc, "restore_state", None) is not None


def test_state_and_watermark_survive_a_restore():
    proc = _processor()
    # Two hours of events: the 00:00 window closes as event time reaches 01:05; 01:00 stays open.
    proc.process(_rb([_at(0, 1), _at(30, 2)]))
    proc.process(_rb([_at(65, 5)]))

    snapshot = proc.snapshot_state()
    assert snapshot is not None, "the open 01:00 window is state and must be snapshotted"

    revived = _processor()
    revived.restore_state(snapshot)

    # The 01:00 window's partial survived: flushing must still carry the 5 folded into it.
    flushed = revived.finalize()
    rows = pa.Table.from_batches(flushed).to_pydict()
    assert dict(zip(rows["w"], rows["total"], strict=True)) == {_BASE + dt.timedelta(hours=1): 5}, (
        "the open window's running partial was lost across the restore"
    )


def test_the_restored_watermark_still_rejects_late_rows():
    """The watermark is state too — a lost one re-admits rows it had already ruled late."""
    proc = _processor()
    proc.process(_rb([_at(0, 1), _at(30, 2)]))
    proc.process(_rb([_at(65, 5)]))  # watermark now 01:05 - 5m = 01:00

    revived = _processor()
    revived.restore_state(proc.snapshot_state())

    # A row at 00:10 is far behind the restored watermark → dropped as late, so the 00:00
    # window must not reappear. With the watermark lost, this row would be accepted and a
    # duplicate 00:00 window emitted after it was already published.
    revived.process(_rb([_at(10, 99)]))
    rows = pa.Table.from_batches(revived.finalize()).to_pydict()

    assert _BASE not in rows["w"], (
        "a late row resurrected an already-emitted window: the watermark did not survive the "
        "restore"
    )
    assert dict(zip(rows["w"], rows["total"], strict=True)) == {_BASE + dt.timedelta(hours=1): 5}


def test_a_watermark_with_no_open_windows_still_survives():
    """Advanced watermark, empty partials: the snapshot must still carry event time forward.

    `flush` empties the running partials but leaves the watermark set. Snapshotting nothing
    here would silently rewind event time to the next batch's maximum on restore, re-admitting
    rows that were already ruled late.
    """
    proc = _processor()
    proc.process(_rb([_at(600, 1)]))  # watermark → 10:00 - 5m = 09:55
    proc.finalize()  # drops the running partials; the watermark remains

    snapshot = proc.snapshot_state()
    assert snapshot is not None, "a watermark with no open windows is still state"

    revived = _processor()
    revived.restore_state(snapshot)

    # A row hours behind the restored watermark is late → dropped → nothing to emit. With the
    # watermark lost, it would be accepted and its window emitted.
    revived.process(_rb([_at(0, 42)]))
    assert revived.finalize() == [], (
        "a row behind the restored watermark was accepted: the watermark did not survive a "
        "snapshot taken with no open windows"
    )
