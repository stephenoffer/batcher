"""Workstream B — watermark-bounded windowed streaming aggregation.

A windowed `group_by(window(...))` with `.with_watermark(...)` over an unbounded
source emits each window once the watermark passes its end, drops late rows, and
flushes open windows at end-of-stream — with state bounded by the open-window count.
The emitted result equals the batch aggregate over the on-time rows.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

_BASE = dt.datetime(2024, 1, 1, 0, 0, 0)
_SCHEMA = pa.schema([("ts", pa.timestamp("us")), ("v", pa.int64())])


def _at(minute: int, v: int) -> dict:
    return {"ts": _BASE + dt.timedelta(minutes=minute), "v": v}


def _rb(rows: list[dict]) -> pa.RecordBatch:
    return pa.RecordBatch.from_pylist(rows, schema=_SCHEMA)


def _batches():
    # Micro-batches walking event time forward (within the 5-minute lateness) across
    # hourly windows — no row lands behind the watermark, so none is dropped.
    yield _rb([_at(0, 1), _at(30, 2)])  # 00:00 window
    yield _rb([_at(65, 5), _at(90, 3)])  # 01:00 window; advancing wm closes 00:00
    yield _rb([_at(150, 4)])  # 02:00 window; closes 01:00
    yield _rb([_at(200, 6)])  # 03:00 window; closes 02:00 (03:00 flushed at end)


def _stream():
    return bt.from_batches(_batches, _SCHEMA, bounded=False)


def _windowed(ds):
    return (
        ds.with_watermark("ts", "5m")
        .group_by(w=bt.window(col("ts"), "1h"))
        .agg(total=col("v").sum())
    )


@pytest.mark.integration
def test_windowed_iter_batches_matches_batch_oracle():
    streamed = pa.Table.from_batches(list(_windowed(_stream()).iter_batches()))
    got = dict(zip(*[streamed.to_pydict()[c] for c in ("w", "total")], strict=True))
    # Batch oracle over the same rows (no watermark → all windows).
    oracle_tbl = pa.Table.from_batches(list(_batches()))
    batch = _windowed(bt.from_arrow(oracle_tbl)).to_pydict()
    oracle = dict(zip(*[batch[c] for c in ("w", "total")], strict=True))
    # Hour buckets: 00:00→{0',30'}=3, 01:00→{90',65'}=8, 02:00→{150'}=4, 03:00→{200'}=6.
    assert (
        got
        == oracle
        == {
            _BASE: 3,
            _BASE + dt.timedelta(hours=1): 8,
            _BASE + dt.timedelta(hours=2): 4,
            _BASE + dt.timedelta(hours=3): 6,
        }
    )


@pytest.mark.integration
def test_windowed_append_to_memory_sink():
    q = _windowed(_stream()).write.memory(
        "m_win", trigger=bt.Trigger.available_now(), output_mode="append"
    )
    q.await_termination()
    out = bt.read_memory("m_win").to_pydict()
    got = dict(zip(out["w"], out["total"], strict=True))
    assert got == {
        _BASE: 3,
        _BASE + dt.timedelta(hours=1): 8,
        _BASE + dt.timedelta(hours=2): 4,
        _BASE + dt.timedelta(hours=3): 6,
    }


@pytest.mark.integration
def test_late_rows_are_dropped():
    # A row far behind the watermark (00:10) arriving after time has advanced to
    # 03:00 is dropped, so the 00:00 window keeps its original total.
    def batches():
        yield _rb([_at(0, 1), _at(30, 2)])
        yield _rb([_at(190, 9)])  # advances watermark well past the 00:00 window
        yield _rb([_at(10, 100)])  # late → dropped (00:00 window already closed)

    ds = (
        bt.from_batches(batches, _SCHEMA, bounded=False)
        .with_watermark("ts", "5m")
        .group_by(w=bt.window(col("ts"), "1h"))
        .agg(total=col("v").sum())
    )
    out = pa.Table.from_batches(list(ds.iter_batches())).to_pydict()
    got = dict(zip(out["w"], out["total"], strict=True))
    assert got[_BASE] == 3  # 1 + 2, the late 100 dropped


@pytest.mark.integration
def test_windowed_state_cap_fails_loudly_instead_of_oom():
    # A tiny `streaming_state_max_bytes`: the retained open-window state exceeds it, so
    # the fold raises a clear ResourceError (the stalled-watermark / huge-key-space
    # signal) instead of growing unbounded toward OOM.
    import dataclasses

    from batcher._internal.errors import ResourceError
    from batcher.config import active_config, config_context

    cfg = active_config()
    tiny = cfg.replace(memory=dataclasses.replace(cfg.memory, streaming_state_max_bytes=1))
    with config_context(tiny), pytest.raises(ResourceError, match="streaming aggregate state"):
        list(_windowed(_stream()).iter_batches())


@pytest.mark.integration
def test_windowed_default_cap_does_not_break_normal_stream():
    # The generous derived default cap never trips on a normally-advancing watermark.
    streamed = pa.Table.from_batches(list(_windowed(_stream()).iter_batches()))
    assert streamed.num_rows > 0
