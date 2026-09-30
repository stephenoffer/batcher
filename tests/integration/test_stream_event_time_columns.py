"""Two ways a real event-time column reached `window`/`with_watermark` and went wrong.

* A column **derived** beneath the watermark (a decoded Kafka JSON field) raised
  `KeyError`: the windowed fold read the watermark column off the raw source batch, which
  does not have it.
* A broker's ``timestamp`` is epoch **milliseconds**. As a bare int64 the engine read it as
  microseconds, so an hour of events landed in a single 1970 window.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.io.formats.streaming.broker.schema import broker_schema

pytestmark = pytest.mark.integration

_BASE = dt.datetime(2024, 1, 1)


def test_a_watermark_on_a_derived_column_windows_correctly():
    inner = pa.struct([("ts", pa.timestamp("us")), ("n", pa.int64())])
    schema = pa.schema([("value", inner)])

    def feed():
        for i in range(4):
            row = {"ts": _BASE + dt.timedelta(minutes=30 * i), "n": 1}
            yield pa.record_batch({"value": [row]}, schema=schema)

    windowed = (
        bt.from_batches(feed, schema, bounded=False)
        .select(ts=col("value").struct.field("ts"), n=col("value").struct.field("n"))
        .with_watermark("ts", "10m")
        .group_by(w=bt.window(col("ts"), "1h"))
        .agg(c=col("n").sum())
    )
    rows = [r for b in windowed.iter_batches() for r in b.to_pylist()]
    assert sorted((r["w"], r["c"]) for r in rows) == [
        (_BASE, 2),
        (_BASE + dt.timedelta(hours=1), 2),
    ]


def test_the_broker_timestamp_is_a_millisecond_timestamp():
    assert broker_schema().field("timestamp").type == pa.timestamp("ms")


@pytest.mark.parametrize("streaming", [False, True], ids=["batch", "stream"])
def test_windowing_a_broker_timestamp_uses_its_real_time(streaming):
    millis = int(_BASE.replace(tzinfo=dt.UTC).timestamp() * 1000)
    stamps = [millis, millis + 30 * 60_000, millis + 90 * 60_000]
    ts_type = broker_schema().field("timestamp").type
    schema = pa.schema([("timestamp", ts_type), ("n", pa.int64())])
    batch = pa.record_batch({"timestamp": pa.array(stamps, type=ts_type), "n": [1, 2, 3]}, schema)
    if streaming:
        ds = bt.from_batches(lambda: iter([batch]), schema, bounded=False).with_watermark(
            "timestamp", "10m"
        )
    else:
        ds = bt.from_arrow(pa.Table.from_batches([batch]))
    windowed = ds.group_by(w=bt.window(col("timestamp"), "1h")).agg(c=col("n").sum())
    rows = [r for b in windowed.iter_batches() for r in b.to_pylist()]
    got = sorted((r["w"].replace(tzinfo=None), r["c"]) for r in rows)
    assert got == [(_BASE, 3), (_BASE + dt.timedelta(hours=1), 3)]


_SCHEMA = pa.schema([("k", pa.int64()), ("v", pa.int64())])


def _endless_shapes():
    def feed():
        yield pa.record_batch({"k": [1, 2], "v": [3, 4]}, schema=_SCHEMA)

    stream = bt.from_batches(feed, _SCHEMA, bounded=False)
    return {
        "aggregate": stream.group_by("k").agg(s=col("v").sum()),
        "distinct": stream.distinct(),
        "top_n": stream.sort("v").limit(1),
    }


@pytest.mark.parametrize("shape", ["aggregate", "distinct", "top_n"])
def test_iter_batches_warns_when_a_stream_would_emit_only_at_end(shape):
    from batcher._internal.errors import PerformanceWarning

    ds = _endless_shapes()[shape]
    with pytest.warns(PerformanceWarning, match="emits once, at end of input"):
        next(iter(ds.iter_batches()), None)


def test_a_row_wise_stream_and_a_bounded_aggregate_do_not_warn():
    import warnings

    from batcher._internal.errors import PerformanceWarning

    def feed():
        yield pa.record_batch({"k": [1], "v": [2]}, schema=_SCHEMA)

    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        list(bt.from_batches(feed, _SCHEMA, bounded=False).filter(col("v") > 0).iter_batches())
        list(
            bt.from_pydict({"k": [1], "v": [2]}).group_by("k").agg(s=col("v").sum()).iter_batches()
        )
