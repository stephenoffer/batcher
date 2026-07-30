"""Property: the watermark operators agree with a batch oracle when nothing is late.

The watermark operators are the streaming machinery with the most retained state and the
least coverage, and their bugs have all been the same shape: right on the fixed example a
unit test used, wrong on a different *micro-batch chunking* or a different event-time
order. A stream-stream join once matched every pair because both sides named their time
column the same; the dedup once re-emitted a key it had evicted; the seen-key state once
selected its event-time column twice.

Chunking is the variable those bugs hide behind, so it is the one this varies. Every case
here is built so **no row is late** — event times are non-decreasing and the lateness is
wider than any batch's spread — which pins the answer to a batch oracle exactly:

* an un-late windowed aggregate equals the batch aggregate over the same windows;
* an un-late dedup equals `distinct` over the subset;
* an un-late interval join equals the inner join filtered to the interval.

Anything the watermark drops or evicts here would be a divergence, which is the point.
"""

from __future__ import annotations

import datetime as dt
import uuid

import pyarrow as pa
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import batcher as bt
from batcher import col

pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = [pytest.mark.property, pytest.mark.integration]

_BASE = dt.datetime(2024, 1, 1)
_SCHEMA = pa.schema([("ts", pa.timestamp("us")), ("k", pa.int64()), ("v", pa.int64())])

_HC = [HealthCheck.too_slow, HealthCheck.function_scoped_fixture]
_PROP = settings(max_examples=25, deadline=None, suppress_health_check=_HC)


@st.composite
def _events(draw: st.DrawFn) -> pa.Table:
    """Rows whose event time never goes backwards — so nothing is ever late."""
    n = draw(st.integers(min_value=0, max_value=24))
    gaps = draw(st.lists(st.integers(min_value=0, max_value=90), min_size=n, max_size=n))
    keys = draw(st.lists(st.integers(min_value=0, max_value=3), min_size=n, max_size=n))
    vals = draw(st.lists(st.integers(min_value=-5, max_value=5), min_size=n, max_size=n))
    seconds, running = [], 0
    for g in gaps:
        running += g
        seconds.append(running)
    return pa.table(
        {
            "ts": pa.array([_BASE + dt.timedelta(seconds=s) for s in seconds], pa.timestamp("us")),
            "k": pa.array(keys, pa.int64()),
            "v": pa.array(vals, pa.int64()),
        },
        schema=_SCHEMA,
    )


def _stream(table: pa.Table, chunk: int, *, bounded: bool = False):
    batches = table.to_batches(max_chunksize=chunk) if table.num_rows else []

    def gen():
        yield from batches

    return bt.from_batches(gen, _SCHEMA, bounded=bounded)


def _sorted_rows(table: pa.Table) -> list[tuple]:
    cols = sorted(table.column_names)
    return sorted(tuple(r[c] for c in cols) for r in table.to_pylist())


def _collect(batches) -> pa.Table:
    batches = [b for b in batches if b.num_rows]
    return pa.Table.from_batches(batches) if batches else pa.table({})


@_PROP
@given(_events(), st.sampled_from([1, 3, 7]))
def test_an_un_late_windowed_aggregate_equals_the_batch_oracle(table, chunk):
    """Every window closes or is flushed, and no row is dropped, so the streamed windows
    are exactly the batch aggregate's."""
    windowed = (
        _stream(table, chunk)
        .with_watermark("ts", "1h")
        .group_by(w=bt.window(col("ts"), "1m"))
        .agg(total=col("v").sum(), n=bt.count())
    )
    streamed = _collect(list(windowed.iter_batches()))
    oracle = (
        bt.from_arrow(table)
        .group_by(w=bt.window(col("ts"), "1m"))
        .agg(total=col("v").sum(), n=bt.count())
        .collect()
    )
    if not table.num_rows:
        assert streamed.num_rows == 0
        return
    assert _sorted_rows(streamed) == _sorted_rows(oracle)


@_PROP
@given(_events(), st.sampled_from([1, 3, 7]))
def test_an_un_late_dedup_equals_distinct(table, chunk):
    """With a lateness wider than the whole stream nothing evicts, so the watermark dedup
    is exact deduplication — the same rows `distinct` keeps."""
    deduped = _stream(table, chunk).drop_duplicates_within_watermark(
        ["k"], event_time="ts", lateness="24h"
    )
    streamed = _collect(list(deduped.iter_batches()))
    if not table.num_rows:
        return
    keys = sorted(set(table.column("k").to_pylist()))
    assert sorted(streamed.column("k").to_pylist()) == keys


@_PROP
@given(_events(), st.sampled_from([1, 5]))
def test_dedup_keeps_the_earliest_row_per_key(table, chunk):
    """`keep="first" by event time` is the documented rule, and a chunking that splits a
    key's occurrences across batches must not change which one survives."""
    deduped = _stream(table, chunk).drop_duplicates_within_watermark(
        ["k"], event_time="ts", lateness="24h"
    )
    streamed = _collect(list(deduped.iter_batches()))
    if not table.num_rows:
        return
    earliest: dict[int, dt.datetime] = {}
    for row in table.to_pylist():
        earliest.setdefault(row["k"], row["ts"])
        earliest[row["k"]] = min(earliest[row["k"]], row["ts"])
    got = {r["k"]: r["ts"] for r in streamed.to_pylist()}
    assert got == earliest


@_PROP
@given(_events(), st.sampled_from([1, 4]))
def test_an_un_late_interval_join_equals_the_filtered_inner_join(table, chunk):
    """Both sides carry the same timestamps, so every matching pair is 0s apart and inside
    any interval — the streamed join must therefore be the plain inner join."""
    left = _stream(table, chunk)
    right = _stream(table, chunk)
    joined = left.join_stream(
        right, on="k", left_time="ts", right_time="ts", within="1h", lateness="24h"
    )
    streamed = _collect(list(joined.iter_batches()))
    oracle = bt.from_arrow(table).join(bt.from_arrow(table), on="k", how="inner").collect()
    assert streamed.num_rows == oracle.num_rows


@_PROP
@given(_events(), st.sampled_from([2, 6]))
def test_a_streaming_query_over_a_windowed_aggregate_matches_its_iterator(table, chunk):
    """The `write(...)` engine and the `iter_batches` driver share a fold; a chunking that
    changes one and not the other would be a fork in the semantics."""

    def build(source):
        return (
            source.with_watermark("ts", "1h")
            .group_by(w=bt.window(col("ts"), "1m"))
            .agg(total=col("v").sum())
        )

    iterated = _collect(list(build(_stream(table, chunk)).iter_batches()))
    name = f"prop-wm-{uuid.uuid4().hex}"
    query = build(_stream(table, chunk, bounded=True)).write.memory(
        name, trigger=bt.Trigger.available_now(), output_mode="append"
    )
    assert query.await_termination(60) is True
    assert query.exception() is None, query.exception()
    written = bt.read_memory(name).collect()
    if not table.num_rows:
        assert written.num_rows == 0
        return
    assert _sorted_rows(written) == _sorted_rows(iterated)
