"""Streaming (incremental) aggregation equals batch aggregation and DuckDB (W7).

Drives a group-by over a multi-batch source one micro-batch at a time, folding each
batch's partial state into one running state via the native `combine`. The result
must equal materializing the whole input and aggregating.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col, count


def _chunks(n_chunks: int = 6, per: int = 1000) -> list[pa.RecordBatch]:
    # Deterministic multi-batch data so the streaming path folds several partials.
    batches = []
    for c in range(n_chunks):
        ks = [(i + c) % 7 for i in range(per)]
        vs = [(i * 3 + c) % 50 for i in range(per)]
        batches.append(pa.record_batch({"k": ks, "v": pa.array(vs, type=pa.int64())}))
    return batches


def _stream(batches: list[pa.RecordBatch]):
    # A genuine streaming (re-iterable, iterator-backed) source.
    return bt.from_batches(lambda: iter(batches), batches[0].schema)


def _streamed(ds) -> pa.Table:
    parts = list(ds.iter_batches())
    return pa.Table.from_batches(parts) if parts else pa.table({})


def _norm(t: pa.Table) -> set:
    return {
        tuple(round(v, 6) if isinstance(v, float) else v for v in row.values())
        for row in t.to_pylist()
    }


def test_streaming_grouped_aggregate_matches_batch_and_duckdb(duck):
    batches = _chunks()
    full = pa.Table.from_batches(batches)
    duck.register("t", full)

    def q(ds):
        return ds.group_by("k").agg(
            s=col("v").sum(), n=count(), a=col("v").mean(), hi=col("v").max()
        )

    streamed = _streamed(q(_stream(batches)))
    batch_result = q(bt.from_arrow(full)).collect()
    expected = duck.sql("SELECT k, SUM(v) s, COUNT(*) n, AVG(v) a, MAX(v) hi FROM t GROUP BY k")

    assert _norm(streamed) == _norm(batch_result)
    assert _norm(streamed) == _norm(expected.to_arrow_table())


def test_streaming_global_aggregate_matches_batch(duck):
    batches = _chunks()
    full = pa.Table.from_batches(batches)

    def q(ds):
        return ds.group_by().agg(s=col("v").sum(), n=count())

    streamed = _streamed(q(_stream(batches))).to_pydict()
    batch_result = q(bt.from_arrow(full)).collect().to_pydict()
    assert streamed == batch_result


def test_streaming_aggregate_with_filter_matches_batch():
    batches = _chunks()
    full = pa.Table.from_batches(batches)

    def q(ds):
        return ds.filter(col("v") > 10).group_by("k").agg(s=col("v").sum())

    streamed = _streamed(q(_stream(batches)))
    batch_result = q(bt.from_arrow(full)).collect()
    assert _norm(streamed) == _norm(batch_result)


def test_streaming_topn_matches_batch_and_duckdb(duck):
    # head(k) over a sorted stream keeps only the running best k rows. Use a unique
    # sort key (`id`) so the top-k is unambiguous (no boundary ties).
    n_chunks, per = 6, 1000
    batches = []
    for c in range(n_chunks):
        ids = list(range(c * per, c * per + per))
        # shuffle-ish so the global top-k spans multiple chunks
        ids = ids[::-1] if c % 2 else ids
        batches.append(
            pa.record_batch({"id": pa.array(ids, type=pa.int64()), "g": [(i % 3) for i in ids]})
        )
    full = pa.Table.from_batches(batches)
    duck.register("t", full)

    streamed = _streamed(_stream(batches).sort("id", descending=True).head(20))
    batch_result = bt.from_arrow(full).sort("id", descending=True).head(20).collect()
    expected = duck.sql("SELECT * FROM t ORDER BY id DESC LIMIT 20")

    assert streamed.to_pydict() == batch_result.to_pydict()  # ordered top-N → exact
    assert _norm(streamed) == _norm(expected.to_arrow_table())


def test_streaming_tumbling_window_aggregate_matches_duckdb(duck):
    # A tumbling event-time window is a group-by on the time bucket, so it streams
    # via the same incremental aggregate driver — state bounded by the number of
    # open windows, not the input size.
    chunks = [
        pa.record_batch(
            {
                "ts": pa.array(list(range(c * 1000, (c + 1) * 1000)), type=pa.int64()),
                "v": pa.array([(i % 9) for i in range(1000)], type=pa.int64()),
            }
        )
        for c in range(5)
    ]
    full = pa.Table.from_batches(chunks)
    duck.register("t", full)
    window = (col("ts") / 100).floor()  # tumbling window of width 100 over event-time

    def q(ds):
        return ds.group_by(win=window).agg(s=col("v").sum(), n=count())

    streamed = _streamed(q(_stream(chunks)))
    batch_result = q(bt.from_arrow(full)).collect()
    expected = duck.sql("SELECT floor(ts / 100.0) win, SUM(v) s, COUNT(*) n FROM t GROUP BY 1")

    assert _norm(streamed) == _norm(batch_result)
    assert _norm(streamed) == _norm(expected.to_arrow_table())


def test_streaming_distinct_matches_batch_and_duckdb(duck):
    # DISTINCT streams with bounded memory (group-by-all-cols, no aggregates).
    batches = _chunks()
    full = pa.Table.from_batches(batches)
    duck.register("t", full)

    def q(ds):
        return ds.select("k", "v").distinct()

    streamed = _streamed(q(_stream(batches)))
    batch_result = q(bt.from_arrow(full)).collect()
    expected = duck.sql("SELECT DISTINCT k, v FROM t")
    assert _norm(streamed) == _norm(batch_result)
    assert _norm(streamed) == _norm(expected.to_arrow_table())
