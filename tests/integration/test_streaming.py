"""Streaming execution: breaker-free pipelines run incrementally, == batch result."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col
from batcher.plan.logical import is_streamable


def _factory(n_batches=50, counter=None):
    def f():
        for i in range(n_batches):
            if counter is not None:
                counter["n"] += 1
            yield pa.record_batch({"x": list(range(i * 10, i * 10 + 10)), "y": [i] * 10})

    return f


SCHEMA = pa.schema([("x", pa.int64()), ("y", pa.int64())])


def test_streaming_is_lazy():
    counter = {"n": 0}
    ds = bt.from_batches(_factory(50, counter), SCHEMA).filter(col("x") % 2 == 0)
    it = ds.iter_batches()
    [next(it) for _ in range(3)]
    assert counter["n"] <= 4  # did not pull all 50 source batches


def test_streaming_matches_batch():
    ds = bt.from_batches(_factory(50), SCHEMA).filter(col("x") > 100).select(z=col("x") * col("y"))
    streamed = pa.Table.from_batches(list(ds.iter_batches()))
    batched = ds.collect()
    assert sorted(streamed.column("z").to_pylist()) == sorted(batched.column("z").to_pylist())


def test_streaming_map_batches():
    # Streaming batch-inference style pipeline (map_batches is per-batch, streamable).
    def add_pred(batch: pa.RecordBatch) -> pa.RecordBatch:
        pred = pa.array([v * 2 for v in batch.column("x").to_pylist()], pa.int64())
        return batch.append_column("pred", pred)

    ds = bt.from_batches(_factory(30), SCHEMA).map_batches(add_pred)
    streamed = pa.Table.from_batches(list(ds.iter_batches()))
    batched = ds.collect()
    assert streamed.num_rows == batched.num_rows
    assert sorted(streamed.column("pred").to_pylist()) == sorted(batched.column("pred").to_pylist())


def test_streamability_classification():
    base = bt.from_batches(_factory(2), SCHEMA)
    assert is_streamable(base.filter(col("x") > 0).select("x")._plan)
    # A breaker (group_by) makes it non-streamable → falls back to materialize.
    assert not is_streamable(base.group_by("y").agg(s=col("x").sum())._plan)


def test_streaming_falls_back_for_breakers():
    # group_by isn't streamable; iter_batches() still works (materializes).
    ds = bt.from_batches(_factory(10), SCHEMA).group_by("y").agg(s=col("x").sum())
    out = pa.Table.from_batches(list(ds.iter_batches()))
    assert out.num_rows == ds.collect().num_rows


def test_streaming_sort_matches_collect_and_is_ordered():
    # A top-level sort streams from the out-of-core bucket pipeline (input → disk,
    # then globally-ordered buckets yielded one at a time) — bounded memory, and the
    # streamed order is identical to collect() position-by-position.
    import numpy as np

    rng = np.random.default_rng(7)
    t = pa.table(
        {
            "k": rng.integers(0, 5000, 20000).astype("int64"),
            "v": rng.integers(0, 100, 20000).astype("int64"),
        }
    )
    ds = bt.from_arrow(t).filter(col("v") > 50).sort("k")
    streamed = pa.Table.from_batches(list(ds.iter_batches()))
    collected = ds.collect()
    assert streamed.column("k").to_pylist() == collected.column("k").to_pylist()


def test_streaming_top_n_sort_matches_collect():
    # A streamed top-N stops once `limit` rows are emitted (never sorts past budget).
    import numpy as np

    rng = np.random.default_rng(9)
    t = pa.table({"k": rng.integers(0, 100000, 30000).astype("int64")})
    ds = bt.from_arrow(t).sort("k", descending=True).limit(20)
    streamed = pa.Table.from_batches(list(ds.iter_batches()))
    assert streamed.column("k").to_pylist() == ds.collect().column("k").to_pylist()


def test_streaming_join_matches_collect():
    # A top-level join streams each co-partitioned bucket-pair's output as produced.
    import numpy as np

    rng = np.random.default_rng(8)
    left = bt.from_arrow(
        pa.table(
            {
                "k": rng.integers(0, 200, 10000).astype("int64"),
                "a": rng.integers(0, 10, 10000).astype("int64"),
            }
        )
    )
    right = bt.from_arrow(
        pa.table({"k": np.arange(200).astype("int64"), "b": np.arange(200).astype("int64")})
    )
    jds = left.join(right, on="k")

    def _rowset(tb):
        cols = tb.column_names
        return sorted(tuple(r[c] for c in cols) for r in tb.to_pylist())

    assert _rowset(pa.Table.from_batches(list(jds.iter_batches()))) == _rowset(jds.collect())
