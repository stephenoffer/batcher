"""Streaming execution (`iter_batches()`) across source types.

A breaker-free pipeline (scan/filter/project) is consumed one source batch at a
time in bounded memory. Any source works because it implements `iter_batches`;
capable sources additionally get projection + predicate pushed into the stream
(the engine's `Filter` re-checks, so correctness never depends on the push).
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt


@pytest.mark.integration
def test_streaming_parquet_filter_project(tmp_path):
    path = str(tmp_path / "t.parquet")
    pq.write_table(
        pa.table({"x": list(range(1000)), "y": [i * 2 for i in range(1000)]}),
        path,
        row_group_size=100,
    )
    ds = bt.read.parquet(path).filter(bt.col("x") > 900).select("x")
    rows = sorted(v for b in ds.iter_batches() for v in b.column("x").to_pylist())
    assert rows == list(range(901, 1000))


@pytest.mark.integration
def test_streaming_parquet_dataset_pushdown(tmp_path):
    """Streaming over a partitioned dataset (the path Delta also uses)."""
    out = str(tmp_path / "pd")
    bt.from_arrow(
        pa.table({"k": [i % 5 for i in range(500)], "v": list(range(500))})
    ).write.parquet(out, partition_by=["k"])
    ds = bt.read.parquet_dataset(out).filter(bt.col("v") >= 480).select("v")
    rows = sorted(v for b in ds.iter_batches() for v in b.column("v").to_pylist())
    assert rows == list(range(480, 500))


@pytest.mark.integration
def test_streaming_unbounded_iterator_source():
    """An unbounded/iterator source (the Kafka-like shape) streams batch-by-batch."""

    def factory():
        for i in range(3):
            yield pa.RecordBatch.from_pydict({"x": [i * 10, i * 10 + 1]})

    ds = bt.from_batches(factory, pa.schema([("x", pa.int64())])).filter(bt.col("x") > 5)
    rows = sorted(v for b in ds.iter_batches() for v in b.column("x").to_pylist())
    assert rows == [10, 11, 20, 21]


@pytest.mark.integration
def test_streaming_matches_collect(tmp_path):
    path = str(tmp_path / "t.parquet")
    pq.write_table(pa.table({"x": list(range(200)), "y": list(range(200))}), path)
    pipe = bt.read.parquet(path).filter(bt.col("y") < 50).select("x")
    streamed = sorted(v for b in pipe.iter_batches() for v in b.column("x").to_pylist())
    collected = sorted(pipe.collect().column("x").to_pylist())
    assert streamed == collected == list(range(50))
