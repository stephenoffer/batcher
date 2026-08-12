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


@pytest.mark.integration
@pytest.mark.parametrize("fmt", ["parquet", "csv", "arrow"])
def test_streaming_write_matches_collect(tmp_path, fmt):
    """A breaker-free file→file write streams batch-by-batch yet equals collect+write."""
    src = str(tmp_path / "src.parquet")
    pq.write_table(
        pa.table({"x": list(range(2000)), "y": [str(i) for i in range(2000)]}),
        src,
        row_group_size=128,
    )
    pipe = bt.read.parquet(src).filter(bt.col("x") < 700).select("x", "y")
    out = str(tmp_path / f"out.{fmt}")
    getattr(pipe.write, fmt)(out)

    read_back = getattr(bt.read, fmt)(out).collect()
    ref = pipe.collect()
    assert read_back.num_rows == ref.num_rows == 700
    assert sorted(read_back.column("x").to_pylist()) == list(range(700))


@pytest.mark.integration
def test_streaming_write_is_chosen_only_for_lazy_sources(tmp_path):
    """The streaming write path bounds driver memory for lazy (file) sources; a
    resident in-memory source keeps the collect path (already in RAM, and it persists
    per-column stats)."""
    from batcher.api.terminal.core import _streaming_write_eligible

    src = str(tmp_path / "src.parquet")
    pq.write_table(pa.table({"x": list(range(64))}), src)

    file_ds = bt.read.parquet(src).filter(bt.col("x") < 32).select("x")
    mem_ds = bt.from_pydict({"x": list(range(64))}).filter(bt.col("x") < 32).select("x")
    # args: distributed, partition_by, max_rows_per_file, num_files, target_bytes_per_file
    args = (False, None, None, None, None)
    assert _streaming_write_eligible(file_ds._plan, file_ds._sources, *args) is True
    assert _streaming_write_eligible(mem_ds._plan, mem_ds._sources, *args) is False
    # A breaker (sort) is never eligible — it must materialize first.
    sorted_ds = bt.read.parquet(src).sort("x")
    assert _streaming_write_eligible(sorted_ds._plan, sorted_ds._sources, *args) is False


@pytest.mark.integration
def test_unbounded_breaker_error_names_offending_node():
    """An unbounded source under a breaker that must materialize raises a `PlanError`
    that names the *blocking* operator, so the message is actionable rather than a
    generic "must materialize".

    It used to name the plan's root ("top-level Sort"), which is the culprit only when the
    breaker happens to be at the root. `Aggregate(Sort(scan))` reported the aggregate — the
    one operator in that query that does stream. The refusal now asks Kyber which nodes
    cannot emit (`kyber.streaming.blocking_operators`), so the case below still names the
    sort and the deeper case names it too.
    """
    from batcher._internal.errors import PlanError

    def factory():
        yield pa.RecordBatch.from_pydict({"x": [3, 1, 2]})

    def stream():
        return bt.from_batches(factory, pa.schema([("x", pa.int64())]), bounded=False)

    with pytest.raises(PlanError, match="its sort cannot emit"):
        list(stream().sort("x").iter_batches())

    # The case the root-naming version got wrong: the blocking operator is *under* a
    # streamable one, and it is the sort that has to be named.
    nested = stream().sort("x").group_by("x").agg(n=bt.count())
    with pytest.raises(PlanError, match="its sort cannot emit"):
        list(nested.iter_batches())


@pytest.mark.integration
def test_streaming_topn_over_unbounded_source():
    # `sort + limit` over an unbounded source streams via `stream_topn`, keeping only the
    # running best N rows — the global top-N, identical to sorting the whole input then
    # taking the first N.
    def factory():
        yield pa.RecordBatch.from_pydict({"x": [5, 2, 8]})
        yield pa.RecordBatch.from_pydict({"x": [1, 9, 3]})
        yield pa.RecordBatch.from_pydict({"x": [7, 4, 6]})

    ds = bt.from_batches(factory, pa.schema([("x", pa.int64())]), bounded=False).sort("x").limit(4)
    out = pa.Table.from_batches(list(ds.iter_batches())).column("x").to_pylist()
    assert out == [1, 2, 3, 4]  # global top-4 ascending, not per-batch


@pytest.mark.integration
def test_streaming_limit_short_circuits_over_unbounded_source():
    # `limit(n)` over an unbounded source reads only until n rows are produced, then stops —
    # it must not drain the whole (endless) source.
    consumed: list[int] = []

    def factory():
        for i in range(1000):
            consumed.append(i)
            yield pa.RecordBatch.from_pydict({"x": [i]})

    ds = bt.from_batches(factory, pa.schema([("x", pa.int64())]), bounded=False).limit(3)
    out = pa.Table.from_batches(list(ds.iter_batches())).column("x").to_pylist()
    assert out == [0, 1, 2]
    assert len(consumed) < 10  # short-circuited, did not read all 1000 batches


@pytest.mark.integration
def test_streaming_global_aggregate_over_empty_stream_yields_one_row():
    # A keyless aggregate over an empty stream still yields exactly one row (COUNT=0,
    # SUM=NULL) — what SQL/DuckDB/collect() produce. The incremental fold has no partial to
    # finalize (it skips empty batches), so it falls back to the engine's empty-input result
    # rather than silently yielding zero rows.
    def factory():
        yield pa.RecordBatch.from_pydict({"v": pa.array([], type=pa.int64())})

    ds = bt.from_batches(factory, pa.schema([("v", pa.int64())]), bounded=False).agg(
        n=bt.col("v").count(), s=bt.col("v").sum()
    )
    out = pa.Table.from_batches(list(ds.iter_batches())).to_pydict()
    assert out == {"n": [0], "s": [None]}


@pytest.mark.integration
def test_streaming_distinct_over_unbounded_source():
    # `distinct()` over an unbounded source folds identical rows into one running group
    # (state bounded by the distinct-row count), yielding the exact set once drained.
    def factory():
        yield pa.RecordBatch.from_pydict({"k": ["a", "b", "a"]})
        yield pa.RecordBatch.from_pydict({"k": ["b", "c", "a"]})

    ds = bt.from_batches(factory, pa.schema([("k", pa.string())]), bounded=False).distinct()
    out = sorted(pa.Table.from_batches(list(ds.iter_batches())).column("k").to_pylist())
    assert out == ["a", "b", "c"]


@pytest.mark.integration
def test_streaming_write_empty_result_keeps_schema(tmp_path):
    """A streamable write whose filter removes every row still writes a valid empty
    file with the right columns (the empty-stream path)."""
    src = str(tmp_path / "src.parquet")
    pq.write_table(pa.table({"x": list(range(10)), "y": list(range(10))}), src)
    out = str(tmp_path / "empty.parquet")
    bt.read.parquet(src).filter(bt.col("x") < 0).select("x", "y").write.parquet(out)
    t = pq.read_table(out)
    assert t.num_rows == 0
    assert t.column_names == ["x", "y"]
