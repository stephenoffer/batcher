"""End-to-end predicate pushdown into the reader (correctness + that it happens).

Pushdown is a pure I/O optimization: the engine keeps its `Filter`, so results
must be byte-identical to the unfiltered-then-filtered path. We also verify the
predicate actually reaches a capable source's `read()`.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt


@pytest.mark.integration
def test_parquet_filter_pushdown_correct(tmp_path):
    path = str(tmp_path / "t.parquet")
    pq.write_table(
        pa.table({"x": list(range(100)), "y": [i * 2 for i in range(100)]}),
        path,
        row_group_size=10,
    )
    out = bt.read.parquet(path).filter(bt.col("x") > 50).select("x").collect()
    assert sorted(out.column("x").to_pylist()) == list(range(51, 100))


@pytest.mark.integration
def test_parquet_dataset_filter_pushdown_correct(tmp_path):
    out_dir = str(tmp_path / "p")
    bt.from_arrow(pa.table({"k": [0, 0, 1, 1], "v": [1, 2, 3, 4]})).write.parquet(
        out_dir, partition_by=["k"]
    )
    out = bt.read.parquet_dataset(out_dir).filter(bt.col("v") >= 3).collect()
    assert sorted(out.column("v").to_pylist()) == [3, 4]


@pytest.mark.integration
def test_predicate_reaches_source(tmp_path, monkeypatch):
    path = str(tmp_path / "t.parquet")
    pq.write_table(pa.table({"x": list(range(20))}), path)

    from batcher.io.formats.structured.parquet import ParquetSource

    seen: dict[str, object] = {}
    original = ParquetSource.read

    def spy(self, projection=None, predicate=None):
        seen["predicate"] = predicate
        return original(self, projection, predicate)

    monkeypatch.setattr(ParquetSource, "read", spy)
    bt.read.parquet(path).filter(bt.col("x") > 10).collect()
    assert seen["predicate"] is not None  # the pushed predicate reached the reader


@pytest.mark.integration
def test_pushdown_matches_no_pushdown(tmp_path):
    """The pushed result equals the same query with pushdown disabled (engine-only)."""
    path = str(tmp_path / "t.parquet")
    pq.write_table(pa.table({"x": list(range(50)), "y": list(range(50))}), path)

    pushed = bt.read.parquet(path).filter((bt.col("x") > 10) & (bt.col("y") < 40)).collect()

    from batcher.io.formats.structured.parquet import ParquetSource

    flag = ParquetSource.supports_predicate
    try:
        ParquetSource.supports_predicate = False
        engine_only = (
            bt.read.parquet(path).filter((bt.col("x") > 10) & (bt.col("y") < 40)).collect()
        )
    finally:
        ParquetSource.supports_predicate = flag

    assert pushed.sort_by("x").equals(engine_only.sort_by("x"))


@pytest.mark.integration
def test_distributed_pushdown_matches_single_node(tmp_path):
    """Distributed reads push projection + predicate to each worker's split read."""
    path = str(tmp_path / "big.parquet")
    pq.write_table(
        pa.table({"k": [i % 5 for i in range(1000)], "v": list(range(1000))}),
        path,
        row_group_size=50,
    )

    def q(**kw):
        ds = bt.read.parquet(path).filter(bt.col("v") > 500).group_by("k").agg(s=bt.col("v").sum())
        return sorted(ds.collect(**kw).to_pylist(), key=lambda r: r["k"])

    assert q() == q(distributed=True, num_workers=4)
