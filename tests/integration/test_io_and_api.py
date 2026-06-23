"""IO round-trips and small API conveniences."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher import col

pytest.importorskip("batcher._native", reason="native engine not built")


def test_parquet_write_read_roundtrip(tmp_path):
    p = str(tmp_path / "out.parquet")
    bt.from_pydict({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]}).filter(col("x") > 1).write.parquet(p)
    assert bt.read.parquet(p).collect().to_pydict() == {"x": [2, 3, 4], "y": [20, 30, 40]}


def test_csv_write_read_roundtrip(tmp_path):
    p = str(tmp_path / "out.csv")
    bt.from_pydict({"a": [1, 2], "b": [3, 4]}).write.csv(p)
    assert bt.read.csv(p).collect().to_pydict() == {"a": [1, 2], "b": [3, 4]}


def test_count_and_iter_batches():
    assert bt.from_pydict({"x": list(range(10))}).filter(col("x") >= 7).count() == 3
    sizes = [
        b.num_rows for b in bt.from_pydict({"x": list(range(100))}).iter_batches(batch_size=40)
    ]
    assert sizes == [40, 40, 20]


def test_drop_rename_with_column():
    ds = bt.from_pydict({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    assert ds.drop("b").collect().column_names == ["a", "c"]
    assert ds.rename({"a": "x"}).collect().column_names == ["x", "b", "c"]
    out = ds.with_column("d", col("a") + col("b")).collect().to_pydict()
    assert out["d"] == [4, 6]


def test_projection_pushdown_reads_subset_from_parquet(tmp_path):
    p = str(tmp_path / "wide.parquet")
    pq.write_table(pa.table({"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8]}), p)
    # Only 'a' and 'c' are needed; the query must still be correct.
    out = bt.read.parquet(p).filter(col("a") > 1).select("a", "c").collect()
    assert out.to_pydict() == {"a": [2], "c": [6]}
