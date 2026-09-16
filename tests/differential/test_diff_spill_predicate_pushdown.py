"""`collect(spill=True)` pushes a filter into the Parquet read, and still matches DuckDB.

Every out-of-core phase reads its source through one tap, and that tap passed a projection but
never a predicate. So a selective filter over a clustered Parquet table pruned its row groups
under `collect()` and decoded all of them under `collect(spill=True)`, which is the scheduling
chosen precisely because the input is large. The phase's map sub-plan still applies the
`Filter` to every morsel, so pushing the predicate may only ever change how much is read.

Each operator the spill path has -- aggregate, join, sort, window -- is checked against DuckDB,
positionally where the result is ordered, and each carries the control that makes the test
able to fail: the reader returned fewer rows than the table holds.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered
from batcher.io.formats.structured.parquet.source import ParquetSource

pytestmark = pytest.mark.differential

_ROWS = 40_000


@pytest.fixture
def rows_read(monkeypatch):
    seen = {"rows": 0}
    stream = ParquetSource.iter_batches

    def _stream(self, projection=None, predicate=None):
        for batch in stream(self, projection, predicate):
            seen["rows"] += batch.num_rows
            yield batch

    monkeypatch.setattr(ParquetSource, "iter_batches", _stream)
    return seen


@pytest.fixture
def path(tmp_path):
    rng = np.random.default_rng(3)
    ids = np.arange(_ROWS, dtype="int64")
    table = pa.table(
        {
            "id": ids,
            "g": rng.integers(0, 7, _ROWS),
            "v": pa.array(
                [None if i % 97 == 0 else float(x) for i, x in enumerate(rng.random(_ROWS))]
            ),
        }
    )
    pq.write_table(table, tmp_path / "t.parquet", row_group_size=2_000)
    return str(tmp_path / "t.parquet")


def test_aggregate(duck, path, rows_read):
    ds = bt.read.parquet(path).filter(bt.col("id") >= 36_000).group_by("g").agg(s=bt.col("v").sum())
    sql = f"SELECT g, sum(v) AS s FROM read_parquet('{path}') WHERE id >= 36000 GROUP BY g"
    assert_same(ds.collect(spill=True), duck.sql(sql))
    assert 0 < rows_read["rows"] < _ROWS // 2


def test_sort(duck, path, rows_read):
    ds = bt.read.parquet(path).filter(bt.col("id") < 3_000).sort("v", "id", descending=True)
    sql = (
        f"SELECT id, g, v FROM read_parquet('{path}') WHERE id < 3000 "
        "ORDER BY v DESC NULLS LAST, id DESC"
    )
    assert_same_ordered(ds.collect(spill=True), duck.sql(sql))
    assert 0 < rows_read["rows"] < _ROWS // 2


def test_join(duck, path, rows_read):
    left = bt.read.parquet(path).filter(bt.col("id") >= 38_000)
    right = (
        bt.read.parquet(path).filter(bt.col("id") >= 38_000).select("id", bt.col("g").alias("g2"))
    )
    ds = left.join(right, on="id")
    sql = (
        f"SELECT l.id, l.g, l.v, r.g AS g2 FROM read_parquet('{path}') l "
        f"JOIN read_parquet('{path}') r ON l.id = r.id WHERE l.id >= 38000 AND r.id >= 38000"
    )
    assert_same(ds.collect(spill=True), duck.sql(sql))
    assert 0 < rows_read["rows"] < _ROWS


def test_window(duck, path, rows_read):
    ds = (
        bt.read.parquet(path)
        .filter(bt.col("id") < 2_000)
        .with_columns(bt.col("v").sum().over(partition_by="g", order_by="id").alias("run"))
    )
    sql = (
        f"SELECT id, g, v, sum(v) OVER (PARTITION BY g ORDER BY id) AS run "
        f"FROM read_parquet('{path}') WHERE id < 2000"
    )
    assert_same(ds.collect(spill=True), duck.sql(sql))
    assert 0 < rows_read["rows"] < _ROWS // 2


def test_a_predicate_the_reader_cannot_use_still_answers(duck, path, rows_read):
    """No pruning is possible on an unclustered column; the answer must not depend on it."""
    ds = bt.read.parquet(path).filter(bt.col("v") > 0.99).group_by("g").agg(n=bt.col("id").count())
    sql = f"SELECT g, count(*) AS n FROM read_parquet('{path}') WHERE v > 0.99 GROUP BY g"
    assert_same(ds.collect(spill=True), duck.sql(sql))
