"""COUNT(DISTINCT) / n_unique aggregate differential tests vs DuckDB."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count


@pytest.fixture
def t(duck):
    rng = np.random.default_rng(0)
    tbl = pa.table(
        {
            "g": (np.arange(400) % 6).astype("int64"),
            # Few distinct values per group so DISTINCT actually collapses.
            "v": rng.integers(0, 20, 400).astype("int64"),
            "s": pa.array([f"k{i % 13}" for i in range(400)]),
        }
    )
    duck.register("t", tbl)
    return tbl


def test_count_distinct_grouped_vs_duckdb(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .group_by("g")
        .agg(nv=col("v").n_unique(), ns=col("s").n_unique(), n=count())
        .collect()
    )
    expected = duck.sql(
        "SELECT g, COUNT(DISTINCT v) nv, COUNT(DISTINCT s) ns, COUNT(*) n FROM t GROUP BY g"
    )
    assert_same(out, expected)


def test_count_distinct_global_vs_duckdb(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).group_by().agg(nv=col("v").n_unique()).collect()
    expected = duck.sql("SELECT COUNT(DISTINCT v) nv FROM t")
    assert_same(out, expected)


def test_count_distinct_via_sql(duck, t):
    from conftest import assert_same

    q = "SELECT g, COUNT(DISTINCT v) AS nv FROM t GROUP BY g"
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


def test_count_distinct_excludes_nulls_vs_duckdb(duck):
    from conftest import assert_same

    tbl = pa.table(
        {
            "g": pa.array([1, 1, 1, 2, 2], type=pa.int64()),
            "v": pa.array([5, None, 5, None, None], type=pa.int64()),
        }
    )
    duck.register("u", tbl)
    out = bt.from_arrow(tbl).group_by("g").agg(nv=col("v").n_unique()).collect()
    assert_same(out, duck.sql("SELECT g, COUNT(DISTINCT v) nv FROM u GROUP BY g"))


def test_count_distinct_partition_independent():
    # Mergeability: same result whether the input is one chunk or many.
    rng = np.random.default_rng(2)
    tbl = pa.table(
        {
            "g": (np.arange(300) % 4).astype("int64"),
            "v": rng.integers(0, 15, 300).astype("int64"),
        }
    )

    def run(ds):
        return ds.group_by("g").agg(nv=col("v").n_unique()).sort("g").collect().to_pylist()

    whole = run(bt.from_arrow(tbl))
    chunked = run(bt.from_arrow(tbl.to_batches(max_chunksize=37)))
    assert whole == chunked
