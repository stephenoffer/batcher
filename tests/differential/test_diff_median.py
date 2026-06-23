"""MEDIAN aggregate (exact, mergeable list-state) differential tests vs DuckDB."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


@pytest.fixture
def t(duck):
    rng = np.random.default_rng(0)
    tbl = pa.table(
        {
            "k": (np.arange(400) % 5).astype("int64"),
            "v": rng.integers(0, 100, 400).astype("int64"),
            "f": rng.normal(0, 1, 400),
        }
    )
    duck.register("t", tbl)
    return tbl


def test_median_grouped(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).group_by("k").agg(m=col("v").median(), mf=col("f").median()).collect()
    assert_same(out, duck.sql("SELECT k, median(v) m, median(f) mf FROM t GROUP BY k"))


def test_median_global(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).group_by().agg(m=col("v").median()).collect()
    assert_same(out, duck.sql("SELECT median(v) m FROM t"))


def test_median_even_and_odd_counts(duck):
    from conftest import assert_same

    tbl = pa.table({"k": [1, 1, 1, 1, 2, 2, 2], "v": [4, 2, 3, 1, 3, 1, 2]})
    duck.register("e", tbl)
    out = bt.from_arrow(tbl).group_by("k").agg(m=col("v").median()).collect()
    assert_same(out, duck.sql("SELECT k, median(v) m FROM e GROUP BY k"))


def test_median_via_sql(duck, t):
    from conftest import assert_same

    q = "SELECT k, median(v) AS m FROM t GROUP BY k"
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


def test_median_partition_independent():
    # Mergeability: same result whether input is one chunk or many.
    rng = np.random.default_rng(2)
    tbl = pa.table(
        {"k": (np.arange(300) % 4).astype("int64"), "v": rng.integers(0, 50, 300).astype("int64")}
    )

    def run(ds):
        return ds.group_by("k").agg(m=col("v").median()).collect().to_pylist()

    def norm(rows):
        return sorted((r["k"], r["m"]) for r in rows)

    one = run(bt.from_arrow(tbl.combine_chunks().to_batches()))
    many = run(bt.from_arrow(tbl.combine_chunks().to_batches(max_chunksize=16)))
    assert norm(one) == norm(many)
