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


def _skewed_with_nulls():
    """A hot key (75% of rows) with many distinct values, ~10% nulls, and an all-null
    group — the shape that makes the exact per-group distinct set exceed a tight
    memory budget, forcing the bounded out-of-core n_unique path."""
    rng = np.random.default_rng(5)
    n = 20000
    k = np.concatenate([np.zeros(15000, "int64"), rng.integers(1, 50, n - 15000).astype("int64")])
    vals = rng.integers(0, 5000, n).astype("int64")  # high cardinality → big distinct sets
    v = pa.array(vals, mask=rng.random(n) < 0.1)  # ~10% NULL (excluded by COUNT DISTINCT)
    k = np.concatenate([k, np.full(60, 99, "int64")])  # all-null group → 0
    v = pa.concat_arrays([v, pa.array(np.zeros(60, "int64"), mask=np.ones(60, bool))])
    return pa.table({"k": k, "v": v})


def _tight_cap():
    # A memory cap small enough that the per-group distinct set spills, forcing the
    # bounded out-of-core n_unique path (a hot key is counted on disk).
    from batcher.config import Config, MemoryConfig

    return Config().replace(memory=MemoryConfig(max_memory_bytes=1 << 14))


def test_count_distinct_grouped_spilled(duck):
    from batcher.config import config_context
    from conftest import assert_same

    tbl = _skewed_with_nulls()
    duck.register("s", tbl)
    with config_context(_tight_cap()):
        out = bt.from_arrow(tbl).group_by("k").agg(nv=col("v").n_unique()).collect()
    assert_same(out, duck.sql("SELECT k, COUNT(DISTINCT v) nv FROM s GROUP BY k"))


def test_count_distinct_global_spilled(duck):
    from batcher.config import config_context
    from conftest import assert_same

    tbl = _skewed_with_nulls()
    duck.register("s", tbl)
    with config_context(_tight_cap()):
        out = bt.from_arrow(tbl).group_by().agg(nv=col("v").n_unique()).collect()
    assert_same(out, duck.sql("SELECT COUNT(DISTINCT v) nv FROM s"))


def test_count_distinct_strings_spilled(duck):
    # The native-type sorted-run path must be exact for non-numeric values too (an
    # f64 cast would collide strings), so a high-cardinality string column spills and
    # still matches DuckDB.
    from batcher.config import config_context
    from conftest import assert_same

    rng = np.random.default_rng(7)
    n = 20000
    k = np.concatenate([np.zeros(15000, "int64"), rng.integers(1, 40, n - 15000).astype("int64")])
    s = pa.array([f"u{i}" for i in rng.integers(0, 8000, n)])
    tbl = pa.table({"k": k, "s": s})
    duck.register("ss", tbl)
    with config_context(_tight_cap()):
        out = bt.from_arrow(tbl).group_by("k").agg(ns=col("s").n_unique()).collect()
    assert_same(out, duck.sql("SELECT k, COUNT(DISTINCT s) ns FROM ss GROUP BY k"))


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
