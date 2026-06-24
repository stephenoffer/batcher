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


def _skewed_with_nulls():
    """A hot key (75% of rows) plus ~10% null values and an all-null group — the
    shape that makes the exact per-group value list exceed a tight memory budget."""
    rng = np.random.default_rng(3)
    n = 20000
    k = np.concatenate([np.zeros(15000, "int64"), rng.integers(1, 50, n - 15000).astype("int64")])
    vals = rng.integers(0, 1000, n).astype("float64")
    v = pa.array(vals, mask=rng.random(n) < 0.1)  # ~10% NULL (Arrow nulls, not NaN)
    # An all-null group (key 99) whose median must come back NULL.
    k = np.concatenate([k, np.full(60, 99, "int64")])
    v = pa.concat_arrays([v, pa.array(np.zeros(60), mask=np.ones(60, bool))])
    return pa.table({"k": k, "v": v})


# A memory cap small enough that the per-group value list spills, forcing the bounded
# out-of-core median/quantile path (a hot key is computed on disk).
def _tight_cap():
    from batcher.config import Config, MemoryConfig

    return Config().replace(memory=MemoryConfig(max_memory_bytes=1 << 14))


def test_median_grouped_spilled(duck):
    from batcher.config import config_context
    from conftest import assert_same

    tbl = _skewed_with_nulls()
    duck.register("s", tbl)
    with config_context(_tight_cap()):
        out = bt.from_arrow(tbl).group_by("k").agg(m=col("v").median()).collect()
    assert_same(out, duck.sql("SELECT k, median(v) m FROM s GROUP BY k"))


@pytest.mark.parametrize("p", [0.1, 0.25, 0.5, 0.9])
def test_quantile_grouped_spilled(duck, p):
    from batcher.config import config_context
    from conftest import assert_same

    tbl = _skewed_with_nulls()
    duck.register("s", tbl)
    with config_context(_tight_cap()):
        out = bt.from_arrow(tbl).group_by("k").agg(q=col("v").quantile(p)).collect()
    assert_same(out, duck.sql(f"SELECT k, quantile_cont(v, {p}) q FROM s GROUP BY k"))


def test_median_global_spilled(duck):
    from batcher.config import config_context
    from conftest import assert_same

    tbl = _skewed_with_nulls()
    duck.register("s", tbl)
    with config_context(_tight_cap()):
        out = bt.from_arrow(tbl).group_by().agg(m=col("v").median()).collect()
    assert_same(out, duck.sql("SELECT median(v) m FROM s"))


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
