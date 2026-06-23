"""Variance/stddev aggregate differential tests vs DuckDB."""

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
            "g": (np.arange(300) % 5).astype("int64"),
            "v": rng.integers(0, 100, 300).astype("int64"),
            "f": rng.normal(0, 1, 300),
        }
    )
    duck.register("t", tbl)
    return tbl


def test_var_stddev_grouped_vs_duckdb(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .group_by("g")
        .agg(var=col("v").var(), sd=col("v").std(), fvar=col("f").var(), n=count())
        .collect()
    )
    expected = duck.sql(
        "SELECT g, var_samp(v) var, stddev_samp(v) sd, var_samp(f) fvar, COUNT(*) n "
        "FROM t GROUP BY g"
    )
    assert_same(out, expected)


def test_var_stddev_global_vs_duckdb(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).group_by().agg(var=col("v").var(), sd=col("v").std()).collect()
    expected = duck.sql("SELECT var_samp(v) var, stddev_samp(v) sd FROM t")
    assert_same(out, expected)


def test_var_stddev_via_sql(duck, t):
    from conftest import assert_same

    q = "SELECT g, var_samp(v) AS var, stddev_samp(v) AS sd FROM t GROUP BY g"
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


def test_var_stddev_partition_independent():
    # Mergeability: same result whether input is one chunk or many.
    rng = np.random.default_rng(1)
    tbl = pa.table(
        {"g": (np.arange(200) % 4).astype("int64"), "v": rng.integers(0, 50, 200).astype("int64")}
    )

    def run(ds):
        return ds.group_by("g").agg(var=col("v").var()).collect().to_pylist()

    def norm(rows):
        return sorted((r["g"], round(r["var"], 6)) for r in rows)

    one = run(bt.from_arrow(tbl.combine_chunks().to_batches()))
    many = run(bt.from_arrow(tbl.combine_chunks().to_batches(max_chunksize=16)))
    assert norm(one) == norm(many)
