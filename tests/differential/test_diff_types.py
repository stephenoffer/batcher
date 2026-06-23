"""Narrow numeric types are normalized at ingestion and behave like wide ones."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "k": pa.array([1, 2, 1, 2, 3], pa.int32()),
            "a": pa.array([10, 20, 30, 40, 50], pa.int16()),
            "b": pa.array([1, 2, 3, 4, 5], pa.int8()),
            "f": pa.array([1.5, 2.5, 3.5, 4.5, 5.5], pa.float32()),
            "u": pa.array([100, 200, 300, 400, 500], pa.uint32()),
        }
    )
    duck.register("t", tbl)
    return tbl


def test_narrow_groupby_agg(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .group_by("k")
        .agg(s=col("a").sum(), af=col("f").mean(), n=count(), mx=col("u").max())
        .collect()
    )
    assert_same(
        out, duck.sql("SELECT k, SUM(a) s, AVG(f) af, COUNT(*) n, MAX(u) mx FROM t GROUP BY k")
    )


def test_narrow_filter_projection(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).filter(col("u") > 200).select(r=col("a") * col("k") + col("b")).collect()
    assert_same(out, duck.sql("SELECT a*k + b AS r FROM t WHERE u > 200"))


def test_narrow_join(duck, t):
    from conftest import assert_same

    dim = pa.table({"k": pa.array([1, 2, 3], pa.int32()), "name": ["x", "y", "z"]})
    duck.register("dim", dim)
    out = bt.from_arrow(t).join(bt.from_arrow(dim), on="k").select("k", "name", "a").collect()
    assert_same(out, duck.sql("SELECT t.k, dim.name, t.a FROM t JOIN dim USING(k)"))
