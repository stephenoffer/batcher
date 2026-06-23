"""Differential coverage for `Dataset.pivot` (SQL PIVOT / pivot_table) vs DuckDB."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential


def _long():
    # Note: (b, y) and (c, x) combinations are absent → pivot fills them with NULL.
    return pa.table(
        {
            "idx": pa.array(["a", "a", "b", "b", "c"]),
            "k": pa.array(["x", "y", "x", "x", "y"]),
            "v": pa.array([1, 2, 3, 5, 9], type=pa.int64()),
        }
    )


def test_pivot_sum_matches_duckdb(duck):
    from conftest import assert_same

    out = bt.from_arrow(_long()).pivot(index=["idx"], on="k", values="v").collect()
    duck.register("t", _long())
    assert_same(out, duck.sql("PIVOT t ON k USING sum(v) GROUP BY idx"))


def test_pivot_explicit_columns(duck):
    from conftest import assert_same

    out = (
        bt.from_arrow(_long())
        .pivot(index=["idx"], on="k", values="v", columns=["x", "y"])
        .collect()
    )
    duck.register("t", _long())
    assert_same(out, duck.sql("PIVOT t ON k IN ('x', 'y') USING sum(v) GROUP BY idx"))


def test_pivot_count(duck):
    from conftest import assert_same

    out = (
        bt.from_arrow(_long()).pivot(index=["idx"], on="k", values="v", aggregate="count").collect()
    )
    duck.register("t", _long())
    assert_same(out, duck.sql("PIVOT t ON k USING count(v) GROUP BY idx"))
