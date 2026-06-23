"""Differential coverage for ``Expr.first(order_by=)`` / ``Expr.last(order_by=)``.

These lower to the deterministic, mergeable ``arg_min``/``arg_max`` aggregates: the
value at the first/last row in `order_by` order. DuckDB's ``first(x ORDER BY t)`` /
``last(x ORDER BY t)`` and ``arg_min``/``arg_max(x, t)`` are the references. Unique
order keys per group keep the result unambiguous.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def _t():
    return pa.table(
        {
            "g": ["a", "a", "a", "b", "b", "c"],
            "v": pa.array([10, 20, 30, 40, 50, 60], type=pa.int64()),
            "t": pa.array([1, 3, 2, 5, 4, 7], type=pa.int64()),
        }
    )


def test_first_last_grouped_matches_duckdb(duck):
    from conftest import assert_same

    duck.register("t", _t())
    out = (
        bt.from_arrow(_t())
        .group_by("g")
        .agg(f=col("v").first(order_by=col("t")), l=col("v").last(order_by=col("t")))
        .collect()
    )
    assert_same(
        out,
        duck.sql("SELECT g, arg_min(v, t) f, arg_max(v, t) l FROM t GROUP BY g"),
    )


def test_first_last_global_no_group(duck):
    from conftest import assert_same

    duck.register("t", _t())
    out = bt.from_arrow(_t()).agg(
        f=col("v").first(order_by=col("t")), l=col("v").last(order_by=col("t"))
    )
    assert_same(
        out.collect(),
        duck.sql("SELECT arg_min(v, t) f, arg_max(v, t) l FROM t"),
    )


def test_first_last_vs_sql_ordered_aggregate(duck):
    from conftest import assert_same

    duck.register("t", _t())
    out = bt.from_arrow(_t()).group_by("g").agg(f=col("v").first(order_by=col("t"))).collect()
    assert_same(out, duck.sql("SELECT g, first(v ORDER BY t) f FROM t GROUP BY g"))
