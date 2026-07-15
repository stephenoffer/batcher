"""Aggregating an all-null (Arrow ``Null``-typed) column must match DuckDB, not error.

A column that is entirely null carries Arrow's ``Null`` data type (e.g. ``SELECT NULL AS x``,
or a ``from_pydict`` column of all ``None``). The typed accumulator kernels rejected it —
``sum``/``min``/``max``/``avg`` raised "aggregate `sum` is not supported for column type
Null" where DuckDB returns NULL, and ``count(x)`` returned the group size instead of 0.

``coerce_null_call_inputs`` substitutes an all-null ``Int64`` column at the aggregate input
boundary (the sibling of the sort-side ``coerce_null_sort_key``, ledger B215), so every
kernel behaves: sum/min/max/mean → NULL, count of non-null → 0.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from conftest import assert_same

pytestmark = pytest.mark.differential


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {"g": pa.array([1, 1, 2], pa.int64()), "x": pa.array([None, None, None], pa.null())}
    )
    duck.register("t", tbl)
    return tbl


def test_grouped_sum_of_all_null_column_is_null(duck, t):
    """``SUM``/``MIN``/``MAX``/``AVG`` over an all-null column are NULL per group, not an error."""
    out = (
        bt.from_arrow(t)
        .group_by("g")
        .agg(s=col("x").sum(), mn=col("x").min(), mx=col("x").max(), a=col("x").mean())
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT g, sum(x) s, min(x) mn, max(x) mx, avg(x) a FROM t GROUP BY g"
        ),
    )


def test_grouped_count_of_all_null_column_is_zero(duck, t):
    """``COUNT(x)`` over an all-null column is 0 per group (nulls ignored), not the row count."""
    out = bt.from_arrow(t).group_by("g").agg(c=col("x").count()).collect()
    assert_same(out, duck.sql("SELECT g, count(x) c FROM t GROUP BY g"))


def test_global_agg_of_all_null_column(duck, t):
    """The no-GROUP-BY path agrees too: global sum → NULL, count → 0."""
    out = bt.from_arrow(t).agg(s=col("x").sum(), c=col("x").count()).collect()
    assert_same(out, duck.sql("SELECT sum(x) s, count(x) c FROM t"))
