"""Differential: float leaves nested inside a list/struct key must fold like top-level.

`crate::keys::canonicalize_float_keys` is the one canonical form every hash path (group
assign, shuffle, join, distinct, window partition) derives key identity from. It folded
`-0.0`/`0.0` and unified every NaN for a *top-level* `Float64` key, but not for a float
buried inside a `List`/`Struct` key — arrow's `RowConverter` splits `-0.0`/`0.0` at every
depth, so a `GROUP BY`/`JOIN`/`DISTINCT` on a list-of-floats or struct-with-float column
silently split one group into two (and dropped `-0.0 == 0.0` join matches). DuckDB folds
signed zero inside nested values, so these queries disagreed with the oracle.

Regression tests for that fix — every assertion fails on the pre-fix engine.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from conftest import assert_same


def test_group_by_list_of_float_folds_signed_zero(duck) -> None:
    """`GROUP BY` on a `list<double>` folds `[-0.0]` and `[0.0]` into one group."""
    table = pa.table(
        {
            "k": pa.array([[-0.0], [0.0], [1.5], [-0.0]], type=pa.list_(pa.float64())),
            "v": [1, 2, 3, 4],
        }
    )
    got = bt.from_arrow(table).group_by("k").agg(s=bt.col("v").sum()).collect()
    duck.register("t", table)
    assert_same(got, duck.sql("SELECT k, SUM(v) AS s FROM t GROUP BY k"))


def test_join_on_list_of_float_matches_signed_zero(duck) -> None:
    """A join keyed on a `list<double>` matches `[-0.0]` with `[0.0]`."""
    left = pa.table({"k": pa.array([[-0.0], [1.5]], type=pa.list_(pa.float64())), "l": [10, 20]})
    right = pa.table({"k": pa.array([[0.0], [2.5]], type=pa.list_(pa.float64())), "r": [30, 40]})
    got = bt.from_arrow(left).join(bt.from_arrow(right), on="k", how="inner").collect()
    duck.register("l", left)
    duck.register("r", right)
    assert_same(got, duck.sql("SELECT * FROM l JOIN r USING (k)"))


def test_group_by_list_of_float_unifies_nan(duck) -> None:
    """Every NaN bit-pattern inside a list key groups together, matching DuckDB.

    ``assert_same`` compares a NaN list key by equality (``[nan] != [nan]``), so this
    asserts the structural invariant directly: the two NaN patterns collapse to one group
    (sum 1+2=3) plus the 1.0 group (sum 3) — exactly what DuckDB produces.
    """
    import math

    nan1 = float("nan")
    nan2 = -float("nan")  # a different NaN bit pattern (sign bit set)
    table = pa.table(
        {
            "k": pa.array([[nan1], [nan2], [1.0]], type=pa.list_(pa.float64())),
            "v": [1, 2, 3],
        }
    )
    got = bt.from_arrow(table).group_by("k").agg(s=bt.col("v").sum()).collect().to_pydict()
    # Two groups: the unified-NaN group and the 1.0 group, each summing to 3.
    assert len(got["k"]) == 2, f"NaN patterns must unify to one group: {got}"
    nan_sum = next(s for k, s in zip(got["k"], got["s"]) if math.isnan(k[0]))
    one_sum = next(s for k, s in zip(got["k"], got["s"]) if not math.isnan(k[0]))
    assert nan_sum == 3, f"both NaN rows must fold together (1+2): {got}"
    assert one_sum == 3

    # DuckDB agrees on the count (the whole point of matching the oracle).
    duck.register("tn", table)
    assert duck.sql("SELECT COUNT(*) FROM (SELECT k FROM tn GROUP BY k)").fetchone()[0] == 2


def test_group_by_struct_with_float_field_folds_signed_zero(duck) -> None:
    """`GROUP BY` on a `struct<a:double>` folds the `-0.0`/`0.0` field into one group."""
    ty = pa.struct([("a", pa.float64())])
    table = pa.table(
        {
            "k": pa.array([{"a": -0.0}, {"a": 0.0}, {"a": 3.0}], type=ty),
            "v": [1, 2, 3],
        }
    )
    got = bt.from_arrow(table).group_by("k").agg(s=bt.col("v").sum()).collect()
    duck.register("ts", table)
    assert_same(got, duck.sql("SELECT k, SUM(v) AS s FROM ts GROUP BY k"))


def test_distinct_list_of_float_folds_signed_zero(duck) -> None:
    """`DISTINCT` on a `list<double>` collapses `[-0.0]` and `[0.0]` to one row."""
    table = pa.table({"f": pa.array([[-0.0], [0.0], [1.5], [1.5]], type=pa.list_(pa.float64()))})
    got = bt.from_arrow(table).distinct(["f"]).collect()
    duck.register("td", table)
    assert_same(got, duck.sql("SELECT DISTINCT f FROM td"))
