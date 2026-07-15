"""Differential (vs DuckDB) regressions for plan-layer bug-hunt fixes.

- A mid-expression alias buried inside a projection must not be pruned by the
  projection/pushdown rules (the `walk.referenced_columns` `Aliased` omission made
  the optimizer drop the underlying column and fail with "unknown column").
- The `list_max`/`list_min` etc. dtype fidelity check lives with the schema-lie
  unit tests; here we pin the end-to-end correctness of the alias fix.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def t(duck):
    tbl = pa.table({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40], "k": [1, 1, 2, 2]})
    duck.register("t", tbl)
    return tbl


def test_nested_alias_not_pruned(duck, t):
    from conftest import assert_same

    inner = bt.from_arrow(t).select((bt.col("a") + bt.col("b")).alias("t"), bt.col("b"))
    out = inner.select((bt.col("t").alias("z") + bt.lit(1)).alias("r")).collect()
    assert_same(out, duck.sql("SELECT (a + b) + 1 AS r FROM t"))


def test_list_max_preserves_int_precision(duck):
    # Fixed (ledger B222): `list.max`/`list.min` over an int64 list gather the exact element
    # instead of routing through f64, so a value above 2^53 survives. `list.sum` still routes
    # through f64 (a separate, deliberately-deferred return-dtype decision — see the findings).
    from conftest import assert_same_ordered

    big = 9007199254740993  # 2**53 + 1, not representable as float64
    tbl = pa.table({"l": pa.array([[big, 1]], pa.list_(pa.int64()))})
    duck.register("li", tbl)
    out = bt.from_arrow(tbl).select(m=bt.col("l").list.max()).collect()
    assert_same_ordered(out, duck.sql("SELECT list_max(l) AS m FROM li"))


def test_nested_alias_survives_join_and_filter(duck):
    from conftest import assert_same

    left = pa.table({"k": [1, 2, 3], "x": [5, 6, 7]})
    right = pa.table({"k": [1, 2, 3], "y": [70, 80, 90]})
    duck.register("l", left)
    duck.register("r", right)
    out = (
        bt.from_arrow(left)
        .join(bt.from_arrow(right), on="k")
        .filter((bt.col("x").alias("q")) > bt.lit(5))
        .select("k", "x", "y")
        .collect()
    )
    assert_same(
        out,
        duck.sql("SELECT l.k, x, y FROM l JOIN r USING (k) WHERE x > 5"),
    )
