"""`projection_scan` rewrites preserve results vs DuckDB (nulls + empty covered)."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
import batcher.kyber.rules.extra.projection_scan
from _harness import assert_same, assert_same_ordered
from batcher import col


def _data() -> pa.Table:
    return pa.table(
        {
            "x": [3, 1, 4, 1, 5, 9, 2, 6, 5, 3],
            "y": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }
    )


def _nonnull() -> pa.Table:
    schema = pa.schema([pa.field("x", pa.int64(), nullable=False), pa.field("y", pa.int64())])
    return pa.table({"x": [3, 1, 4, 1, 5], "y": [10, 20, 30, 40, 50]}, schema=schema)


# --- ordering-elimination rules ----------------------------------------------


def test_dedupe_sort_keys(duck):
    # Ordered on purpose: this rule rewrites a *sort*, so an order-independent comparison
    # would pass just as happily if deduping the keys had destroyed the ordering entirely.
    # (x, y) is a total order over this data, so there is no tie ambiguity to make it flaky.
    t = _data()
    duck.register("t", t)
    out = bt.from_arrow(t).sort("x", "y", "x").collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY x, y"))


def test_sort_before_full_sample(duck):
    """A sample of a sorted relation comes back sorted, full-fraction or not.

    This used to expect the *unordered* `SELECT * FROM t`, and it passed, because
    `eliminate_sort_before_sample` deleted the `Sort` beneath the `Sample`. That rewrite was
    justified on the sampled **multiset** being order-independent, which is true and is not
    sufficient: `Sample` is order-preserving in the engine, so deleting the sort changed the
    order of the rows the user asked for. The rule is gone (see
    `kyber/rules/extra/projection_scan.py`), so the sort now survives and the expectation is
    the sorted one.

    Compared against `ORDER BY x, y` rather than `ORDER BY x` because that is a total order
    over this data, so no tie-break ambiguity can make the comparison flaky.
    """
    t = _data()
    duck.register("t", t)
    out = bt.from_arrow(t).sort("x").sample(fraction=1.0).collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY x, y"))


# --- schema NOT-NULL null checks ---------------------------------------------


def test_drop_always_true_not_null(duck):
    t = _nonnull()
    duck.register("t", t)
    out = bt.from_arrow(t).filter(col("x").is_not_null()).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE x IS NOT NULL"))


def test_impossible_is_null_is_empty(duck):
    t = _nonnull()
    duck.register("t", t)
    out = bt.from_arrow(t).filter(col("x").is_null()).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE x IS NULL"))  # empty both sides


def test_not_null_on_empty_input(duck):
    schema = pa.schema([pa.field("x", pa.int64(), nullable=False), pa.field("y", pa.int64())])
    t = pa.table({"x": [], "y": []}, schema=schema)
    duck.register("t", t)
    out = bt.from_arrow(t).filter(col("x").is_not_null()).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE x IS NOT NULL"))


# --- sample bounds -----------------------------------------------------------


def test_sample_n_zero_is_empty(duck):
    t = _data()
    duck.register("t", t)
    out = bt.from_arrow(t).sample(n=0).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE 1 = 0"))


def test_full_fraction_sample_is_identity(duck):
    t = _data()
    duck.register("t", t)
    out = bt.from_arrow(t).sample(fraction=1.0).collect()
    assert_same(out, duck.sql("SELECT * FROM t"))


def test_nested_full_fraction_same_seed(duck):
    t = _data()
    duck.register("t", t)
    # Two full-fraction samples with the same seed compose to one (min = 1.0) → identity.
    out = bt.from_arrow(t).sample(fraction=1.0, seed=1).sample(fraction=1.0, seed=1).collect()
    assert_same(out, duck.sql("SELECT * FROM t"))


# --- projection / filter cleanups --------------------------------------------


def test_merge_projection_renames(duck):
    t = _data()
    duck.register("t", t)
    out = (
        bt.from_arrow(t)
        .select(a=col("x"), keep=col("y"))
        .select(p=col("a") + col("keep"), q=col("a") * 2)
        .collect()
    )
    assert_same(out, duck.sql("SELECT x + y AS p, x * 2 AS q FROM t"))


def test_drop_self_cast_in_projection(duck):
    t = _data()
    duck.register("t", t)
    out = bt.from_arrow(t).select(r=col("x").cast("int64")).collect()
    assert_same(out, duck.sql("SELECT CAST(x AS BIGINT) AS r FROM t"))


def test_drop_self_cast_in_filter(duck):
    t = _data()
    duck.register("t", t)
    out = bt.from_arrow(t).filter(col("x").cast("int64") > 2).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE CAST(x AS BIGINT) > 2"))


def test_drop_self_cast_in_sort_key(duck):
    # Ordered on purpose, for the same reason as `test_dedupe_sort_keys`: dropping the
    # no-op cast must leave the sort key intact, and only an ordered comparison can see that.
    # x is distinct here, so the order is total.
    t = pa.table({"x": [5, 3, 1, 4, 2], "y": [50, 30, 10, 40, 20]})
    duck.register("t", t)
    out = bt.from_arrow(t).sort(col("x").cast("int64")).select("x", "y").collect()
    assert_same_ordered(out, duck.sql("SELECT x, y FROM t ORDER BY x"))
