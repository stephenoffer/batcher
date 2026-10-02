"""A global window ordered by an *expression* must split into ordered buckets, and agree.

A window with no `PARTITION BY` is cut by the range partitioner on its leading `ORDER BY`
key, and that partitioner reads the key's values from a column. `order_by=[col("a") * 2 +
col("b")]` has no column, so the ordered-bucket route declined it: `collect(spill=True)` and
`iter_batches()` fell back to windowing the whole relation in memory, and on distributed data
`collect(distributed=True)` raised `PlanError` (audit finding F092). The fix hoists the leading
key into a hidden column below the window and projects it away above, the rewrite a computed
sort key and a computed partition key already took (`plan.logical.hoist_window_keys`).

Every case is checked against DuckDB, on each path that cuts the window: the in-memory
`collect()`, the spilled `collect(spill=True)` at two bucket counts, and `iter_batches()`.
The order key is built from a unique `rid`, or the functions are peer-deterministic
(`rank`, `dense_rank`, the running folds), so the window-tie exception in
`.claude/rules/python-control-plane.md` cannot make a correct answer look wrong.

`assert_same` is order-independent, which is right here: a window's output is an unordered
relation and the comparison keys every row by `rid`.
"""

from __future__ import annotations

import duckdb
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_N = 3000


def _rows(n: int) -> pa.Table:
    return pa.table(
        {
            "rid": pa.array(range(n), pa.int64()),
            "a": pa.array([(i * 37) % 101 for i in range(n)], pa.int64()),
            "b": pa.array([None if i % 11 == 0 else (i * 13) % 7 for i in range(n)], pa.int64()),
            "v": pa.array([None if i % 9 == 0 else float(i % 29) - 14.0 for i in range(n)]),
        }
    )


#: (batcher window builder, the DuckDB `OVER` clause it must equal). Each order key is an
#: expression, so none of these reach the ordered-bucket route without the hoist.
_CASES = {
    "row_number_unique_expr": (
        lambda ds: ds.with_columns(w=bt.row_number().over(order_by=[bt.col("rid") * -3 + 7])),
        "row_number() OVER (ORDER BY rid * -3 + 7)",
    ),
    "rank_dup_expr": (
        lambda ds: ds.with_columns(w=bt.rank().over(order_by=[bt.col("a") % 17])),
        "rank() OVER (ORDER BY a % 17)",
    ),
    "dense_rank_dup_expr": (
        lambda ds: ds.with_columns(w=bt.dense_rank().over(order_by=[bt.col("a") + bt.col("a")])),
        "dense_rank() OVER (ORDER BY a + a)",
    ),
    "running_sum_expr": (
        lambda ds: ds.with_columns(w=bt.col("v").sum().over(order_by=[bt.col("a") * 2 + 1])),
        "sum(v) OVER (ORDER BY a * 2 + 1)",
    ),
    "running_count_expr_then_column": (
        lambda ds: ds.with_columns(
            w=bt.col("v").count().over(order_by=[bt.col("a") - 50, bt.col("rid")])
        ),
        "count(v) OVER (ORDER BY a - 50, rid)",
    ),
}


def _duck(rows: pa.Table, over: str):
    con = duckdb.connect()
    con.register("t", rows)
    return con.sql(f"SELECT rid, a, b, v, {over} AS w FROM t")


@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize("path", ["collect", "spill_2", "spill_7", "iter_batches"])
def test_a_computed_global_order_key_matches_duckdb(case, path):
    rows = _rows(_N)
    build, over = _CASES[case]
    ds = build(bt.from_arrow(rows))
    if path == "collect":
        got = ds.collect()
    elif path.startswith("spill_"):
        got = ds.collect(spill=True, num_partitions=int(path.split("_")[1]))
    else:
        got = pa.Table.from_batches(list(ds.iter_batches()))
    assert got.column_names == ["rid", "a", "b", "v", "w"]
    assert_same(got, _duck(rows, over))


@pytest.mark.parametrize("n", [0, 1])
def test_empty_and_single_row(n):
    rows = _rows(n)
    build, over = _CASES["rank_dup_expr"]
    ds = build(bt.from_arrow(rows))
    for got in (ds.collect(), ds.collect(spill=True, num_partitions=3)):
        assert got.column_names == ["rid", "a", "b", "v", "w"]
        assert_same(got, _duck(rows, over))


def test_the_rewrite_reaches_the_ordered_bucket_route():
    """The positive control for the tests above: without it they would pass on a fallback.

    The in-memory kernel computes these windows correctly, so the answers alone cannot show
    that the bounded route ran. This pins the route itself: the raw plan is declined, its
    hoisted form is admitted, and the hidden key is gone from what the caller sees.
    """
    from batcher.dist.global_window import supports_ordered_bucket_offsets
    from batcher.plan.logical import hoist_window_keys

    plan = _CASES["rank_dup_expr"][0](bt.from_arrow(_rows(10)))._plan
    window = plan if type(plan).__name__ == "Window" else plan.input
    assert supports_ordered_bucket_offsets(window) is False
    hoisted, keep = hoist_window_keys(window)
    assert supports_ordered_bucket_offsets(hoisted) is True
    assert all(not c.startswith("__") for c in keep)
    assert hoisted.order_keys[0].expr.name.startswith("__win_order")
