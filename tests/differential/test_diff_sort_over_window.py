"""A `sort` above a window must survive optimization and order the rows, on every scheduling.

The estimator used to hand a window's input ordering straight through, so a
`with_row_index(...)` below a window claimed the window's output was ordered by the row
index and `sort_elimination_from_ordering` deleted the `sort` above it. That holds only for
the in-memory window kernel, which scatters results back to input positions. A spilled
partitioned window grace-partitions by its `PARTITION BY` keys and a spilled global window
streams range buckets of its `ORDER BY` key, and both emit rows in bucket order. Whether a
run spills is Carbonite's call on measured memory pressure, made after Kyber, so the same
plan could come back ordered on one run and reordered on the next in the same process.
Reported as `[5, 0, 1, 2, 3, 4]` from a `sort("_row")`.

Every comparison here is order-sensitive (`assert_same_ordered`). The multiset was always
right, so `assert_same` would pass on the bug.
"""

from __future__ import annotations

from collections.abc import Callable

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered
from batcher import col

#: 60 rows puts several partitions in every spill bucket, which is what reorders them.
_N = 60

_SCHEDULES: dict[str, Callable[[bt.Dataset], pa.Table]] = {
    "collect": lambda ds: ds.collect(),
    "iter_batches": lambda ds: pa.Table.from_batches(list(ds.iter_batches())),
    "num_partitions_4": lambda ds: ds.collect(num_partitions=4),
    "spill": lambda ds: ds.collect(spill=True),
    "spill_num_partitions_4": lambda ds: ds.collect(spill=True, num_partitions=4),
}


def _columns() -> dict[str, list]:
    x = [None if i % 17 == 0 else (i * 7) % 11 for i in range(_N)]
    return {"x": x, "g": [i % 4 for i in range(_N)]}


@pytest.fixture
def base(duck) -> bt.Dataset:
    cols = _columns()
    duck.register("t", pa.table({**cols, "_row": pa.array(range(_N), pa.int64())}))
    return bt.from_arrow(pa.table(cols)).with_row_index("_row")


@pytest.mark.parametrize("schedule", list(_SCHEDULES))
def test_the_reported_two_query_shape_comes_back_in_row_order(duck, base, schedule):
    """A global window over the source first, then a partitioned one, each sorted by `_row`."""
    run = _SCHEDULES[schedule]
    first = base.with_columns(p=col("x").shift(1).over(order_by=["_row"])).sort("_row")
    assert_same_ordered(
        run(first),
        duck.sql("SELECT x, g, _row, lag(x) OVER (ORDER BY _row) AS p FROM t ORDER BY _row"),
    )
    second = base.with_columns(f=col("x").is_first_distinct(col("_row"))).sort("_row")
    assert_same_ordered(
        run(second),
        duck.sql(
            "SELECT x, g, _row, row_number() OVER (PARTITION BY x ORDER BY _row) = 1 AS f "
            "FROM t ORDER BY _row"
        ),
    )


@pytest.mark.parametrize("schedule", list(_SCHEDULES))
def test_a_sort_above_a_partitioned_window_orders_the_rows(duck, base, schedule):
    ds = base.with_columns(
        s=col("x").sum().over(partition_by=["g"], order_by=["_row"]),
    ).sort("_row")
    assert_same_ordered(
        _SCHEDULES[schedule](ds),
        duck.sql(
            "SELECT x, g, _row, sum(x) OVER (PARTITION BY g ORDER BY _row) AS s "
            "FROM t ORDER BY _row"
        ),
    )


@pytest.mark.parametrize("schedule", list(_SCHEDULES))
def test_a_sort_above_a_global_window_on_another_key_orders_the_rows(duck, base, schedule):
    """A global window streams buckets of its own `ORDER BY` key, which is not `_row`."""
    ds = base.with_columns(r=col("_row").shift(1).over(order_by=["g", "_row"])).sort("_row")
    assert_same_ordered(
        _SCHEDULES[schedule](ds),
        duck.sql("SELECT x, g, _row, lag(_row) OVER (ORDER BY g, _row) AS r FROM t ORDER BY _row"),
    )


def test_the_sort_is_still_in_the_optimized_plan(base):
    """The plan-level cause, checked against a positive control that the sort *can* go."""
    windowed = base.with_columns(f=col("x").is_first_distinct(col("_row"))).sort("_row")
    assert "sort" in windowed.explain()
    assert "sort" not in base.filter(col("g") > 0).sort("_row").explain(), (
        "positive control: a sort over an order-preserving filter of the row index is elided"
    )
