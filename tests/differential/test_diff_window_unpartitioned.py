"""An unpartitioned window must run, and must match SQL's bare ``OVER (ORDER BY ...)``.

``over(partition_by=None)`` is what a caller holds when the partition keys come from a
variable that may be unset, and it is the direct spelling of SQL's unpartitioned window.
It used to reach ``list(None)`` inside the plan builder and raise a bare
``TypeError: 'NoneType' object is not iterable`` — from the control plane, with nothing
naming the argument at fault.

Window results are positional, so every assertion here is ordered: `assert_same` is
order-independent and would pass on a correctly-valued but wrongly-ordered frame.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered

# `o` is globally unique, so a running frame is unambiguous with or without a partition.
ROWS = {
    "k": ["a", "a", "a", "b", "b", "c", "c", "c", "c", "a"],
    "o": [10, 21, 32, 11, 22, 43, 13, 24, 35, 5],
    "v": [1.0, None, 3.0, -1.0, 2.5, 0.0, 7.0, None, -2.0, 5.0],
}


@pytest.fixture
def t(duck):
    duck.register("t", pa.table(ROWS))


@pytest.mark.parametrize(
    ("name", "body"),
    [
        ("sum", "sum(v)"),
        ("mean", "avg(v)"),
        ("min", "min(v)"),
        ("max", "max(v)"),
        ("count", "count(v)"),
    ],
)
@pytest.mark.parametrize("keys", [None, (), []], ids=["none", "empty-tuple", "empty-list"])
def test_unpartitioned_running_aggregate_matches_duckdb(t, duck, name, body, keys):
    """`None` and the empty sequences all mean "no partition", and all agree with SQL."""
    w = getattr(bt.col("v"), name)().over(partition_by=keys, order_by="o")
    got = bt.from_pydict(ROWS).with_columns(y=w).sort("o").collect()
    assert_same_ordered(
        got,
        duck.sql(
            f"select k, o, v, {body} over (order by o "
            "rows between unbounded preceding and current row) as y from t order by o"
        ),
    )


def test_none_order_by_is_an_unordered_window(t, duck):
    """`order_by=None` means unordered — the whole partition, broadcast to every row."""
    w = bt.col("v").sum().over(partition_by="k", order_by=None)
    got = bt.from_pydict(ROWS).with_columns(y=w).sort("o").collect()
    assert_same_ordered(
        got,
        duck.sql("select k, o, v, sum(v) over (partition by k) as y from t order by o"),
    )


def test_both_none_is_a_whole_table_window(t, duck):
    """Both keys `None` is SQL's bare ``OVER ()`` — one value over the whole input."""
    w = bt.col("v").sum().over(partition_by=None, order_by=None)
    got = bt.from_pydict(ROWS).with_columns(y=w).sort("o").collect()
    assert_same_ordered(got, duck.sql("select k, o, v, sum(v) over () as y from t order by o"))


def test_value_function_accepts_none_keys(t, duck):
    """The value-function `over` (a different method) takes `None` on both keys too."""
    got = (
        bt.from_pydict(ROWS)
        .with_columns(y=bt.lag(bt.col("v"), 1).over(partition_by=None, order_by="o"))
        .sort("o")
        .collect()
    )
    assert_same_ordered(
        got, duck.sql("select k, o, v, lag(v,1) over (order by o) as y from t order by o")
    )
