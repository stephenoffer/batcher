"""``.over(...)`` on any expression, not only on aggregates and window functions.

A composite expression binds every aggregate and window inside it, a window function merges
the outer keys with its own, and a row-level expression is unchanged. `descending` and
`nulls_last` order the keys, `first`/`last` take their order from the window, and the two
mapping strategies the `Window` operator cannot compute are refused.

DuckDB's window SQL is the oracle; Polars 1.40 is checked on the shapes where the two agree.
Results are sorted by a unique row id before comparison, so row order is compared exactly.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.differential

_TABLES = {
    "nulls": pa.table(
        {
            "id": pa.array(range(8), pa.int64()),
            "g": pa.array(["a", "a", "b", "b", "b", None, None, "a"]),
            "t": pa.array([3, 1, None, 2, 2, 5, 4, 2], pa.int64()),
            "x": pa.array([1.0, None, 4.0, -0.0, 2.0, 7.0, None, 3.0]),
        }
    ),
    "empty": pa.table(
        {
            "id": pa.array([], pa.int64()),
            "g": pa.array([], pa.string()),
            "t": pa.array([], pa.int64()),
            "x": pa.array([], pa.float64()),
        }
    ),
    "one_row": pa.table({"id": [0], "g": ["a"], "t": [1], "x": [2.5]}),
    "duplicates": pa.table(
        {
            "id": [0, 1, 2, 3],
            "g": ["a", "a", "a", "a"],
            "t": [1, 1, 1, 2],
            "x": [2.0, 2.0, 2.0, 2.0],
        }
    ),
}


@pytest.fixture(params=sorted(_TABLES))
def table(request) -> pa.Table:
    return _TABLES[request.param]


def _ours(table: pa.Table, **exprs) -> pa.Table:
    # Ordered by pyarrow, not the engine: an engine sort above a partitioned window is
    # eliminated on a source an earlier query in the process read (a Kyber defect reproduced
    # at the base commit), and this file tests the windows rather than that.
    return bt.from_arrow(table).select("id", **exprs).collect().sort_by("id")


def test_composite_expression_binds_each_aggregate(duck, table):
    duck.register("t", table)
    ours = _ours(
        table,
        share=(bt.col("x") / bt.col("x").sum()).over("g"),
        centered=(bt.col("x") - bt.col("x").mean()).over(["g"]),
        same=(bt.col("x") + 1).over("g"),
    )
    expected = duck.sql(
        "SELECT id, x / sum(x) OVER (PARTITION BY g) AS share, "
        "x - avg(x) OVER (PARTITION BY g) AS centered, x + 1 AS same FROM t ORDER BY id"
    )
    assert_same_ordered(ours, expected)


def test_a_composed_aggregate_takes_the_whole_window(duck, table):
    """`sum(empty_value=0)` and `max(nan_policy="ignore")` wrap their aggregates in a scalar;
    every aggregate inside takes the partition, order and frame."""
    duck.register("t", table)
    x = bt.col("x")
    ours = _ours(
        table,
        filled=x.sum(empty_value=0).over(
            partition_by=["g"], order_by=["t", "id"], frame=(None, None)
        ),
        nan_max=x.max(nan_policy="ignore").over(partition_by=["g"]),
        root=x.abs().sum().sqrt().over("g"),
    )
    whole = "PARTITION BY g ORDER BY t, id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING"
    expected = duck.sql(
        f"SELECT id, coalesce(sum(x) OVER ({whole}), 0.0) AS filled, "
        "coalesce(max(CASE WHEN NOT isnan(x) THEN x END) OVER (PARTITION BY g), "
        "max(x) OVER (PARTITION BY g)) AS nan_max, "
        "sqrt(sum(abs(x)) OVER (PARTITION BY g)) AS root FROM t ORDER BY id"
    )
    assert_same_ordered(ours, expected)


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
def test_order_dependent_windows_under_over(duck, table, descending, nulls_last):
    duck.register("t", table)
    direction = "DESC" if descending else "ASC"
    nulls = "NULLS LAST" if nulls_last else "NULLS FIRST"
    order = {"order_by": ["t", "id"], "descending": descending, "nulls_last": nulls_last}
    ours = _ours(
        table,
        prev=bt.col("x").shift(1).over("g", **order),
        running=bt.col("x").cum_sum().over("g", **order),
        rn=bt.row_number().over("g", **order),
    )
    w = f"PARTITION BY g ORDER BY t {direction} {nulls}, id {direction} {nulls}"
    expected = duck.sql(
        f"SELECT id, lag(x) OVER ({w}) AS prev, "
        f"sum(x) OVER ({w} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running, "
        f"row_number() OVER ({w}) AS rn FROM t ORDER BY id"
    )
    assert_same_ordered(ours, expected)


@pytest.mark.parametrize("ignore_nulls", [True, False])
@pytest.mark.parametrize("descending", [False, True])
def test_first_and_last_take_their_order_from_the_window(duck, table, descending, ignore_nulls):
    """`first`/`last` over a window pick the row the grouped aggregate picks, nulls included."""
    duck.register("t", table)
    ours = _ours(
        table,
        f=bt.col("x")
        .first(ignore_nulls=ignore_nulls)
        .over("g", order_by="t", descending=descending),
        l=bt.col("x")
        .last(ignore_nulls=ignore_nulls)
        .over("g", order_by="t", descending=descending),
    )
    key = bt.col("t") * (-1 if descending else 1)
    grouped = (
        bt.from_arrow(table)
        .group_by("g")
        .agg(
            f=bt.col("x").first(key, ignore_nulls=ignore_nulls),
            l=bt.col("x").last(key, ignore_nulls=ignore_nulls),
        )
    )
    joined = bt.from_arrow(table).select("id", "g").join(grouped, on="g", how="left")
    # A null group key never equals itself in a join, so compare those rows separately.
    expected = joined.select("id", "f", "l").collect().sort_by("id")
    non_null = [i for i, g in enumerate(table.column("g").to_pylist()) if g is not None]
    assert [ours.column(c).to_pylist()[i] for c in ("f", "l") for i in non_null] == [
        expected.column(c).to_pylist()[i] for c in ("f", "l") for i in non_null
    ]


def test_rank_keeps_its_own_order_and_adds_the_partition(duck, table):
    duck.register("t", table)
    ours = _ours(table, r=bt.col("x").rank(descending=True).over("g"))
    expected = duck.sql(
        "SELECT id, rank() OVER (PARTITION BY g ORDER BY x DESC NULLS LAST) AS r FROM t ORDER BY id"
    )
    assert_same_ordered(ours, expected)


def test_an_outer_order_replaces_the_inner_one(duck, table):
    """``col("x").rank().over(order_by="t")`` ranks by ``t``: the outer order wins."""
    duck.register("t", table)
    ours = _ours(
        table,
        r=bt.col("x").rank().over("g", order_by=["t", "id"]),
        f=bt.col("x").first("id").over("g", order_by="t"),
    )
    grouped = bt.from_arrow(table).group_by("g").agg(f=bt.col("x").first("t"))
    expected_first = dict(zip(*grouped.to_pydict().values(), strict=True))
    expected = duck.sql(
        "SELECT id, rank() OVER (PARTITION BY g ORDER BY t, id) AS r FROM t ORDER BY id"
    )
    assert ours.column("r").to_pylist() == [row[1] for row in expected.fetchall()]
    groups = table.column("g").to_pylist()
    for got, g in zip(ours.column("f").to_pylist(), groups, strict=True):
        if g is not None:
            assert got == expected_first[g]


@pytest.mark.parametrize("strategy", ["join", "explode"])
def test_unsupported_mapping_strategies_are_refused(strategy):
    with pytest.raises(PlanError, match="group_to_rows"):
        bt.col("x").sum().over("g", mapping_strategy=strategy)
    with pytest.raises(PlanError, match="mapping_strategy must be one of"):
        bt.col("x").sum().over("g", mapping_strategy="bogus")


def test_polars_agrees_on_partitioned_shapes(table):
    pl = pytest.importorskip("polars")
    frame = pl.from_arrow(table).filter(pl.col("g").is_not_null() & pl.col("x").is_not_null())
    theirs = frame.select(
        "id",
        share=(pl.col("x") / pl.col("x").sum()).over("g"),
        prev=pl.col("x").shift(1).over("g", order_by=["t", "id"]),
        same=(pl.col("x") * 2).over("g"),
    ).sort("id")
    ours = (
        bt.from_arrow(table)
        .filter(bt.col("g").is_not_null() & bt.col("x").is_not_null())
        .select(
            "id",
            share=(bt.col("x") / bt.col("x").sum()).over("g"),
            prev=bt.col("x").shift(1).over("g", order_by=["t", "id"], nulls_last=False),
            same=(bt.col("x") * 2).over("g"),
        )
        .collect()
        .sort_by("id")
    )
    assert ours.to_pydict() == theirs.to_dict(as_series=False)
