"""A sort feeding an order-sensitive aggregate must survive optimization.

`eliminate_sort_before_aggregate` deletes a `Sort` below a `GROUP BY` because grouping is
order-independent. Most aggregates are too — but not all:

* `list_agg` / `array_agg` collect their values **in arrival order**;
* `arg_min` / `arg_max` (and the `first`/`last` that lower to them) and `mode` break ties
  by arrival order.

Dropping the sort under those returns a different answer, not a faster plan. This file
pins each against DuckDB, and pins that the rewrite still fires for the order-independent
aggregates so the optimization is not lost.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col
from batcher.kyber.optimizer import optimize_logical
from batcher.plan.logical import Sort
from batcher.plan.visitor import walk


def _table() -> pa.Table:
    # Duplicate `x` values give `arg_min`/`arg_max`/`mode` real ties to break.
    return pa.table(
        {
            "g": ["a", "a", "a", "b", "b", "b", "b"],
            "x": [3, 1, 4, 1, 5, 1, 5],
            "y": [10, 20, 30, 40, 50, 60, 70],
        }
    )


@pytest.fixture
def data(duck):
    t = _table()
    duck.register("t", t)
    return bt.from_arrow(t)


def _sort_survives(dataset) -> bool:
    plan = optimize_logical(dataset._plan, sources=dataset._sources)
    return any(isinstance(n, Sort) for n in walk(plan))


@pytest.mark.differential
def test_array_agg_keeps_its_sort(data, duck):
    # Order-independent on purpose. The `GROUP BY` discards the sort's *row* order, so what
    # is under test is the order *inside* each group's list — which the value comparison
    # checks element by element. Pinning the row order too would only make the test brittle.
    got = data.sort("x").group_by("g").agg(l=col("x").array_agg()).collect()
    assert_same(got, duck.sql("select g, list(x order by x) as l from t group by g"))


@pytest.mark.differential
def test_array_agg_sort_is_not_eliminated(data):
    assert _sort_survives(data.sort("x").group_by("g").agg(l=col("x").array_agg()))


@pytest.mark.differential
def test_order_independent_aggregate_still_drops_its_sort(data, duck):
    # The optimization must not be lost for the aggregates that genuinely cannot see order.
    ds = data.sort("x").group_by("g").agg(s=col("x").sum(), n=col("x").count())
    assert not _sort_survives(ds)
    assert_same(ds.collect(), duck.sql("select g, sum(x) as s, count(x) as n from t group by g"))


@pytest.mark.differential
def test_mixed_aggregates_keep_the_sort_when_any_is_order_sensitive(data):
    ds = data.sort("x").group_by("g").agg(s=col("x").sum(), l=col("x").array_agg())
    assert _sort_survives(ds)


@pytest.mark.differential
def test_top_n_sort_is_never_eliminated(data, duck):
    # A `Sort` carrying a limit changes *which* rows are aggregated, not just their order.
    # Compared order-independently on purpose: the `GROUP BY` above discards the sort's
    # ordering, so what is under test is *which* three rows survived the limit, not the
    # order the groups come back in.
    got = data.sort("x", descending=True).limit(3).group_by("g").agg(s=col("x").sum()).collect()
    assert_same(
        got,
        duck.sql("select g, sum(x) as s from (select * from t order by x desc limit 3) group by g"),
    )
