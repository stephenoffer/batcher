"""Kyber plans the same query the same way, whatever budget it is given.

Two properties, both cheap and both load-bearing:

*Determinism.* Planning the same query twice must give the same plan. A rule that reads a
set's iteration order, a dict built from an unordered scan, or anything derived from object
identity produces a plan that varies run to run -- so a benchmark measures different plans on
different runs, and a bug reproduces only sometimes.

*Confluence.* The rewrite phases run **to a fixpoint**, bounded by
`OptimizerConfig.fixpoint_iterations` (default 8). If the rules are confluent that bound never
binds and the plan is the same whatever it is set to. If two rules undo each other, iteration
stops wherever the budget ran out -- so the plan silently depends on a tuning knob, and on the
knob's *parity* at that.

**Note what this file deliberately does not assert, because the obvious version is worthless.**
"Optimizing an already-optimized plan changes nothing" is true by construction here: the driver
already runs those phases to a fixpoint, so a second call cannot rewrite anything, and the
assertion can never fail. The first draft of this file asserted exactly that, across twenty-two
shapes, and passed with `transpose_adjacent_windows`'s termination guard deleted from the
engine. Varying the *budget* is the version with teeth: with that same guard removed, the
window pair swaps forever, the optimizer logs "REWRITE phase did not reach a fixpoint in 64
iterations (a non-confluent rule?)", and budgets 2/3/9 produce a different plan from 8/64.

Both tests run with the **plan cache disabled**. That is not tidiness either: `optimize_logical`
is memoized on the plan's content key, so two identical queries hit the memo and compare equal
without the optimizer running twice -- which would make the determinism test vacuous in the
same way.

The comparison is on the logical plan's IR, so no physical costing or hardware detection enters
it and the test does not go red because the machine got busy.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher.config import Config

pytestmark = pytest.mark.unit


def _config(iterations: int) -> Config:
    """A config with the plan cache off and a chosen fixpoint budget."""
    base = Config()
    optimizer = dataclasses.replace(
        base.optimizer, plan_cache_entries=0, fixpoint_iterations=iterations
    )
    return dataclasses.replace(base, optimizer=optimizer)


#: Budgets the plan must be identical across. Includes both parities either side of the
#: default, because a pair of rules that undo each other lands on a different plan depending
#: on whether the budget is odd or even -- comparing two even budgets would miss it.
_BUDGETS = (2, 3, 8, 9, 64)


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    left = pa.table(
        {
            "k": ["a", "b", "a", "c"] * 20,
            "v": list(range(80)),
            "t": list(range(80)),
            "g": ["x", "x", "y", "y"] * 20,
        }
    )
    right = pa.table({"k": ["a", "b", "c"], "w": [10, 20, 30]})
    ldir = tmp_path_factory.mktemp("l")
    rdir = tmp_path_factory.mktemp("r")
    for part in range(4):
        pq.write_table(left, ldir / f"p{part}.parquet")
    pq.write_table(right, rdir / "r.parquet")
    return str(ldir), str(rdir)


def _build(name: str, ldir: str, rdir: str):
    ds = bt.read.parquet(ldir)
    rs = bt.read.parquet(rdir)
    builders = {
        "project": lambda: ds.select("k", "v"),
        "filter": lambda: ds.filter(bt.col("v") > 1),
        "with_columns": lambda: ds.with_columns(z=bt.col("v") * 2),
        "limit": lambda: ds.limit(2),
        "sort": lambda: ds.sort("v"),
        "topn": lambda: ds.sort("v").limit(2),
        "aggregate": lambda: ds.group_by("k").agg(s=bt.col("v").sum()),
        "distinct": lambda: ds.distinct(),
        "distinct_subset": lambda: ds.distinct(subset=["k"]),
        "join_inner": lambda: ds.join(rs, on="k"),
        "join_left": lambda: ds.join(rs, on="k", how="left"),
        "window_partitioned": lambda: ds.with_columns(
            r=bt.row_number().over(partition_by="k", order_by="v")
        ),
        "window_global": lambda: ds.with_columns(r=bt.row_number().over(order_by="v")),
        # The shape that motivates the fixpoint property: two windows the transpose rule
        # reorders, under the projection that rule inserts to preserve column order.
        "two_windows": lambda: ds.with_columns(
            a=bt.row_number().over(partition_by="k", order_by="v"),
            b=bt.col("v").sum().over(partition_by="g"),
        ),
        "row_index": lambda: ds.with_row_index("i"),
        "union": lambda: ds.union(bt.read.parquet(ldir)),
        "agg_then_sort": lambda: ds.group_by("k").agg(s=bt.col("v").sum()).sort("s"),
        "filter_then_agg": lambda: (
            ds.filter(bt.col("v") > 1).group_by("k").agg(s=bt.col("v").sum())
        ),
        "join_then_agg": lambda: ds.join(rs, on="k").group_by("k").agg(s=bt.col("v").sum()),
        "sort_then_window": lambda: ds.sort("v").with_columns(
            r=bt.row_number().over(partition_by="k", order_by="v")
        ),
        "deep_filters": lambda: (
            ds.filter(bt.col("v") > 1)
            .filter(bt.col("v") < 70)
            .select("k", "v")
            .filter(bt.col("k") != "c")
        ),
        "proj_chain": lambda: (
            ds.with_columns(a=bt.col("v") + 1).with_columns(b=bt.col("a") * 2).select("k", "b")
        ),
    }
    return builders[name]()


_SHAPES = (
    "agg_then_sort",
    "aggregate",
    "deep_filters",
    "distinct",
    "distinct_subset",
    "filter",
    "filter_then_agg",
    "join_inner",
    "join_left",
    "join_then_agg",
    "limit",
    "proj_chain",
    "project",
    "row_index",
    "sort",
    "sort_then_window",
    "topn",
    "two_windows",
    "union",
    "window_global",
    "window_partitioned",
    "with_columns",
)


@pytest.mark.parametrize("shape", _SHAPES)
def test_planning_the_same_query_twice_gives_the_same_plan(shape, sources):
    from batcher import kyber

    config = _config(8)
    first = _build(shape, *sources)
    second = _build(shape, *sources)
    a = kyber.optimize_logical(first._plan, config, first._sources)
    b = kyber.optimize_logical(second._plan, config, second._sources)
    assert a.to_ir() == b.to_ir(), (
        f"{shape} planned differently on two identical inputs -- some rule is reading an "
        "unordered container or object identity, so the plan varies run to run"
    )


@pytest.mark.parametrize("shape", _SHAPES)
def test_the_plan_does_not_depend_on_the_fixpoint_budget(shape, sources):
    from batcher import kyber

    plans = {}
    for budget in _BUDGETS:
        ds = _build(shape, *sources)
        plans[budget] = kyber.optimize_logical(ds._plan, _config(budget), ds._sources).to_ir()

    reference = plans[_BUDGETS[0]]
    differing = [b for b, ir in plans.items() if ir != reference]
    assert not differing, (
        f"{shape} plans differently at fixpoint_iterations={differing} than at "
        f"{_BUDGETS[0]} -- the rewrite phase is not reaching a fixpoint, so two rules are "
        "undoing each other and the plan depends on a tuning knob"
    )


def test_the_optimizer_actually_rewrites_something(sources):
    """Guard against a vacuous suite.

    Both properties above hold trivially of an optimizer that does nothing at all. This pins
    that at least one shape is genuinely rewritten, so what is being asserted is a property
    of real work.
    """
    from batcher import kyber

    ds = _build("deep_filters", *sources)
    optimized = kyber.optimize_logical(ds._plan, _config(8), ds._sources)
    assert optimized.to_ir() != ds._plan.to_ir(), (
        "the optimizer returned the input plan unchanged for a shape with three stacked "
        "filters and a projection -- if pushdown moved elsewhere, pick another shape"
    )
