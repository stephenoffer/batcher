"""Plan-shape unit tests for `sort_elimination_from_ordering`."""

from __future__ import annotations

import batcher as bt
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.ordering import sort_elimination_from_ordering
from batcher.plan.logical import Scan, Sort


def _t():
    return bt.from_pydict({"x": [3, 1, 2], "y": [30, 10, 20]})


def _num_sorts(ir: dict) -> int:
    n = 1 if ir.get("op") == "sort" else 0
    for v in ir.values():
        if isinstance(v, dict):
            n += _num_sorts(v)
        elif isinstance(v, list):
            n += sum(_num_sorts(i) for i in v if isinstance(i, dict))
    return n


def _ctx():
    # No bound sources: a Scan's order is unknown, but a lower Sort still establishes
    # ordering structurally — enough for these unit cases.
    return Optimizer()._context()


def test_rule_registered():
    assert "sort_elimination_from_ordering" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_redundant_resort_eliminated():
    plan = _t().sort("x").sort("x")._plan
    ir = Optimizer().optimize(plan).ir
    assert _num_sorts(ir) == 1  # the outer, redundant sort is gone
    assert ir["op"] == "sort" and ir["input"]["op"] == "scan"


def test_coarser_resort_is_prefix_eliminated():
    # Sorted by (x, y); re-sorting by x alone is redundant (x is a prefix).
    plan = _t().sort("x", "y").sort("x")._plan
    ir = Optimizer().optimize(plan).ir
    assert _num_sorts(ir) == 1
    keys = [k["expr"]["name"] for k in ir["keys"]]
    assert keys == ["x", "y"]  # the surviving sort is the finer (x, y) one


def test_finer_resort_not_eliminated():
    # Sorted by x only; re-sorting by (x, y) needs the extra key — keep both.
    plan = _t().sort("x").sort("x", "y")._plan
    ir = Optimizer().optimize(plan).ir
    assert _num_sorts(ir) == 2


def test_descending_resort_not_eliminated():
    plan = _t().sort("x").sort("x", descending=True)._plan
    ir = Optimizer().optimize(plan).ir
    assert _num_sorts(ir) == 2


def test_unknown_order_is_noop():
    plan = _t().sort("x")._plan
    assert isinstance(plan, Sort) and isinstance(plan.input, Scan)
    assert sort_elimination_from_ordering(plan, _ctx()) is None


def test_topn_sort_not_eliminated():
    inner = _t().sort("x")._plan
    topn = Sort(inner, inner.keys, limit=2)
    assert sort_elimination_from_ordering(topn, _ctx()) is None


# --- direction-aware orderings ------------------------------------------------------------
#
# `RelStats.sorted_by` carries each key's direction, so a descending ordering is tracked and
# consumed exactly as an ascending one is. Before that, `ORDER BY ts DESC` delivered no
# ordering at all and none of these cases could fire.


def test_redundant_descending_resort_eliminated():
    plan = _t().sort("x", descending=True).sort("x", descending=True)._plan
    ir = Optimizer().optimize(plan).ir
    assert _num_sorts(ir) == 1
    assert ir["keys"][0]["descending"] is True  # and it is the descending one that survived


def test_coarser_descending_resort_is_prefix_eliminated():
    plan = _t().sort("x", "y", descending=True).sort("x", descending=True)._plan
    ir = Optimizer().optimize(plan).ir
    assert _num_sorts(ir) == 1
    assert [k["expr"]["name"] for k in ir["keys"]] == ["x", "y"]


def test_ascending_resort_over_a_descending_input_is_not_eliminated():
    """The mirror of `test_descending_resort_not_eliminated`: opposite directions are
    different orderings, and neither satisfies the other."""
    plan = _t().sort("x", descending=True).sort("x")._plan
    ir = Optimizer().optimize(plan).ir
    assert _num_sorts(ir) == 2


def test_descending_ordering_survives_a_projection():
    """A projection renames columns and reorders nothing, so the direction rides through."""
    ds = _t().sort("x", descending=True).select(bt.col("x").alias("k"))
    plan = ds.sort("k", descending=True)._plan
    ir = Optimizer().optimize(plan).ir
    assert _num_sorts(ir) == 1


def test_nulls_first_resort_over_a_nulls_last_input_is_not_eliminated():
    """Null placement is part of the ordering wherever a null can actually appear."""
    plan = _t().sort("x").sort("x", nulls_first=True)._plan
    ir = Optimizer().optimize(plan).ir
    assert _num_sorts(ir) == 2


# --- the removed `topn_over_sorted_input_to_limit` -------------------------------------------
#
# It rewrote `Sort(x, keys, limit=n)` -> `Limit(x, n)`. Sound, and dead: `Sort.limit` is set
# only by rules in FUSION (phase 5) and later, so a REWRITE (phase 2) rule never sees one.
# `ds.sort(...).limit(n)` reaches REWRITE as `Limit(Sort(...))`, which the rule above already
# collapses. Its own tests passed because they built `Sort(limit=n)` by hand and called the
# function directly -- a shape the optimizer does not produce there.


def test_the_topn_rule_is_not_registered():
    assert "topn_over_sorted_input_to_limit" not in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_a_top_n_over_a_sorted_input_still_collapses_to_a_limit():
    """The behaviour the removed rule aimed at, achieved by the plain-sort rule above.

    Built through the public API rather than by hand, which is the whole point: the shape the
    optimizer actually sees is a `Limit` above a plain `Sort`, not a `Sort` carrying a limit.
    """
    plan = _t().sort("x", descending=True).sort("x", descending=True).limit(2)._plan
    ir = Optimizer().optimize(plan).ir
    assert _num_sorts(ir) == 1


# --- a window delivers no ordering ------------------------------------------------------------
#
# The estimator used to pass a window's input ordering through, so `with_row_index` below a
# window let this rule delete a `sort("_row")` above it. Only the in-memory kernel keeps input
# positions; a spilled window emits grace buckets (partitioned) or `ORDER BY` range buckets
# (global), and Carbonite decides to spill after Kyber has planned. The `with_row_index` case
# without a window is the positive control: the same sort over an order-preserving operator is
# still removed, so a surviving sort above a window is the window's doing.


def _indexed(n: int = 6):
    data = {"x": [(i * 7) % 11 for i in range(n)], "g": [i % 4 for i in range(n)]}
    return bt.from_pydict(data).with_row_index("_row")


def _windowed_by_partition(n: int = 6):
    return _indexed(n).with_columns(s=bt.col("x").sum().over(partition_by=["g"], order_by=["_row"]))


def _windowed_globally():
    return _indexed().with_columns(p=bt.col("x").shift(1).over(order_by=["_row"]))


def test_positive_control_a_sort_over_the_row_index_is_eliminated():
    ir = Optimizer().optimize(_indexed().filter(bt.col("g") > 0).sort("_row")._plan).ir
    assert _num_sorts(ir) == 0


def test_a_window_delivers_no_ordering_to_the_estimator():
    ctx = _ctx()
    assert ctx.estimator.estimate(_indexed()._plan).sorted_by, "positive control"
    assert ctx.estimator.estimate(_windowed_by_partition()._plan).sorted_by == ()
    assert ctx.estimator.estimate(_windowed_globally()._plan).sorted_by == ()


def test_a_sort_above_a_partitioned_window_survives():
    ir = Optimizer().optimize(_windowed_by_partition().sort("_row")._plan).ir
    assert _num_sorts(ir) == 1
    assert ir["op"] == "sort"


def test_a_sort_above_a_global_window_survives():
    ir = Optimizer().optimize(_windowed_globally().sort("_row")._plan).ir
    assert _num_sorts(ir) == 1


def test_a_sort_above_a_first_distinct_marker_survives():
    """The reported shape: `is_first_distinct` lowers to a window partitioned by the value."""
    ds = _indexed().with_columns(f=bt.col("x").is_first_distinct(bt.col("_row")))
    assert _num_sorts(Optimizer().optimize(ds.sort("_row")._plan).ir) == 1


def test_the_optimized_plan_keeps_the_rows_and_their_order_when_spilled():
    """Semantics: the optimized plan, run through the spilling window, still orders by `_row`.

    40 rows over four partitions is enough for the grace-partitioned window to emit its buckets
    out of row order, which is what made the eliminated sort visible.
    """
    from batcher.api.dataset.frame import Dataset

    ds = _windowed_by_partition(40).sort("_row")
    _physical, logical, _decisions = Optimizer().optimize_full(ds._plan)
    optimized = Dataset(logical, ds._sources)
    expected = ds.collect().to_pydict()
    assert expected["_row"] == list(range(40))
    assert optimized.collect(spill=True).to_pydict() == expected
