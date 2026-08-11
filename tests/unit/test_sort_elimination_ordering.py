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
