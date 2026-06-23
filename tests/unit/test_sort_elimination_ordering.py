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
