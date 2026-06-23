"""Plan-shape unit tests for `constant_propagation`."""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.algebraic import constant_propagation
from batcher.plan.logical import Filter


def _t():
    return bt.from_pydict({"x": [1, 2, 3], "y": [10, 20, 30]})


def test_rule_registered():
    assert "constant_propagation" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_propagates_into_sibling():
    plan = _t().filter((col("x") == 5) & (col("y") > col("x")))._plan
    out = constant_propagation(plan, None)
    assert isinstance(out, Filter)
    ir = out.predicate.to_ir()
    # y > x  →  y > 5  (right comparison now col-vs-literal)
    assert ir["right"]["op"] == "gt"
    assert ir["right"]["right"]["value"]["int"] == 5


def test_keeps_defining_equality():
    plan = _t().filter((col("x") == 5) & (col("y") > col("x")))._plan
    ir = constant_propagation(plan, None).predicate.to_ir()
    # the `x = 5` conjunct is preserved (so the filter still applies it)
    assert ir["left"]["op"] == "eq"
    assert ir["left"]["right"]["value"]["int"] == 5


def test_no_equality_is_noop():
    plan = _t().filter(col("y") > col("x"))._plan
    assert constant_propagation(plan, None) is None


def test_no_op_when_nothing_to_substitute():
    # `x = 5 AND y > 1`: y > 1 has no `x` to replace → no change.
    plan = _t().filter((col("x") == 5) & (col("y") > 1))._plan
    assert constant_propagation(plan, None) is None


def test_idempotent():
    plan = _t().filter((col("x") == 5) & (col("y") > col("x")))._plan
    once = constant_propagation(plan, None)
    assert constant_propagation(once, None) is None
