"""Plan-shape, idempotence, and negative tests for the `empty_relation` rules."""

from __future__ import annotations

import dataclasses

import batcher as bt
from batcher import col
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra.empty_relation import (
    aggregate_over_empty,
    filter_false_to_empty,
    project_over_empty,
    window_over_empty,
)
from batcher.plan.expr_ir import Col, Lit
from batcher.plan.logical import Aggregate, Filter, Limit, Project, Projection, Window

_RULES = {
    "aggregate_over_empty",
    "filter_false_to_empty",
    "project_over_empty",
    "window_over_empty",
}


def _scan():
    return bt.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6]})._plan


def _empty_input(node):
    """`node` rebuilt with its input wrapped in the canonical empty marker."""
    return dataclasses.replace(node, input=Limit(node.input, 0))


def test_rules_registered():
    assert {r.name for r in DEFAULT_REGISTRY.rules()} >= _RULES


def test_filter_false_to_empty():
    node = Filter(_scan(), Lit(False))
    out = filter_false_to_empty(node, None)
    assert isinstance(out, Limit) and out.n == 0
    # Idempotent by construction: the output is a `Limit`, which the driver never
    # re-dispatches to this Filter-matching rule.


def test_filter_false_negative():
    assert filter_false_to_empty(Filter(_scan(), Lit(True)), None) is None
    assert filter_false_to_empty(Filter(_scan(), Col("a") > Lit(1)), None) is None


def test_project_over_empty():
    node = Project(Limit(_scan(), 0), (Projection("c", Col("a")),))
    out = project_over_empty(node, None)
    assert isinstance(out, Limit) and out.n == 0 and isinstance(out.input, Project)
    # Idempotent: the rebuilt inner Project no longer sits over a Limit(_, 0).
    assert project_over_empty(out.input, None) is None
    # Negative: a non-empty input is untouched.
    assert project_over_empty(Project(_scan(), (Projection("c", Col("a")),)), None) is None


def test_aggregate_over_empty_grouped():
    agg = bt.from_pydict({"a": [1], "b": [2]}).group_by("b").agg(s=col("a").sum())._plan
    assert isinstance(agg, Aggregate) and agg.group_keys
    out = aggregate_over_empty(_empty_input(agg), None)
    assert isinstance(out, Limit) and out.n == 0 and isinstance(out.input, Aggregate)
    assert aggregate_over_empty(out.input, None) is None  # idempotent
    assert aggregate_over_empty(agg, None) is None  # negative: non-empty input


def test_aggregate_over_empty_global_not_folded():
    # A keyless (global) aggregate emits one row over empty input → must NOT fold.
    agg = bt.from_pydict({"a": [1]}).group_by().agg(n=col("a").sum())._plan
    assert isinstance(agg, Aggregate) and not agg.group_keys
    assert aggregate_over_empty(_empty_input(agg), None) is None


def _window_node(plan):
    while plan is not None and not isinstance(plan, Window):
        plan = getattr(plan, "input", None)
    return plan


def test_window_over_empty():
    plan = (
        bt.from_pydict({"a": [1, 2], "b": [3, 3]})
        .window(partition_by=["b"], functions={"w": ("sum", "a")})
        ._plan
    )
    win = _window_node(plan)
    assert isinstance(win, Window)
    out = window_over_empty(_empty_input(win), None)
    assert isinstance(out, Limit) and out.n == 0 and isinstance(out.input, Window)
    assert window_over_empty(out.input, None) is None  # idempotent
    assert window_over_empty(win, None) is None  # negative: non-empty input
