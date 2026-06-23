"""Projection merging: collapse stacked Projects without duplicating work (W3)."""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.rules.projections import (
    eliminate_identity_project,
    merge_projections,
    push_filter_through_project,
)
from batcher.plan.logical import Project


def _t():
    return bt.from_pydict({"x": [1, 2, 3], "y": [10, 20, 30]})


def _count_projects(ir: dict) -> int:
    n = 1 if ir.get("op") == "project" else 0
    for key in ("input", "left", "right"):
        if isinstance(ir.get(key), dict):
            n += _count_projects(ir[key])
    for child in ir.get("inputs", []) or []:
        n += _count_projects(child)
    return n


def test_two_projects_merge_into_one():
    # with_columns then select -> two stacked Projects -> one.
    plan = _t().with_columns(z=col("x") + col("y")).select("z")._plan
    ir = Optimizer().optimize(plan).ir
    assert _count_projects(ir) == 1
    assert ir["op"] == "project"
    assert ir["input"]["op"] == "scan"


def test_merge_preserves_renames():
    plan = _t().select(a=col("x"), b=col("y")).select(a2=col("a"))._plan
    ir = Optimizer().optimize(plan).ir
    assert _count_projects(ir) == 1


def test_no_merge_when_inner_column_reused():
    # Outer references the inner computed column twice; inlining would compute
    # `x + y` twice, so the merge must be declined.
    plan = _t().with_columns(s=col("x") + col("y")).select(d=col("s") + col("s"))._plan
    assert merge_projections(plan, None) is None  # guard declines duplication
    assert _count_projects(Optimizer().optimize(plan).ir) == 2  # both Projects remain


def test_merge_projections_unit_returns_none_off_project():
    # Input is a Scan, not a Project -> rule does not fire.
    plan = _t().select(a=col("x"))._plan
    assert merge_projections(plan, None) is None


# --- eliminate_identity_project -----------------------------------------------


def test_identity_projection_is_removed():
    # select the same columns in the same order -> a no-op projection.
    plan = _t().select("x", "y")._plan
    assert isinstance(plan, Project)  # the select built a Project
    out = eliminate_identity_project(plan, None)
    assert out is plan.input  # collapsed to its input
    assert Optimizer().optimize(plan).ir["op"] == "scan"


def test_reordering_projection_is_not_identity():
    # same columns, different order -> not identity, keep it.
    plan = _t().select("y", "x")._plan
    assert eliminate_identity_project(plan, None) is None


def test_renaming_projection_is_not_identity():
    # x renamed to a -> not identity, keep it.
    plan = _t().select(a=col("x"), y=col("y"))._plan
    assert eliminate_identity_project(plan, None) is None


# --- push_filter_through_project ----------------------------------------------


def test_filter_pushed_through_passthrough_projection():
    # rename x->a, then filter on a: filter can move below the projection.
    plan = _t().select(a=col("x"), b=col("y")).filter(col("a") > 1)._plan
    out = push_filter_through_project(plan, None)
    assert isinstance(out, Project)  # projection now on top, filter moved under it
    ir = Optimizer().optimize(plan).ir
    assert ir["op"] == "project"
    assert ir["input"]["op"] == "filter"
    assert ir["input"]["input"]["op"] == "scan"


def test_filter_not_pushed_through_computed_column():
    # filter references a COMPUTED column (x+y); pushing would reorder the
    # computation across the filter, so it must stay above the projection.
    plan = _t().with_columns(s=col("x") + col("y")).filter(col("s") > 5)._plan
    assert push_filter_through_project(plan, None) is None
    ir = Optimizer().optimize(plan).ir
    assert ir["op"] == "filter"  # filter remains on top
