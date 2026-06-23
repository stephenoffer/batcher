"""Plan-shape unit tests for transitive join-predicate inference."""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.pushdown import infer_join_predicates
from batcher.plan.logical import Filter, Join


def _fact():
    return bt.from_pydict({"dept_id": [10, 20, 10, 30], "amt": [1, 2, 3, 4]})


def _dim():
    return bt.from_pydict({"dept_id": [10, 20, 30], "region": ["EU", "US", "EU"]})


def _count_ir(ir: dict, predicate_ir: dict) -> int:
    """How many times `predicate_ir` appears as a Filter predicate in the IR tree."""
    count = 0
    if isinstance(ir, dict):
        if ir.get("op") == "filter" and ir.get("predicate") == predicate_ir:
            count += 1
        for v in ir.values():
            count += _count_ir(v, predicate_ir)
    elif isinstance(ir, list):
        for v in ir:
            count += _count_ir(v, predicate_ir)
    return count


def test_rule_registered():
    assert "infer_join_predicates" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_constraint_inferred_from_right_to_left():
    # Filter the dimension (right input) on the join key; expect it mirrored to fact.
    join = _fact().join(_dim().filter(col("dept_id") == 10), on="dept_id")._plan
    assert isinstance(join, Join)
    out = infer_join_predicates(join, None)
    assert isinstance(out, Join)
    assert isinstance(out.left, Filter)  # fact side now carries the inferred filter


def test_inner_join_no_key_constraint_is_noop():
    # A filter on a non-key column does not transfer.
    join = _fact().join(_dim().filter(col("region") == "EU"), on="dept_id")._plan
    assert infer_join_predicates(join, None) is None


def test_outer_join_is_noop():
    join = _fact().join(_dim().filter(col("dept_id") == 10), on="dept_id", how="left")._plan
    assert infer_join_predicates(join, None) is None


def test_idempotent_no_refire():
    join = _fact().join(_dim().filter(col("dept_id") == 10), on="dept_id")._plan
    once = infer_join_predicates(join, None)
    assert infer_join_predicates(once, None) is None


def test_full_optimizer_constrains_both_sides():
    # End to end: the `dept_id = 10` predicate should reach BOTH scans.
    plan = _fact().join(_dim().filter(col("dept_id") == 10), on="dept_id")._plan
    ir = Optimizer().optimize(plan).ir
    predicate_ir = (col("dept_id") == 10).to_ir()
    assert _count_ir(ir, predicate_ir) >= 2
