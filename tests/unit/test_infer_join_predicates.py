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


def _contains(expr, target: dict) -> bool:
    """Whether `target` appears anywhere inside the expression `expr`."""
    if expr == target:
        return True
    if isinstance(expr, dict):
        return any(_contains(v, target) for v in expr.values())
    if isinstance(expr, list):
        return any(_contains(v, target) for v in expr)
    return False


def _count_ir(ir: dict, predicate_ir: dict) -> int:
    """How many Filters in the IR tree *constrain* by `predicate_ir`.

    The predicate is counted when it appears anywhere inside a Filter's predicate, not only
    when it is the whole of it. A pushed predicate is routinely `AND`-ed with another one
    the optimizer inferred for the same side (a join key's implied `IS NOT NULL`, say), and
    an equality test on the whole predicate then reports zero for a side that is in fact
    constrained — measuring the shape of the conjunction rather than the thing under test.
    """
    count = 0
    if isinstance(ir, dict):
        if ir.get("op") == "filter" and _contains(ir.get("predicate"), predicate_ir):
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
