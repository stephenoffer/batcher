"""`PhysicalPlan.op_budgets()` — the per-operator spill budgets Core ships to the
engine (W1). The map keys are the pre-order `op_id`s Kyber assigns in
`_annotate_ops`; only positively-sized (breaker) operators appear, so unsized
streaming/unknown operators fall back to the global budget in the data plane.
"""

from __future__ import annotations

import batcher as bt
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.optimizer import _annotate_ops
from batcher.plan.ids import OpId
from batcher.plan.physical import PhysicalOp, PhysicalPlan
from batcher.plan.resource import ResourceBounds


def _op(op_id: int, kind: str, mem: int) -> PhysicalOp:
    return PhysicalOp(
        op_id=OpId(op_id),
        kind=kind,
        backend="native",
        algorithm="",
        bounds=ResourceBounds(m_max_bytes=mem, c_max_credits=0, n_max_parallelism=0),
        inputs=(),
    )


def test_op_budgets_keeps_only_positive_bounds_keyed_by_op_id():
    plan = PhysicalPlan(
        ir={},
        output_schema=None,
        ops=(
            _op(0, "Aggregate", 4096),
            _op(1, "Filter", 0),  # unsized streaming op → excluded (falls back to global)
            _op(2, "Sort", 8192),
        ),
    )
    assert plan.op_budgets() == {0: 4096, 2: 8192}


def test_op_budgets_empty_when_no_ops():
    # The bootstrap plan with no PhysicalOp DAG ships an empty map (every operator
    # falls back to the global budget), reproducing the pre-W1 behaviour.
    assert PhysicalPlan(ir={}, output_schema=None).op_budgets() == {}


def test_op_budget_ids_are_preorder_and_align_with_annotation():
    """The op ids `_annotate_ops` assigns are a contiguous pre-order range, and
    `op_budgets()` keys are a subset of it — so they line up with the pre-order
    numbering the Rust `IdGen` hands the engine."""
    from batcher.config import active_config

    cfg = active_config()
    ds = bt.from_pydict({"k": [1, 2, 1], "v": [3, 4, 5]}).group_by("k").agg(s=bt.col("v").sum())
    est = CardinalityEstimator([])
    ops = _annotate_ops(ds._plan, est, cfg)
    ids = [int(op.op_id) for op in ops]
    assert ids == list(range(len(ops)))  # contiguous pre-order

    plan = PhysicalPlan(ir={}, output_schema=None, ops=tuple(ops))
    assert set(plan.op_budgets()).issubset(set(ids))
