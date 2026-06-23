"""Carbonite real admission — memory budgeting against the envelope.

Kyber annotates each operator with an estimated memory bound; `BudgetingAdmission`
rejects a plan whose dominant breaker won't fit, with a spill-friendly counter-offer.
It is conservative: unknown-size operators are unbudgeted, so a real query is never
failed on a guess.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count
from batcher.carbonite.base import ResourceContext
from batcher.carbonite.policies import BudgetingAdmission
from batcher.config import active_config
from batcher.plan.ids import OpId
from batcher.plan.physical import PhysicalOp, PhysicalPlan, PlanProperties
from batcher.plan.resource import ResourceBounds


def _ctx() -> ResourceContext:
    return ResourceContext(config=active_config())


def _plan(*memories: int) -> PhysicalPlan:
    ops = tuple(
        PhysicalOp(
            op_id=OpId(i),
            kind="Aggregate",
            backend="native",
            algorithm="",
            bounds=ResourceBounds(m_max_bytes=m, c_max_credits=0, n_max_parallelism=0),
            inputs=(),
            properties=PlanProperties(est_rows=float(m)),
        )
        for i, m in enumerate(memories)
    )
    return PhysicalPlan(ir={}, output_schema=None, ops=ops)


def test_infeasible_when_breaker_exceeds_envelope():
    adm = BudgetingAdmission(available_bytes=1000, soft_limit=0.85)  # envelope = 850
    verdict = adm.validate(_plan(100, 5000, 200), _ctx())  # dominant breaker = 5000
    assert not verdict.feasible
    assert verdict.binding_constraint == "memory"
    assert verdict.suggested_bounds is not None
    assert verdict.suggested_bounds.m_max_bytes == 850


def test_feasible_when_within_envelope():
    adm = BudgetingAdmission(available_bytes=1_000_000, soft_limit=0.85)
    assert adm.validate(_plan(100, 5000, 200), _ctx()).feasible


def test_abstains_with_no_annotations():
    adm = BudgetingAdmission(available_bytes=1)
    assert adm.validate(PhysicalPlan(ir={}, output_schema=None, ops=()), _ctx()).feasible


def test_unknown_size_operators_unbudgeted():
    # Kyber leaves unknown-size operators at m_max_bytes == 0 → never the dominant
    # breaker → feasible even under a tiny envelope.
    adm = BudgetingAdmission(available_bytes=10)
    assert adm.validate(_plan(0, 0, 0), _ctx()).feasible


def test_real_small_query_is_feasible_end_to_end():
    # A normal small query must not be failed by budgeting (regression guard for the
    # raise-on-infeasible path in api/executors).
    t = pa.table({"k": [i % 5 for i in range(1000)], "v": list(range(1000))})
    out = bt.from_arrow(t).filter(col("v") > 500).group_by("k").agg(n=count()).collect()
    assert out.num_rows == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
