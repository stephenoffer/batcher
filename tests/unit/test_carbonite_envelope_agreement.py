"""Three Carbonite decisions reduce a plan to one number, and it has to be the same number.

Admission asks "does this fit?", the estimator publishes the envelope, and the distributed
scheduler grants per-task memory. All three answer from the plan's dominant breaker, blended
toward what the operator family really used. If they compute it separately they can drift,
and the failure is quiet in the worst way: a query admitted against one envelope and then
granted another runs in a budget nobody sized it for.

Each site once spelled the `max` out for itself. Two were folded into
`memory/estimator.py`; admission kept its private copies, while the helper's own docstring
claimed the triplication was gone. These tests check the property rather than the refactor,
so a fourth caller re-deriving the rule fails here rather than passing review.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from batcher.carbonite.base import ResourceContext
from batcher.carbonite.memory.estimator import (
    OperatorMemoryEstimator,
    binding_operator,
    learned_plan_peak,
    peak_operator_bytes,
)
from batcher.carbonite.policies.admission import BudgetingAdmission
from batcher.config import Config
from batcher.plan.physical import PhysicalOp, PhysicalPlan
from batcher.plan.resource import ResourceBounds

pytestmark = pytest.mark.unit


#: Sizes here are in MiB, because admission floors its envelope at one morsel (1 MiB by
#: default) so a streaming plan stays feasible under a sub-morsel budget. Byte-scale
#: fixtures all sit under that floor and every one of them reads as feasible, which makes
#: the test pass while asserting nothing about the comparison it means to exercise.
_MIB = 1 << 20


def _plan(*sizes: int) -> PhysicalPlan:
    """A plan whose i-th operator is sized `sizes[i]` MiB."""
    ops = tuple(
        PhysicalOp(
            op_id=i,
            kind="Join" if n else "Project",
            backend="native",
            algorithm="",
            bounds=ResourceBounds(m_max_bytes=n * _MIB, c_max_credits=0, n_max_parallelism=0),
            inputs=(),
        )
        for i, n in enumerate(sizes)
    )
    return PhysicalPlan(ir={}, output_schema=None, ops=ops)


def _ctx(model=None, envelope: int | None = None) -> ResourceContext:
    return ResourceContext(config=Config(), envelope_bytes=envelope, memory_model=model)


@dataclass
class _DoublingModel:
    """A learned model that says every family really used twice its estimate.

    Blunt on purpose. A site that ignores the model reports half of what a site that
    honours it does, which no shared-constant test would catch.

    It implements `blend_peak`, the *per-operator* primitive, and deliberately not a
    whole-plan aggregate. The envelope is a walk over the plan's schedule, so a model that
    could only answer for a whole plan would have to flatten that walk to a `max` — which is
    the pre-`inputs` reading, and taking it on the warm path would have switched the
    concurrent-peak correction off for every store that had learned anything.
    """

    def blend_peak(self, kind: str, planned: int, row_size: float | None = None) -> int:
        return planned * 2


# --- the rule itself ----------------------------------------------------------


def test_the_envelope_is_the_dominant_breaker() -> None:
    assert peak_operator_bytes(_plan(100, 900, 300)) == 900 * _MIB


def test_an_unsized_plan_has_no_envelope() -> None:
    """`0` means "no estimate", which callers must read as "no evidence", not "fits"."""
    assert peak_operator_bytes(_plan(0, 0)) == 0
    assert binding_operator(_plan(0, 0)) is None


def test_a_cold_store_blends_to_exactly_the_plan_estimate() -> None:
    """With no model there is nothing to blend toward, so nothing may change."""
    plan = _plan(100, 900, 300)
    assert learned_plan_peak(plan, None) == peak_operator_bytes(plan)


def test_a_model_moves_the_envelope() -> None:
    """The negative control for the agreement tests below: the model must matter."""
    plan = _plan(100, 900, 300)
    assert learned_plan_peak(plan, _DoublingModel()) == 1800 * _MIB


def test_the_binding_operator_is_the_one_holding_the_peak() -> None:
    """The verdict names an operator, and it must be the operator the verdict is about."""
    plan = _plan(100, 900, 300)
    binding = binding_operator(plan)
    assert binding is not None
    assert binding.op_id == 1
    assert binding.bounds.m_max_bytes == 900 * _MIB


# --- the agreement ------------------------------------------------------------


@pytest.mark.parametrize("model", [None, _DoublingModel()], ids=["cold", "learned"])
def test_admission_and_the_estimator_size_against_the_same_figure(model) -> None:
    """The property the deduplication exists to guarantee.

    Admission is given an envelope one byte under what the estimator publishes: it must
    refuse. Given exactly that figure, it must admit. That brackets the number admission is
    really using, without reaching into it.
    """
    plan = _plan(4, 32, 8)
    ctx = _ctx(model)
    published = OperatorMemoryEstimator().envelope(plan, ctx).m_max_bytes
    assert published > 0

    # `soft_limit=1.0` so the comparison is against the figure itself, not a fraction.
    just_enough = BudgetingAdmission(available_bytes=published, soft_limit=1.0)
    one_short = BudgetingAdmission(available_bytes=published - 1, soft_limit=1.0)

    assert just_enough.validate(plan, ctx).feasible, "admission refused the published envelope"
    assert not one_short.validate(plan, ctx).feasible, (
        "admission admitted a plan one byte over the published envelope — it is sizing "
        "against a different figure"
    )


def test_admission_names_the_same_operator_the_helper_does() -> None:
    """The actionable half of the verdict must point at the real binding operator."""
    plan = _plan(4, 32, 8)
    verdict = BudgetingAdmission(available_bytes=2 * _MIB, soft_limit=1.0).validate(plan, _ctx())
    assert not verdict.feasible
    binding = binding_operator(plan)
    assert binding is not None
    assert verdict.binding_op == f"{binding.kind}#{int(binding.op_id)}"


def test_a_learned_model_reaches_admission_too() -> None:
    """A model that only some sites honour is worse than no model at all.

    Under the doubling model the plan needs twice its estimate, so an envelope that fits
    the raw estimate must now be refused. A site still reading the raw `max` would admit.
    """
    plan = _plan(4, 32, 8)
    raw = peak_operator_bytes(plan)
    admission = BudgetingAdmission(available_bytes=raw, soft_limit=1.0)
    assert admission.validate(plan, _ctx()).feasible
    assert not admission.validate(plan, _ctx(_DoublingModel())).feasible


def test_the_counter_offer_is_the_envelope_the_query_was_measured_against() -> None:
    """A refusal hands back a bound to re-plan under; a wrong one re-plans into the same wall."""
    plan = _plan(4, 32, 8)
    verdict = BudgetingAdmission(available_bytes=16 * _MIB, soft_limit=1.0).validate(plan, _ctx())
    assert not verdict.feasible
    assert verdict.suggested_bounds is not None
    assert verdict.suggested_bounds.m_max_bytes == 16 * _MIB


def test_a_guess_routes_out_of_core_without_failing_the_query() -> None:
    """Provenance is read here, and a Selinger guess must never *fail* a legitimate query.

    The plan is over budget either way; what changes is whether the conductor may act on
    the refusal. `DEFAULT` provenance means nothing was measured, so the verdict is advice.
    """
    from batcher.plan.stats import Provenance

    plan = _plan(4, 32)
    measured = replace(
        plan,
        ops=tuple(
            replace(op, properties=replace(op.properties, provenance=Provenance.HISTOGRAM))
            for op in plan.ops
        ),
    )
    admission = BudgetingAdmission(available_bytes=2 * _MIB, soft_limit=1.0)
    assert admission.validate(plan, _ctx()).advisory, "a pure guess failed a query"
    assert not admission.validate(measured, _ctx()).advisory, (
        "a measured estimate was downgraded to advice, so nothing can act on it"
    )
