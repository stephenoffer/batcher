"""Per-operator memory estimation — what envelope a plan needs to run in memory.

Kyber annotates each physical operator with a `ResourceBounds` carrying its
estimated peak memory (`m_max_bytes`). Carbonite consumes those: on a linear plan the
engine materializes one pipeline breaker at a time, so the in-memory footprint is
dominated by the single largest breaker rather than the sum of all operators.
`OperatorMemoryEstimator` returns that peak as the envelope the admission check and the
spill decision reason about. `peak_operator_bytes` records where the linear assumption
stops holding.

Everything here is a *rule*, used by more than one caller, and the callers must agree:
a plan admitted against one envelope and granted another is a query admitted into a
budget it was never given. So admission, the estimator, and the distributed grant all
call `learned_plan_peak` and `binding_operator` rather than re-deriving them.

This replaces the permissive bootstrap estimator. It stays conservative: operators
Kyber could not size (`m_max_bytes == 0`) contribute nothing, so a query is never
pushed to spill on a guess — only on an estimate the optimizer actually produced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.plan.resource import ResourceBounds

if TYPE_CHECKING:
    from batcher.carbonite.base import ResourceContext
    from batcher.plan.physical import PhysicalPlan

__all__ = [
    "OperatorMemoryEstimator",
    "binding_operator",
    "learned_plan_peak",
    "peak_operator_bytes",
]


def peak_operator_bytes(plan: PhysicalPlan) -> int:
    """The largest per-operator memory estimate in `plan` (0 if none are sized).

    The dominant breaker bounds a *linear* pipeline's in-memory footprint, and summing
    operators would double-count memory that is never live at the same time.

    **Where this under-counts, stated because the figure is used as a safety bound.** A
    bushy plan holds more than one breaker at once: a hash join's build side is resident
    while the other side runs, so a join of two joins has three hash tables live together
    and this returns the largest of the three. On a four-way bushy join Kyber sized those
    three at 18.2 / 9.1 / 9.1 MB, so this reports 18.2 where a top-join-plus-one-subtree
    reading is 27.4 and all-three-resident is 36.5 — 1.5x to 2x, by arithmetic over the
    optimizer's own estimates rather than by a runtime measurement.

    Fixing it needs the plan's tree, and `PhysicalOp.inputs` — the field that would carry
    it — is hardcoded empty in `kyber/annotate.py` and read by nothing. A concurrency-aware
    rule written against it today silently degenerates to exactly this `max`, which is worse
    than the honest under-count because it *looks* correct. Populating `inputs` is Kyber's
    to do; until then the under-count is real, bounded by the plan's breaker count, and
    covered downstream by the pressure ladder rather than by this estimate.
    """
    return max((op.bounds.m_max_bytes for op in plan.ops), default=0)


def learned_plan_peak(plan: PhysicalPlan, model) -> int:
    """The plan's memory envelope, blended toward measured reality when a model exists.

    Every Carbonite memory decision starts here: admission's fit check, the estimator's
    envelope, and the distributed per-task grant all need the same number, and they must
    agree — a plan admitted against one figure and granted against another is a query
    admitted into a budget it was never given.

    `plan_peak` already folds each operator's plan estimate toward what the family really
    used (`m_peak_bytes`), and passes cold families through unchanged, so on a cold store
    this is exactly `peak_operator_bytes`.

    Args:
        plan: The annotated physical plan.
        model: A `LearnedMemoryModel`, or `None` on a cold store.

    Returns:
        The envelope in bytes; `0` when nothing in the plan could be sized.
    """
    return model.plan_peak(plan.ops) if model is not None else peak_operator_bytes(plan)


def binding_operator(plan: PhysicalPlan):
    """The operator whose memory estimate *is* the plan's envelope, or `None`.

    Every Carbonite memory decision reduces the plan to one number — the dominant
    breaker — and then reports that number with nothing attached. An operator reading
    "this query will spill" has no way back from the figure to the operator that produced
    it, which is the only actionable part: it names which join, aggregate, or sort to
    reshape.

    This is the one implementation of the rule. Three call sites re-derived the `max`
    locally — admission's provenance check, the estimator, the scheduling grant — and
    admission went on re-deriving it after the other two were folded in here, while this
    docstring already claimed the triplication was gone. That is how a "deduplicated"
    helper keeps a surviving copy free to drift from it: the check that matters is whether
    anything still spells the `max` out, not whether a helper exists to spell it once.

    Args:
        plan: The annotated physical plan.

    Returns:
        The `PhysicalOp` holding the peak estimate, or `None` when nothing is sized.
    """
    sized = [op for op in plan.ops if op.bounds.m_max_bytes > 0]
    if not sized:
        return None
    return max(sized, key=lambda op: op.bounds.m_max_bytes)


class OperatorMemoryEstimator:
    """Estimates a plan's memory envelope from Kyber's per-operator bounds.

    The envelope's `m_max_bytes` is the dominant breaker (`peak_operator_bytes`);
    the credit and parallelism fields carry the same conservative defaults the
    bootstrap used so the flow-control and scheduling sides are unaffected until
    they grow their own estimates.

    When a `LearnedMemoryModel` is present on the context (the hub has measured
    `m_peak_bytes` for this operator family), each operator's plan estimate is
    *blended* toward that measured reality before the dominant breaker is taken —
    so admission, spill, and reserve all size against what the query really used,
    not the plan guess alone. Cold families pass through unchanged, so on a cold
    store the envelope equals the plan's own dominant breaker exactly.
    """

    def envelope(self, plan: PhysicalPlan, ctx: ResourceContext) -> ResourceBounds:
        fc = ctx.config.flow_control
        peak = learned_plan_peak(plan, ctx.memory_model)
        # Credits and parallelism take the plan's own widest request when Kyber emitted
        # one, falling back to the configured defaults for an unsized plan. They used to be
        # the configured constants unconditionally, which made the returned envelope a
        # description of the *config* rather than of the plan for two of its three fields —
        # so any consumer reading them (rather than only `m_max_bytes`) would have been
        # told a 200-way shuffle wanted the default 4-way parallelism.
        #
        # `getattr` rather than attribute access for the same reason `plan_peak` uses it:
        # a bare-sized bounds object (a test double carrying only `m_max_bytes`) is a
        # supported shape, and an estimator is never the right place to fail a query.
        return ResourceBounds(
            m_max_bytes=peak,
            c_max_credits=_widest(plan, "c_max_credits") or fc.default_credits,
            n_max_parallelism=_widest(plan, "n_max_parallelism")
            or (ctx.config.execution.parallelism or 0),
        )


def _widest(plan: PhysicalPlan, field: str) -> int:
    """The largest `bounds.<field>` across `plan`'s operators; `0` when none declare it."""
    return max((int(getattr(op.bounds, field, 0) or 0) for op in plan.ops), default=0)
