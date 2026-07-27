"""Per-operator memory estimation — what envelope a plan needs to run in memory.

Kyber annotates each physical operator with a `ResourceBounds` carrying its
estimated peak memory (`m_max_bytes`). Carbonite consumes those: the engine
materializes one pipeline breaker at a time (a linear plan), so the plan's
in-memory footprint is dominated by its single largest breaker rather than the sum
of all operators. `OperatorMemoryEstimator` returns that peak as the envelope the
admission check and the spill decision reason about.

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

__all__ = ["OperatorMemoryEstimator", "binding_operator", "peak_operator_bytes"]


def peak_operator_bytes(plan: PhysicalPlan) -> int:
    """The largest per-operator memory estimate in `plan` (0 if none are sized).

    The dominant breaker bounds the linear pipeline's in-memory footprint; summing
    operators would double-count memory that is never live at the same time.
    """
    return max((op.bounds.m_max_bytes for op in plan.ops), default=0)


def binding_operator(plan: PhysicalPlan):
    """The operator whose memory estimate *is* the plan's envelope, or `None`.

    Every Carbonite memory decision reduces the plan to one number — the dominant
    breaker — and then reports that number with nothing attached. An operator reading
    "this query will spill" has no way back from the figure to the operator that produced
    it, which is the only actionable part: it names which join, aggregate, or sort to
    reshape. Three call sites re-derived this `max` locally (admission's provenance check,
    the estimator, the scheduling grant), so it also removes a triplicated rule.

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
        model = ctx.memory_model
        peak = model.plan_peak(plan.ops) if model is not None else peak_operator_bytes(plan)
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
