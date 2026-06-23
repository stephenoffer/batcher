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

__all__ = ["OperatorMemoryEstimator", "peak_operator_bytes"]


def peak_operator_bytes(plan: PhysicalPlan) -> int:
    """The largest per-operator memory estimate in `plan` (0 if none are sized).

    The dominant breaker bounds the linear pipeline's in-memory footprint; summing
    operators would double-count memory that is never live at the same time.
    """
    return max((op.bounds.m_max_bytes for op in plan.ops), default=0)


class OperatorMemoryEstimator:
    """Estimates a plan's memory envelope from Kyber's per-operator bounds.

    The envelope's `m_max_bytes` is the dominant breaker (`peak_operator_bytes`);
    the credit and parallelism fields carry the same conservative defaults the
    bootstrap used so the flow-control and scheduling sides are unaffected until
    they grow their own estimates.
    """

    def envelope(self, plan: PhysicalPlan, ctx: ResourceContext) -> ResourceBounds:
        fc = ctx.config.flow_control
        return ResourceBounds(
            m_max_bytes=peak_operator_bytes(plan),
            c_max_credits=fc.default_credits,
            n_max_parallelism=ctx.config.execution.parallelism or 0,
        )
