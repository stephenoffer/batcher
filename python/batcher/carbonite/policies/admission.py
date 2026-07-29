"""Admission: does this plan fit the memory envelope, and if not, what is the counter-offer?

Carbonite's first gate. Kyber has already sized each operator; this module decides whether
the plan's dominant materializing operator fits a soft fraction of the memory the process
can actually use, and when it does not, hands back a spill-friendly bound instead of
letting the query OOM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.carbonite.memory.estimator import (
    binding_operator,
    learned_plan_peak,
    peak_contributors,
)
from batcher.carbonite.memory.pressure import total_memory_bytes
from batcher.plan.resource import FeasibilityVerdict, ResourceBounds
from batcher.plan.stats import Provenance

if TYPE_CHECKING:
    from batcher.carbonite.base import ResourceContext
    from batcher.plan.physical import PhysicalPlan

__all__ = ["BudgetingAdmission"]


class BudgetingAdmission:
    """Real admission: reject a plan whose dominant materializing operator would
    not fit the memory envelope, returning a spill-friendly counter-offer.

    Conservative by construction so it never fails a legitimate query: it budgets
    only operators with a *known* size (Kyber leaves unknown-size operators at
    `m_max_bytes == 0`), compares against a soft fraction of physical RAM, and uses
    the single dominant breaker (operators materialize one at a time in a linear
    pipeline) rather than over-summing. With no bounds emitted, it abstains.
    """

    def __init__(
        self, available_bytes: int | None = None, *, soft_limit: float | None = None
    ) -> None:
        # Optional explicit overrides (used by tests / a standalone policy). When
        # left unset, `validate` reads the unified envelope + soft limit from the
        # `ResourceContext` the manager threads in, so admission budgets against the
        # *same* figure as spill and reserve (and the live config, not a stale one).
        self._available = available_bytes
        self._soft = soft_limit

    def validate(self, plan: PhysicalPlan, ctx: ResourceContext) -> FeasibilityVerdict:
        if not plan.ops:
            return FeasibilityVerdict(feasible=True)  # no annotations → abstain
        available = self._available
        if available is None:
            available = (
                ctx.envelope_bytes if ctx.envelope_bytes is not None else total_memory_bytes()
            )
        soft = self._soft if self._soft is not None else ctx.config.memory.soft_limit
        envelope = int(available * soft)
        # Cross-query admission: subtract what concurrent queries already hold
        # against the shared buffer pool, so N queries that each individually fit the
        # envelope are not all admitted into a collective OOM.
        if self._available is None:
            from batcher.carbonite.memory.pool import current_process_pool

            pool = current_process_pool()
            if pool is not None:
                envelope = max(0, envelope - pool.used)
        # The envelope can never be smaller than one morsel: the engine must hold at
        # least a single morsel to make any progress, and a *streaming* operator's whole
        # footprint is one morsel (`min(morsel_rows·width, morsel_bytes)`). Flooring here
        # keeps a streaming/tiny plan feasible under a sub-morsel budget (it would
        # otherwise be rejected as infeasible with "no out-of-core path", since a
        # streaming op has nothing to spill) — a no-op for any realistic budget, which is
        # orders of magnitude larger than a morsel. A genuine breaker that materializes
        # more than this floor still exceeds it and routes to the spill path.
        envelope = max(envelope, ctx.config.execution.morsel_bytes)
        # Blend each operator's plan estimate toward its measured peak (learned from
        # `m_peak_bytes`) before taking the dominant breaker, so admission budgets
        # against what the family really used — admitting a query the plan over-sized
        # (avoiding a needless spill route) and catching one the plan under-sized
        # (avoiding an OOM). Cold families pass through unchanged.
        #
        # The shared rule, not a local `max`: admission and the distributed grant must
        # size against the identical figure, or a query is admitted against one envelope
        # and granted another.
        peak = learned_plan_peak(plan, ctx.memory_model)
        if peak <= envelope:
            return FeasibilityVerdict(feasible=True)
        # Over budget: offer the envelope as the per-operator bound so the engine can
        # re-plan with a spill-friendly strategy instead of OOMing.
        #
        # `plan/physical.py` promises that "Carbonite reads provenance to decide how
        # defensively to budget", and this is where it must: the byte figure above is only
        # as trustworthy as the cardinality it was derived from. When the operator that
        # binds the constraint was sized from a pure Selinger guess, the verdict is
        # *advisory* — it still routes the plan out-of-core, but the conductor will not
        # fail a query on it. Rejecting on a guess breaks the admission contract that a
        # guess never fails a legitimate query.
        binding = binding_operator(plan)
        return FeasibilityVerdict(
            feasible=False,
            binding_constraint="memory",
            suggested_bounds=ResourceBounds(
                m_max_bytes=envelope, c_max_credits=0, n_max_parallelism=0
            ),
            # Naming the operator is what turns "this query will spill" into something a
            # reader can act on: it says which join/aggregate/sort to reshape.
            binding_op=None if binding is None else f"{binding.kind}#{int(binding.op_id)}",
            advisory=_rests_on_a_guess(plan, binding),
        )


def _rests_on_a_guess(plan: PhysicalPlan, binding) -> bool:
    """Whether the envelope this verdict rejects on was derived from a guess.

    Reads **every operator that contributes to the peak**, not just the largest one. On a
    linear pipeline they are the same operator and this is exactly the previous rule; on a
    bushy plan the peak is a *sum* over operators alive at the same moment, and a sum of an
    EXACT term and a guessed one is a guess. Asking only the larger term would fail a
    legitimate query on the strength of the smaller — which is precisely what the admission
    contract forbids ("a guess never fails a legitimate query").

    Args:
        plan: The annotated physical plan.
        binding: The operator named in the verdict, or `None` when nothing was sized.

    Returns:
        True when the verdict should route the plan out-of-core without failing it.
    """
    if binding is None:
        return True
    contributors = peak_contributors(plan) or (binding,)
    return any(op.properties.provenance is Provenance.DEFAULT for op in contributors)
