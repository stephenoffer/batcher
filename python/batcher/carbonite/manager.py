"""The Carbonite resource manager entry point.

Validates plans for feasibility, hands out credit windows and memory reservations,
and decides when a query must spill. It is a thin orchestrator: it composes one
policy of each kind (admission, spill, flow control, memory estimation — see
`carbonite.base`) plus the memory subsystem (buffer pool + pressure monitor) and
delegates to them. `validate` returns real counter-offers Kyber re-plans around;
`reserve` accounts against the process-wide buffer pool; `should_spill` compares a
plan's estimated envelope to live memory so a large query goes out-of-core instead
of OOMing. An alternate policy plugs in by being passed to the constructor.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager

from batcher.carbonite.base import (
    AdmissionPolicy,
    FlowControlPolicy,
    MemoryEstimator,
    ResourceContext,
    SchedulingPolicy,
)
from batcher.carbonite.memory import OperatorMemoryEstimator, PressureMonitor, process_pool
from batcher.carbonite.policies import (
    AIMDFlowControl,
    BudgetingAdmission,
    DefaultSchedulingPolicy,
    StaticCreditFlowControl,
)
from batcher.config import Config, active_config
from batcher.plan.physical import PhysicalPlan
from batcher.plan.resource import FeasibilityVerdict, ResourceBounds, SchedulingEnvelope

__all__ = ["ResourceManager"]


class ResourceManager:
    """Validates feasibility and allocates resources for execution.

    Composes one policy of each kind and delegates. Pass an alternate policy to
    the constructor to swap the bootstrap default — that is the only seam; there
    is no registry (one policy of each kind exists today).
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        admission: AdmissionPolicy | None = None,
        flow_control: FlowControlPolicy | None = None,
        memory: MemoryEstimator | None = None,
        scheduling: SchedulingPolicy | None = None,
    ) -> None:
        self._config = config or active_config()
        self._pressure = PressureMonitor(self._config)
        # Sample the query's memory envelope ONCE so admission, spill, and reserve
        # all reason about the same figure (no live-RAM drift between decisions).
        self._envelope = self._pressure.envelope_bytes()
        self._ctx = ResourceContext(config=self._config, envelope_bytes=self._envelope)
        # Single-entry envelope cache keyed by plan *identity* (a held reference, so
        # `is` is stable and the object can't be GC'd into an id collision).
        self._peak_plan: object = None
        self._peak_value = 0
        self._admission = admission or BudgetingAdmission()
        self._flow_control = flow_control or StaticCreditFlowControl()
        self._memory = memory or OperatorMemoryEstimator()
        self._scheduling = scheduling or DefaultSchedulingPolicy()

    def validate(self, plan: PhysicalPlan) -> FeasibilityVerdict:
        """Check whether `plan` can run within available resources.

        The default `BudgetingAdmission` compares each operator's estimated memory
        (Kyber's per-operator `ResourceBounds`) against a soft fraction of physical
        RAM and returns a spill-friendly counter-offer when the dominant breaker
        would not fit. Conservative: unknown-size operators are not budgeted, so a
        legitimate query is never failed on a guess.
        """
        return self._admission.validate(plan, self._ctx)

    def grant_credits(self, requested: int) -> int:
        """Grant a credit window (in-flight `RecordBatch` slots) for a data channel.

        One credit = one buffered batch, so the returned window bounds a shuffle
        channel's memory. The default `StaticCreditFlowControl` clamps `requested`
        (typically an operator's `ResourceBounds.c_max_credits`) into a memory-safe
        band derived from `FlowControlConfig`; this is the single authority that
        replaces the engine's hardcoded `DEFAULT_CREDITS`. Always returns >= 1 so a
        channel never stalls at zero credits.
        """
        return self._flow_control.grant(requested, self._ctx)

    def scheduling_envelope(
        self, plan: PhysicalPlan, requested_workers: int | None = None
    ) -> SchedulingEnvelope:
        """Per-Ray-task scheduling grant for `plan` (num_cpus/memory/n_tasks/credits).

        Carbonite protects: it turns Kyber's *desired* parallelism/credits into a
        grant clamped to the live machine. `n_tasks` tracks estimated data size
        (replacing a blind `os.cpu_count()`), `memory_bytes` is the dominant breaker
        split across tasks within a fair share of the budget, and `credits` is the
        flow-control authority's clamp of the plan's widest credit request — so the
        distributed shuffle starts with a metadata-derived window, not a hardcoded 0.
        """
        env = self._scheduling.envelope(
            plan,
            self._ctx,
            requested_workers=requested_workers,
            available_bytes=self._hard_budget(),
        )
        max_credits = max((op.bounds.c_max_credits for op in plan.ops), default=0)
        return dataclasses.replace(env, credits=self.grant_credits(max_credits))

    def adaptive_flow_control(self) -> AIMDFlowControl:
        """Vend an AIMD credit controller for an adaptive shuffle channel.

        The driver-side `grant_credits` sets the *initial* window from the operator's
        estimate; a long-lived channel can instead hold one of these and grow/shrink
        the window per round from observed backpressure (the `ShuffleSession`'s
        opt-in adaptive mode). Stateful — one controller per channel."""
        return AIMDFlowControl(self._config)

    def _peak_bytes(self, plan: PhysicalPlan) -> int:
        """The plan's estimated peak in-memory bytes, computed once per plan.

        `estimated_bytes`, `should_spill`, and `reserve` all consult this, so the
        per-plan envelope is built once rather than three times (C37).
        """
        if plan is not self._peak_plan:
            self._peak_value = self._memory.envelope(plan, self._ctx).m_max_bytes
            self._peak_plan = plan
        return self._peak_value

    def estimated_bytes(self, plan: PhysicalPlan) -> int:
        """Estimated peak in-memory bytes for `plan` (its dominant breaker).

        The figure `reserve` accounts and `should_spill` compares against the
        budget. 0 when Kyber emitted no sizes (an un-estimable plan).
        """
        return self._peak_bytes(plan)

    def should_spill(self, plan: PhysicalPlan) -> bool:
        """Decide whether `plan` should run out-of-core rather than in memory.

        Compares the plan's estimated peak memory (the dominant breaker, via the
        `MemoryEstimator`) against the unified hard budget. When the estimate won't
        fit, the conductor routes the query through the spilling executor so it
        completes under bounded memory instead of OOMing. Conservative: an unsized
        plan (no Kyber estimate) never spills on a guess.
        """
        estimated = self._peak_bytes(plan)
        if estimated <= 0:
            return False
        return estimated > self._hard_budget()

    def _soft_budget(self) -> int:
        """Bytes a query aims to stay under (the admission/throttle threshold)."""
        return int(self._envelope * self._config.memory.soft_limit)

    def _hard_budget(self) -> int:
        """Bytes a query may hold in memory before it must spill (the spill/reserve
        cap). Both `should_spill` and `reserve` use this one figure, derived from the
        once-sampled envelope, so the two decisions never disagree."""
        return int(self._envelope * self._config.memory.hard_limit)

    @contextmanager
    def reserve(self, m_bytes: int) -> Iterator[bool]:
        """Reserve `m_bytes` against the process-wide buffer pool for the block.

        Accounts the reservation on entry and releases it on exit (even if the
        block raises), so concurrent queries and the transfer layer see a single
        shared envelope. Yields whether the reservation fit; a `False` means the
        pool is already over budget and the caller should be on the spill path.
        The pool is sized to Carbonite's hard memory envelope — the same figure
        `should_spill` compares against, so the two decisions stay consistent.
        """
        pool = process_pool(self._hard_budget())
        with pool.reserve(m_bytes) as granted:
            yield granted

    def default_bounds(self) -> ResourceBounds:
        """Permissive bounds used until Kyber emits per-operator bounds."""
        fc = self._config.flow_control
        return ResourceBounds(
            m_max_bytes=1 << 62,
            c_max_credits=fc.default_credits,
            n_max_parallelism=self._config.execution.parallelism or 0,
        )
