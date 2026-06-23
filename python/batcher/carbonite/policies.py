"""Bootstrap resource policies — the permissive single-node defaults.

Each class implements one `carbonite.base` policy `Protocol` and reproduces the
behavior the bootstrap `ResourceManager` has today: everything is feasible,
nothing spills, every credit is granted, and the memory envelope is permissive.
These are the seam's default occupants; real policies replace them by being
constructed in their place.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from batcher.carbonite.memory.estimator import peak_operator_bytes
from batcher.carbonite.memory.pressure import total_memory_bytes
from batcher.config import Config, active_config
from batcher.plan.resource import FeasibilityVerdict, ResourceBounds, SchedulingEnvelope

if TYPE_CHECKING:
    from batcher.carbonite.base import ResourceContext
    from batcher.plan.physical import PhysicalPlan

__all__ = [
    "AIMDFlowControl",
    "BudgetingAdmission",
    "DefaultSchedulingPolicy",
    "StaticCreditFlowControl",
    "credit_ceiling",
]


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
        # Cross-query admission (C56): subtract what concurrent queries already hold
        # against the shared buffer pool, so N queries that each individually fit the
        # envelope are not all admitted into a collective OOM.
        if self._available is None:
            from batcher.carbonite.memory.pool import current_process_pool

            pool = current_process_pool()
            if pool is not None:
                envelope = max(0, envelope - pool.used)
        peak = max((op.bounds.m_max_bytes for op in plan.ops), default=0)
        if peak <= envelope:
            return FeasibilityVerdict(feasible=True)
        # Over budget: offer the envelope as the per-operator bound so the engine can
        # re-plan with a spill-friendly strategy instead of OOMing.
        return FeasibilityVerdict(
            feasible=False,
            binding_constraint="memory",
            suggested_bounds=ResourceBounds(
                m_max_bytes=envelope, c_max_credits=0, n_max_parallelism=0
            ),
        )


def credit_ceiling(config: Config) -> int:
    """The upper bound on a shuffle channel's credit window (count *and* bytes).

    The count ceiling (`default_credits x credit_ceiling_factor`) is further capped
    so the window's *bytes* (`credits x morsel_bytes`) never exceed
    `credit_byte_budget` — bounding a channel's buffered memory regardless of row
    width (C53). Always >= 1.
    """
    fc = config.flow_control
    count_ceiling = fc.default_credits * fc.credit_ceiling_factor
    morsel_bytes = max(1, config.execution.morsel_bytes)
    byte_ceiling = max(1, fc.credit_byte_budget // morsel_bytes)
    return max(1, min(count_ceiling, byte_ceiling))


class StaticCreditFlowControl:
    """Credit-window flow control: clamp the requested window to a memory-safe band.

    This is the Carbonite authority that replaces the engine's hardcoded
    `DEFAULT_CREDITS`: one credit = one in-flight `RecordBatch` slot, so the window
    directly bounds a shuffle channel's buffered memory. The window comes from
    `FlowControlConfig`: a non-positive request (operator with no `c_max_credits`
    estimate) gets `default_credits`; a positive request is clamped into
    `[1, credit_ceiling(config)]` (a count *and* byte bound) so neither a stale zero
    stalls the channel nor an over-large estimate (or a wide-row morsel) lets a fast
    producer run unbounded.
    """

    def grant(self, requested: int, ctx: ResourceContext) -> int:
        fc = ctx.config.flow_control
        ceiling = credit_ceiling(ctx.config)
        if requested <= 0:
            return min(fc.default_credits, ceiling)
        return min(max(requested, 1), ceiling)


class AIMDFlowControl:
    """Adaptive credit window via AIMD (additive-increase / multiplicative-decrease).

    The static policy fixes the window; this one *adapts* it from observed
    backpressure, the TCP-style control law the architecture specifies. It starts at
    the config default window and, per round, `observe`s whether the channel was
    congested: a congested round cuts the window by `aimd_beta` (relieve memory
    pressure fast), an uncongested round grows it by `aimd_alpha` (pipeline deeper
    while memory is plentiful). The window is always clamped to the same memory-safe
    band `[1, default_credits x credit_ceiling_factor]` the static policy uses.

    Stateful — hold one per adaptive channel. `grant` ignores its `requested`
    argument because the controller, not the caller, owns the evolving window.
    """

    def __init__(self, config: Config | None = None) -> None:
        cfg = config or active_config()
        fc = cfg.flow_control
        self._alpha = max(1, fc.aimd_alpha)
        self._beta = fc.aimd_beta
        self._floor = 1
        self._ceiling = credit_ceiling(cfg)  # count + byte bound (C53)
        self._window: float = float(min(max(fc.default_credits, self._floor), self._ceiling))

    @property
    def window(self) -> int:
        """The current credit window (clamped to the band)."""
        return self._clamp(self._window)

    def grant(self, requested: int, ctx: ResourceContext) -> int:  # noqa: ARG002
        return self.window

    def observe(self, *, congested: bool) -> int:
        """Update the window from one round's congestion signal; return the new window.

        `congested` is true when the round hit backpressure (e.g. the producer ran
        the window full, or memory pressure was high): cut multiplicatively. Else the
        consumer kept up with headroom to spare: grow additively.
        """
        if congested:
            self._window = max(self._floor, self._window * self._beta)
        else:
            self._window = min(self._ceiling, self._window + self._alpha)
        return self.window

    def _clamp(self, w: float) -> int:
        return int(max(self._floor, min(self._ceiling, w)))


class DefaultSchedulingPolicy:
    """Derive a per-Ray-task `SchedulingEnvelope` from Kyber's per-operator bounds.

    This is where worker fan-out stops being a blind `os.cpu_count()` and starts
    tracking the data: a breaker's `n_max_parallelism` (≈ rows / target-rows-per-task)
    sets the desired task count, clamped to the machine's cpu budget. Per-task memory
    is the dominant breaker's footprint split across those tasks (each holds one
    partition's share), clamped to a fair slice of the live budget so a soft Ray
    `memory=` hint never over-asks. `num_cpus` is the configured per-task share; GPUs
    are 0 here (the GPU map/inference path sets its own `num_gpus`). Credits are filled
    by the manager from its flow-control policy.
    """

    def envelope(
        self,
        plan: PhysicalPlan,
        ctx: ResourceContext,
        *,
        requested_workers: int | None,
        available_bytes: int,
    ) -> SchedulingEnvelope:
        cfg = ctx.config
        # Local fallback only — used when the plan carries no data-driven fan-out.
        # NOT a clamp on the data-driven want: this envelope is consumed only by the
        # distributed path, where the *cluster*-aware `clamp_workers` owns the real
        # cap. Clamping the desired fan-out to the driver's core count here would
        # cap a 100-node job at the driver's cores (the bug N11 fixes).
        cpu_budget = max(1, cfg.execution.parallelism or os.cpu_count() or 4)

        # Desired parallelism: the widest breaker request (≈ rows / target-rows). An
        # explicit user `requested_workers` always wins; an unsized/streaming plan
        # (no breaker estimate) falls back to the local cpu budget. The data-driven
        # `desired` is passed through un-clamped — `clamp_workers` reduces it to live
        # cluster capacity downstream.
        desired = max((op.bounds.n_max_parallelism for op in plan.ops), default=0)
        if requested_workers and requested_workers > 0:
            n_tasks = requested_workers
        elif desired > 0:
            n_tasks = desired
        else:
            n_tasks = cpu_budget
        n_tasks = max(1, n_tasks)

        # Per-task memory: the dominant breaker split across tasks, never below one
        # morsel and never above a fair share of the live budget. 0 (no hint) when
        # Kyber could not size the plan.
        peak = peak_operator_bytes(plan)
        morsel_bytes = max(1, cfg.execution.morsel_rows * cfg.optimizer.row_bytes)
        if peak <= 0:
            memory_bytes = 0
        else:
            per_task = max(morsel_bytes, peak // n_tasks)
            fair_share = (
                max(morsel_bytes, available_bytes // n_tasks) if available_bytes > 0 else per_task
            )
            memory_bytes = min(per_task, fair_share)

        return SchedulingEnvelope(
            num_cpus=cfg.execution.cpus_per_task,
            memory_bytes=int(memory_bytes),
            num_gpus=0.0,
            n_tasks=n_tasks,
            credits=cfg.flow_control.default_credits,
        )
