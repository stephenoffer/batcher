"""The Carbonite resource manager entry point.

Validates plans for feasibility, hands out credit windows and memory reservations,
and decides when a query must spill. It is a thin orchestrator: it composes one policy
of each of the four kinds `carbonite.base` declares (admission, flow control, memory
estimation, scheduling), the `SpillAdvisor` that owns every out-of-core decision, and
the memory subsystem (buffer pool + pressure monitor), and delegates to them.

`validate` returns real counter-offers Kyber re-plans around; `reserve` accounts against
the process-wide buffer pool; `should_spill` — and `spill_reason`, which says *why* —
compare a plan's estimated envelope and live memory so a large query goes out-of-core
instead of OOMing; `stats` reads all of it back in one snapshot. An alternate policy plugs
in by being passed to the constructor, and every delegation goes through it, including the
learned-warm-start paths.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from batcher._internal.hardware import available_cpu_count
from batcher.carbonite.base import (
    AdmissionPolicy,
    FlowControlPolicy,
    MemoryEstimator,
    ResourceContext,
    SchedulingPolicy,
)
from batcher.carbonite.memory import OperatorMemoryEstimator, PressureMonitor, process_pool
from batcher.carbonite.memory.learned import LearnedMemoryModel, learned_memory_model
from batcher.carbonite.memory.pressure import (
    hysteresis_alpha_from_flap,
    load_flap_rate,
)
from batcher.carbonite.policies import (
    AIMDFlowControl,
    BudgetingAdmission,
    DefaultSchedulingPolicy,
    StaticCreditFlowControl,
    learned_channel_morsel_bytes,
    load_shuffle_window,
)
from batcher.carbonite.policies.cpu_budget import effective_core_budget
from batcher.carbonite.policies.morsel import (
    MIN_MORSEL_BYTES,
    MIN_MORSEL_ROWS,
    PRESSURE_FACTORS,
    morsel_target,
)
from batcher.carbonite.policies.spill_advice import SpillAdvisor
from batcher.carbonite.policies.spill_shape import (
    MAX_SPILL_PARTITIONS,
    MIN_SPILL_PARTITIONS,
    SPILL_BYTES_PER_PARTITION,
    SPILL_COMPRESS_ABOVE,
)
from batcher.config import Config, active_config
from batcher.metadata import MetadataHub
from batcher.plan.physical import PhysicalPlan
from batcher.plan.resource import FeasibilityVerdict, ResourceBounds, SchedulingEnvelope

__all__ = ["ResourceManager"]

# Re-exported under their historical private names: the morsel-sizing rules moved to
# `policies.morsel`, but tests and callers name them from here.
_MORSEL_PRESSURE_FACTORS = PRESSURE_FACTORS
_MIN_MORSEL_ROWS = MIN_MORSEL_ROWS
_MIN_MORSEL_BYTES = MIN_MORSEL_BYTES

# Re-exported under their historical private names: the sizing rules moved to
# `policies.spill_shape`, but callers and tests name them from here.
_SPILL_BYTES_PER_PARTITION = SPILL_BYTES_PER_PARTITION
_MIN_SPILL_PARTITIONS = MIN_SPILL_PARTITIONS
_MAX_SPILL_PARTITIONS = MAX_SPILL_PARTITIONS
_SPILL_COMPRESS_ABOVE = SPILL_COMPRESS_ABOVE


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
        hub: MetadataHub | None = None,
        admission: AdmissionPolicy | None = None,
        flow_control: FlowControlPolicy | None = None,
        memory: MemoryEstimator | None = None,
        scheduling: SchedulingPolicy | None = None,
    ) -> None:
        self._config = config or active_config()
        self._hub = hub
        # The learned per-family memory model (fit from measured `m_peak_bytes`). Empty /
        # pass-through on a cold store, so every sizing decision below defaults to the plan
        # estimate until the hub has absorbed real peaks. NOTE for the orchestrator: pass
        # `ctx.hub` here (as `collect_source_stats`/`optimize_full` already receive it) to
        # activate learned sizing end-to-end; without it the manager is byte-for-byte the
        # plan-only behavior.
        self._mem_model: LearnedMemoryModel = learned_memory_model(hub, self._config)
        # De-escalation hysteresis adapted from the measured flap rate (stickier level for
        # a channel that has been observed to oscillate); the static default on a cold store.
        alpha = hysteresis_alpha_from_flap(load_flap_rate(hub))
        self._pressure = PressureMonitor(self._config, hysteresis_alpha=alpha)
        # Sample the query's memory envelope ONCE so admission, spill, and reserve
        # all reason about the same figure (no live-RAM drift between decisions).
        self._envelope = self._pressure.envelope_bytes()
        self._ctx = ResourceContext(
            config=self._config, envelope_bytes=self._envelope, memory_model=self._mem_model
        )
        self._admission = admission or BudgetingAdmission()
        self._flow_control = flow_control or StaticCreditFlowControl()
        self._memory = memory or OperatorMemoryEstimator()
        self._scheduling = scheduling or DefaultSchedulingPolicy()
        # Every out-of-core decision, sized off one peak (cached per plan) and one budget.
        self._spill = SpillAdvisor(
            self._config, self._ctx, self._memory, self._mem_model, self._pressure, self._envelope
        )

    def validate(self, plan: PhysicalPlan) -> FeasibilityVerdict:
        """Check whether `plan` can run within available resources.

        The default `BudgetingAdmission` compares each operator's estimated memory
        (Kyber's per-operator `ResourceBounds`) against a soft fraction of physical
        RAM and returns a spill-friendly counter-offer when the dominant breaker
        would not fit. Conservative: unknown-size operators are not budgeted, so a
        legitimate query is never failed on a guess.
        """
        return self._admission.validate(plan, self._ctx)

    def grant_credits(self, requested: int, *, signature: str | None = None) -> int:
        """Grant a credit window (in-flight `RecordBatch` slots) for a data channel.

        One credit = one buffered batch, so the returned window bounds a shuffle
        channel's memory. The default `StaticCreditFlowControl` clamps `requested`
        (typically an operator's `ResourceBounds.c_max_credits`) into a memory-safe
        band derived from `FlowControlConfig`; this is the single authority that
        replaces the engine's hardcoded `DEFAULT_CREDITS`. Always returns >= 1 so a
        channel never stalls at zero credits.

        When `signature` names a shuffle channel that past runs converged on (a learned
        window recorded via `record_shuffle_window`), that window is granted instead of
        the plan request — clamped to the same memory-safe band — so a recurring shuffle
        starts near its measured sweet spot. A credit window only bounds in-flight
        buffering; it never changes the result. Cold signature → the plan-request path.
        """
        if signature is not None:
            learned = load_shuffle_window(self._hub, signature)
            if learned is not None and learned > 0:
                requested = learned
        # Always through the policy, including the learned path. Clamping the learned window
        # here instead — which is what this did — reproduced `StaticCreditFlowControl`'s band
        # inline and so silently bypassed whichever policy the manager was actually
        # constructed with: a deployment that supplied its own flow control got it for every
        # cold channel and lost it for exactly the recurring ones it had tuned for. The
        # static policy applies the identical clamp, so the default path is unchanged.
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
        # The plan's widest credit request, taken from the memory estimator's envelope so
        # this and every other consumer of "what does this plan want" read one derivation
        # rather than each re-scanning `plan.ops` with its own default.
        max_credits = self._memory.envelope(plan, self._ctx).c_max_credits
        return dataclasses.replace(env, credits=self.grant_credits(max_credits))

    def adaptive_flow_control(self, *, signature: str | None = None) -> AIMDFlowControl:
        """Vend an AIMD credit controller for an adaptive shuffle channel.

        The driver-side `grant_credits` sets the *initial* window from the operator's
        estimate; a long-lived channel can instead hold one of these and grow/shrink
        the window per round from observed backpressure (the `ShuffleSession`'s
        opt-in adaptive mode). Stateful — one controller per channel.

        When `signature` names a recurring shuffle with a learned converged window, the
        controller warm-starts there instead of re-climbing from `default_credits`; the
        AIMD law governs from that point on, so the window it actually uses (and thus the
        result) is unchanged — only its starting point moves. Cold signature → the default
        start."""
        initial = load_shuffle_window(self._hub, signature) if signature is not None else None
        # Thread the learned wide-row width so AIMD's ceiling keeps the same byte bound the
        # static grant enforces — a wide-row channel must not grow its window past
        # `credit_byte_budget` just because it took the adaptive path.
        return AIMDFlowControl(
            self._config,
            initial_window=initial,
            effective_morsel_bytes=learned_channel_morsel_bytes(self._ctx),
        )

    def recommend_morsel_target(
        self, families: Iterable[str] | None = None
    ) -> tuple[int, int] | None:
        """Scale the per-morsel ``(rows, bytes)`` target down under memory pressure.

        Returns the recommended target, or ``None`` to keep the configured one (the
        common, unpressured case). A morsel only *batches* data — it never changes the
        result — so shrinking it is always safe; under pressure a smaller morsel just
        keeps the streaming working set tighter so the engine stays in memory longer
        before it must spill (the "size blocks to memory" lever). The reduction tracks
        the live `PressureMonitor` level and is floored so a morsel never degrades into
        an inefficiently tiny batch.

        Beyond pressure, the *row* count is also capped so the morsel's real byte working
        set — its rows times the widest LEARNED per-row footprint (`m_peak_bytes / rows`,
        measured, not the assumed `optimizer.row_bytes`) — stays within the byte budget.
        A workload whose rows proved far wider than assumed (embeddings, blobs) thus gets a
        row-count that keeps its true working set bounded even before RAM is pressured. A
        morsel only batches data, so this never changes the result; a cold store learns
        nothing and leaves the count at the pressure-only value (``None`` when unpressured).
        """
        # `classify()`, a pure read: the AIMD round is the one component that *samples* the
        # monitor (advancing its de-escalation average). Sizing a morsel must not.
        return morsel_target(self._config, self._pressure.classify(), self._mem_model, families)

    def recommend_parallelism(self) -> int | None:
        """Cores to fan out across when the machine is measurably oversubscribed, else `None`.

        The CPU counterpart to `recommend_morsel_target`. Memory has a pressure monitor, a
        budget, and a spill path; CPU had only the permitted core count, which says what the
        cgroup allows and nothing about what the scheduler delivers. When two workers share a
        node, a quota is binding, or a co-tenant has half the box, fanning out to the permitted
        count adds context switches and cache pressure without adding throughput — the query
        gets slower the harder the engine tries.

        `None` on a quiet machine (the common case), so the engine keeps its own default and
        nothing changes. See `policies.cpu_budget` for the deadband and the floor.

        Returns:
            The reduced core budget, or `None` to leave parallelism alone.
        """
        configured = self._config.execution.parallelism
        if configured > 0:
            return None  # an explicit setting is an instruction, not an estimate
        budget = effective_core_budget()
        return budget if budget < available_cpu_count() else None

    def recommended_config(self, families: Iterable[str] | None = None) -> Config | None:
        """A `Config` with the pressure-scaled morsel target and contention-scaled fan-out,
        or ``None`` to keep the current one. The conductor activates it for the execution
        scope so the adaptation reaches both the in-process engine and the shipped worker
        config. `families` (the plan's operator kinds) narrows the learned width to this
        plan's own data."""
        target = self.recommend_morsel_target(families)
        parallelism = self.recommend_parallelism()
        if target is None and parallelism is None:
            return None
        changes: dict[str, object] = {}
        if target is not None:
            changes["morsel_rows"], changes["morsel_bytes"] = target
        if parallelism is not None:
            changes["parallelism"] = parallelism
        execution = dataclasses.replace(self._config.execution, **changes)
        return dataclasses.replace(self._config, execution=execution)

    def estimated_bytes(self, plan: PhysicalPlan) -> int:
        """Estimated peak in-memory bytes for `plan` — its learned-blended dominant breaker.

        The figure `reserve` accounts and the spill gate compares against the budget.
        `0` when Kyber emitted no sizes. See `policies.spill_advice.SpillAdvisor`.
        """
        return self._spill.peak_bytes(plan)

    def should_spill(self, plan: PhysicalPlan) -> bool:
        """Whether `plan` should run out-of-core rather than in memory.

        The boolean form of `spill_reason`, which documents the two signals behind it.
        """
        return self._spill.spill_reason(plan) is not None

    def spill_reason(self, plan: PhysicalPlan) -> str | None:
        """*Why* `plan` must go out-of-core, or `None` when it need not.

        An estimate over budget is fixed by reshaping the plan; live pressure is fixed by
        finding what else holds memory. See `policies.spill_advice.SpillAdvisor`.
        """
        return self._spill.spill_reason(plan)

    def stats(self) -> dict[str, object]:
        """One reading of every resource Carbonite governs, for telemetry and `explain`.

        The buffer pool, the result cache, the concurrency limiter, and the pressure
        monitor each measure their own corner, and each was readable only by reaching for
        the object that owns it. Reading them together is what makes them a diagnosis: a
        query that spilled with an empty pool and a full cache was starved by *storage*,
        and one that spilled with a full pool and no cache was simply too big.

        Returns:
            A nested dict keyed by subsystem. Corners that were never created (no pool, no
            cache) are omitted rather than reported as zero, which would be a lie about a
            thing that does not exist.
        """
        from batcher.carbonite.cache import current_result_cache
        from batcher.carbonite.memory.pool import current_process_pool

        level = self._pressure.classify()
        out: dict[str, object] = {
            "pressure_level": level.name,
            "envelope_bytes": self._envelope,
            "soft_budget_bytes": self._spill.soft_budget(),
            "hard_budget_bytes": self._spill.hard_budget(),
            "headroom_bytes": self._pressure.headroom_bytes(),
        }
        pool = current_process_pool()
        if pool is not None:
            out["pool"] = pool.stats()
        cache = current_result_cache()
        if cache is not None:
            out["result_cache"] = cache.stats()
        flap = self._pressure.flap_rate()
        if flap is not None:
            out["pressure_flap_rate"] = flap
        return out

    def input_exceeds_budget(self, input_bytes: int) -> bool:
        """Whether reading the sources whole would not fit the memory envelope.

        The dominant term for a scan-heavy query, and the one the plan estimate never
        covered. See `policies.spill_advice.SpillAdvisor.input_exceeds_budget`.
        """
        return self._spill.input_exceeds_budget(input_bytes)

    def recommend_spill_partitions(self, plan: PhysicalPlan) -> int | None:
        """Out-of-core buckets to shard `plan`'s spilled state into, or `None` to keep the
        caller's default. See `policies.spill_advice.SpillAdvisor.partitions`."""
        return self._spill.partitions(plan)

    def flap_rate(self) -> float | None:
        """This run's measured pressure-level flap rate, or `None` if too few samples.

        The producer side of the hysteresis loop: `__init__` reads a *past* run's rate back
        (`load_flap_rate` -> `hysteresis_alpha_from_flap`) to stiffen de-escalation for a
        workload that oscillates. Nothing measured it, so that read was always cold and the
        anti-oscillation mechanism never engaged. The conductor persists this at end of run.

        Returns:
            The fraction of sampled levels that reversed direction, or `None` when the
            monitor took fewer than two samples (nothing to conclude).
        """
        return self._pressure.flap_rate()

    def partitions_for_bounds(self, plan: PhysicalPlan, bounds: ResourceBounds | None) -> int:
        """Fewest spill buckets that make each bucket fit admission's counter-offer.

        The return leg of the Kyber-Carbonite contract; `0` when unconstrained. See
        `policies.spill_advice.SpillAdvisor.partitions_for_bounds`.
        """
        return self._spill.partitions_for_bounds(plan, bounds)

    def recommend_spill_compression(self, plan: PhysicalPlan) -> bool | None:
        """Whether spilling `plan` should compress its buckets, from the learned peak.

        `None` for an un-sized plan (keep the configured default). Lossless either way.
        See `policies.spill_advice.SpillAdvisor.compression`.
        """
        return self._spill.compression(plan)

    def _hard_budget(self) -> int:
        """Bytes a query may hold before it must spill — the one figure the spill gate and
        the reservation share, so the two can never disagree."""
        return self._spill.hard_budget()

    @contextmanager
    def admit(self) -> Iterator[object]:
        """Hold an execution slot for the block, sized to what else is running.

        A context manager rather than a `validate`-style call because the slot must be
        held for the *duration* of execution, and a function that returns a verdict cannot
        bracket that — the query would be counted as running for as long as it took to
        decide, which is not the thing being bounded.

        Yields an `ExecutionGrant` whose `workers` is the rayon pool width this query
        should request. `workers=0` means unbounded, which is what a single query gets and
        what every query gets when `execution.max_concurrent_queries` is 0 (the default) —
        so an unconfigured deployment executes exactly as it did before this existed.

        Yields:
            The `ExecutionGrant` for this query.

        Raises:
            AdmissionTimeout: If the queue is full or the wait deadline passes.
        """
        from batcher.carbonite.policies.concurrency import process_limiter

        limiter = process_limiter(self._config)
        grant = limiter.acquire(timeout=max(0.0, self._config.execution.admission_timeout_s))
        try:
            yield grant
        finally:
            limiter.release()

    @contextmanager
    def reserve(self, m_bytes: int) -> Iterator[bool]:
        """Reserve `m_bytes` against the process-wide buffer pool for the block.

        Accounts the reservation on entry and releases it on exit (even if the
        block raises), so concurrent queries and the transfer layer see a single
        shared envelope. Yields whether the reservation fit; a `False` means the
        pool is already over budget and the caller should be on the spill path.
        The pool is sized to Carbonite's hard memory envelope — the same figure
        `should_spill` compares against, so the two decisions stay consistent.

        Storage yields to execution, in two steps — the two halves of Spark's unified
        memory model, which the cache implements but nothing used to invoke:

        1. **The pressure ladder.** `CacheStore.on_pressure` trims the cache as RSS
           climbs (three-quarters at `ELEVATED`, half at `SPILL`, everything at
           `CRITICAL`). The cache's own bytes are *not* accounted against the buffer
           pool, so without this the two budgets are disjoint and
           `result_cache_max_bytes + memory envelope` can exceed physical RAM — the
           cache silently shrinks the headroom every other Carbonite decision assumes.
           `classify()` (not `level()`) reads the level, so this does not consume the
           AIMD round's sample.
        2. **The exact deficit.** When the pool is still tighter than the request, the
           cache drops *precisely* the shortfall (lowest-value entries first) so its RAM
           goes back to the running query rather than squeezing it.

        Evicting a cache is result-invariant — it only costs a recompute — so this can
        never change an answer, only the memory the query gets to run in.
        """
        from batcher.carbonite.cache import current_result_cache

        pool = process_pool(self._hard_budget())
        cache = current_result_cache()
        if cache is not None:
            cache.on_pressure(self._pressure.classify())
            deficit = m_bytes - pool.available
            if deficit > 0:
                cache.evict_to_free(deficit)
        with pool.reserve(m_bytes) as granted:
            yield granted
