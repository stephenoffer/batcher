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
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

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
    PressureLevel,
    hysteresis_alpha_from_flap,
    load_flap_rate,
)
from batcher.carbonite.policies import (
    AIMDFlowControl,
    BudgetingAdmission,
    DefaultSchedulingPolicy,
    StaticCreditFlowControl,
    credit_ceiling,
    learned_channel_morsel_bytes,
    load_shuffle_window,
)
from batcher.config import Config, active_config
from batcher.metadata import MetadataHub
from batcher.plan.physical import PhysicalPlan
from batcher.plan.resource import FeasibilityVerdict, SchedulingEnvelope

__all__ = ["ResourceManager"]

# How far to shrink the morsel target at each pressure level (adaptive morsel sizing).
# NORMAL keeps the configured target (no entry ⇒ factor 1.0); ELEVATED halves it;
# SPILL/CRITICAL quarter it so the streaming working set stays tight while the engine
# is already under pressure.
_MORSEL_PRESSURE_FACTORS = {
    PressureLevel.ELEVATED: 0.5,
    PressureLevel.SPILL: 0.25,
    PressureLevel.CRITICAL: 0.25,
}
_MIN_MORSEL_ROWS = 1024  # floor: a morsel never shrinks below a cache-efficient batch
_MIN_MORSEL_BYTES = 64 * 1024  # 64 KiB floor (companion byte bound)

# Learned spill-partition sizing: aim each out-of-core bucket at roughly this many bytes
# of the LEARNED peak, so a bigger measured working set shards into more, smaller buckets
# (bounded memory per bucket) and a small one stays coarse. Only *shards* — the shuffle
# is result-invariant in the number of partitions.
_SPILL_BYTES_PER_PARTITION = 128 * 1024 * 1024  # 128 MiB target per spill bucket
_MIN_SPILL_PARTITIONS = 2
_MAX_SPILL_PARTITIONS = 4096
# Above this learned peak a spill bucket compresses well enough to be worth the CPU:
# a large out-of-core state is IO-bound, so trading CPU for less disk/network wins.
_SPILL_COMPRESS_ABOVE = 512 * 1024 * 1024  # 512 MiB


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
                # Honor the byte bound (`credit_byte_budget`) via the learned wide-row width,
                # exactly as the static grant does — otherwise a wide-row shuffle's learned
                # window would be clamped only by the un-corrected count ceiling and buffer far
                # past the byte budget (C53). Cold/narrow model → the plain count ceiling.
                ceiling = credit_ceiling(self._config, learned_channel_morsel_bytes(self._ctx))
                return max(1, min(learned, ceiling))
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
        # `credit_byte_budget` (C53) just because it took the adaptive path.
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
        cfg = self._config.execution
        # A pure read: the AIMD round is the one component that *samples* the monitor
        # (advancing its de-escalation average). Sizing a morsel must not.
        factor = _MORSEL_PRESSURE_FACTORS.get(self._pressure.classify(), 1.0)
        rows = int(cfg.morsel_rows * factor)
        nbytes = int(cfg.morsel_bytes * factor)
        learned_rows = self._learned_morsel_rows(families)
        if learned_rows is not None:
            rows = min(rows, learned_rows)
        # Keep the configured target (fast path) only when neither lever moved anything.
        if factor >= 1.0 and rows >= cfg.morsel_rows:
            return None
        return max(_MIN_MORSEL_ROWS, rows), max(_MIN_MORSEL_BYTES, nbytes)

    def _learned_morsel_rows(self, families: Iterable[str] | None = None) -> int | None:
        """Row cap that keeps a morsel's *measured* byte working set within the budget.

        Uses the widest learned per-row footprint (``rows = morsel_bytes /
        max_bytes_per_row``), restricted to `families` — *this plan's* operator kinds — when
        given, so a narrow plan is sized by its own data rather than throttled by an
        unrelated wide family measured in an earlier query. `None` when nothing is learned
        yet or the learned width is no wider than the configured target already implies (so
        the common case adds no overhead and no change)."""
        widths = self._mem_model.max_bytes_per_row(families)
        if widths is None or widths <= 0:
            return None
        cap = int(self._config.execution.morsel_bytes / widths)
        if cap >= self._config.execution.morsel_rows:
            return None  # learned width is no wider than assumed — nothing to tighten
        return max(_MIN_MORSEL_ROWS, cap)

    def recommended_config(self, families: Iterable[str] | None = None) -> Config | None:
        """A `Config` with the pressure-scaled morsel target, or ``None`` to keep the
        current one. The conductor activates it for the execution scope so the adapted
        morsel reaches both the in-process engine and the shipped worker config. `families`
        (the plan's operator kinds) narrows the learned width to this plan's own data."""
        target = self.recommend_morsel_target(families)
        if target is None:
            return None
        rows, nbytes = target
        execution = dataclasses.replace(
            self._config.execution, morsel_rows=rows, morsel_bytes=nbytes
        )
        return dataclasses.replace(self._config, execution=execution)

    def _peak_bytes(self, plan: PhysicalPlan) -> int:
        """The plan's peak in-memory bytes (learned-blended), computed once per plan.

        `estimated_bytes`, `should_spill`, and `reserve` all consult this, so the
        per-plan envelope is built once rather than three times (C37). The estimator
        blends each operator's plan estimate toward its *measured* peak (learned from
        `m_peak_bytes`) when the hub has one, so all three decisions size against reality;
        on a cold store it is exactly the plan's dominant breaker.
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

    def recommend_spill_partitions(self, plan: PhysicalPlan) -> int | None:
        """Number of out-of-core buckets to shard `plan`'s spilled state into, or ``None``.

        Sizes the bucket count so each bucket holds ~`_SPILL_BYTES_PER_PARTITION` of the
        LEARNED peak (`m_peak_bytes`-blended, not the plan guess): a larger measured
        working set shards into more, smaller buckets so per-bucket memory stays bounded,
        a small one stays coarse. Returns ``None`` when the plan is un-sized (nothing
        learned or estimated) so the caller keeps its default partition count. The number
        of partitions only *shards* the shuffle — the result is identical for any count
        (the mergeable algebra), so this is a pure performance lever.

        NOTE for the orchestrator: consume this in place of the blind
        `partitions_from_physical` fallback to right-size out-of-core sharding.
        """
        peak = self._peak_bytes(plan)
        if peak <= 0:
            return None
        # Prefer the *measured* spill volume when a family has a spill history: buckets shard
        # only the bytes that actually go to disk, which is smaller than the total working-set
        # peak (which includes the in-memory budget that never spills). Fall back to peak when
        # nothing has spilled yet. Result-invariant either way — this only sets bucket count.
        volume = self._mem_model.predicted_spill_bytes(plan.ops)
        basis = volume if volume > 0 else peak
        parts = max(_MIN_SPILL_PARTITIONS, -(-basis // _SPILL_BYTES_PER_PARTITION))  # ceil-div
        return min(_MAX_SPILL_PARTITIONS, int(parts))

    def recommend_spill_compression(self, plan: PhysicalPlan) -> bool | None:
        """Whether spilling `plan` should compress its buckets, from the learned peak.

        A large out-of-core state is IO-bound (disk / object-store), so above
        `_SPILL_COMPRESS_ABOVE` of measured peak, trading CPU for less bytes on the wire
        pays; below it the CPU isn't worth it. Compression is lossless — the un-spilled
        result is byte-identical either way — so this is a pure throughput lever. Returns
        ``None`` for an un-sized plan (keep the configured default).

        NOTE for the orchestrator: map this onto `memory.spill_compression` when spilling.
        """
        peak = self._peak_bytes(plan)
        if peak <= 0:
            return None
        return peak >= _SPILL_COMPRESS_ABOVE

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
