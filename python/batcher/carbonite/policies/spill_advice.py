"""Whether a query goes out of core, and what shape its spilled state takes.

Every question here reduces to one number — the plan's peak in-memory bytes, blended
toward what the family *measured* — compared against one budget. Keeping them together is
what makes them agree: `should_spill`, `reserve`, and the partition count all reason about
the same peak and the same hard budget, and the way that guarantee is lost is by each
deriving its own.

`SpillAdvisor` holds that peak (computed once per plan) plus the budgets, and answers:

- **Does this need to spill at all**, and *why* (`spill_reason`)?
- **How wide** should the spilled state be sharded (`partitions`, `partitions_for_bounds`)?
- **Should the buckets compress** (`compression`)?

None of it can change a result. The mergeable algebra returns an identical merged result
for any partition count, the spill codec is lossless, and spilling itself only trades time
for bounded memory — so a wrong answer here costs latency, and the opposite wrong answer
costs the process. That asymmetry is why the measured signal is allowed to overrule an
absent estimate.

Split out of `ResourceManager` so the manager reads as the governor that composes the
policies rather than as the spill library itself.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from batcher._internal.logging import get_logger, log_kv, note_suppressed
from batcher.carbonite.memory.pressure import PressureLevel
from batcher.carbonite.policies.spill_shape import (
    SPILL_BYTES_PER_PARTITION,
    envelope_shortfall,
    partitions_for_envelope,
    partitions_for_volume,
    should_compress,
    spill_basis,
)

if TYPE_CHECKING:
    from batcher.carbonite.base import MemoryEstimator, ResourceContext
    from batcher.carbonite.memory.learned import LearnedMemoryModel
    from batcher.carbonite.memory.pressure import PressureMonitor
    from batcher.config import Config
    from batcher.plan.physical import PhysicalPlan
    from batcher.plan.resource import ResourceBounds

__all__ = ["SpillAdvisor"]


class SpillAdvisor:
    """The out-of-core decisions for one query, all sized off one peak and one budget."""

    def __init__(
        self,
        config: Config,
        ctx: ResourceContext,
        estimator: MemoryEstimator,
        model: LearnedMemoryModel,
        pressure: PressureMonitor,
        envelope_bytes: int,
        share: float = 1.0,
    ) -> None:
        self._config = config
        self._ctx = ctx
        self._estimator = estimator
        self._model = model
        self._pressure = pressure
        self._envelope = envelope_bytes
        # This query's entitlement to the envelope when several run at once (see
        # `policies.concurrency.query_memory_share`). Exactly `1.0` for the default
        # unbounded-concurrency deployment, so every budget below is unchanged there.
        self._share = min(1.0, max(0.0, share)) or 1.0
        # Single-entry envelope cache keyed by plan *identity* (a held reference, so
        # `is` is stable and the object can't be GC'd into an id collision).
        self._peak_plan: object = None
        self._peak_value = 0

    def peak_bytes(self, plan: PhysicalPlan) -> int:
        """The plan's peak in-memory bytes (learned-blended), computed once per plan.

        The spill decision, the partition count, and the manager's reservation all consult
        this, so the per-plan envelope is built once rather than three times. The estimator
        blends each operator's plan estimate toward its *measured* peak (learned from
        `m_peak_bytes`) when the hub has one, so every decision sizes against reality; on a
        cold store it is exactly the plan's dominant breaker.

        Args:
            plan: The annotated physical plan.

        Returns:
            The peak bytes, or `0` for a plan Kyber could not size.
        """
        if plan is not self._peak_plan:
            self._peak_value = self._estimator.envelope(plan, self._ctx).m_max_bytes
            self._peak_plan = plan
        return self._peak_value

    def spill_reason(self, plan: PhysicalPlan) -> str | None:
        """*Why* `plan` must go out-of-core, or ``None`` when it need not.

        Two independent signals, and the second is why the first is not enough. Kyber emits
        `0` for any operator whose cardinality is unknown, so an un-sized plan would take
        the "fits" fast path and run fully in memory no matter how much of the box was
        already gone. The estimate can therefore only ever *add* a spill; when it does not
        force one, the live **measured** footprint decides. That reading (cgroup-current
        else RSS) is the one number here that cannot be wrong the way an estimate can.

        The two causes call for opposite responses — an estimate over budget is about *this
        plan* and is fixed by reshaping it, while live pressure is about *the box* and is
        fixed by finding whatever else holds memory — so the reason is returned rather than
        collapsed into a boolean that cannot tell them apart.

        `classify()` (not `level()`) reads the level, so this does not consume the AIMD
        round's sample.

        Args:
            plan: The physical plan about to run.

        Returns:
            A short reason string, or `None` if the query fits.
        """
        estimated = self.peak_bytes(plan)
        budget = self.hard_budget()
        if estimated > 0 and estimated > budget:
            return f"estimated peak {estimated} B exceeds the {budget} B memory budget"
        level = self._pressure.classify()
        if level >= PressureLevel.SPILL:
            return f"live memory pressure is {level.name}"
        return self._oom_history_reason(estimated)

    def _oom_history_reason(self, estimated: int) -> str | None:
        """Spill an **un-sized** plan in a cgroup that has already been OOM-killed.

        The third signal, and the only one that is evidence rather than inference. The first
        two both go quiet in the same situation: Kyber emits `0` for an operator whose
        cardinality it cannot estimate, and live pressure is measured *now*, before the query
        that will cause the problem has allocated anything. A worker that was killed, restarted
        by the scheduler, and handed the same un-sized plan therefore takes the "fits" fast path
        straight back into the kill — and because the kill is a signal from the kernel rather
        than an exception, each iteration looks like a fresh cold start.

        Deliberately narrow. It applies only when the estimate is absent, because a plan Kyber
        *did* size has already been compared against the budget by the first signal and
        overruling that would spill queries that measurably fit. Spilling is result-invariant,
        so the cost of acting on stale evidence is latency; the cost of ignoring it is the
        process.
        """
        if estimated > 0:
            return None
        from batcher.carbonite.memory.kernel import kernel_memory_state

        state = kernel_memory_state()
        if not state.was_oom_killed:
            return None
        return (
            f"this container has been OOM-killed {state.oom_kills} time(s) and the plan is "
            "un-sized, so it goes out-of-core rather than repeating the kill"
        )

    def input_exceeds_budget(self, input_bytes: int) -> bool:
        """Whether reading the sources whole would not fit the memory envelope.

        The in-memory path resolves every source to a list of Arrow batches *before* the
        engine runs, so the input is resident in full no matter how small the result is —
        a `GROUP BY` returning four rows still materializes every projected column of
        every row. That makes the input, not the operator state, the dominant term for a
        scan-heavy query, and it is the one term the plan estimate never covered:
        `m_max_bytes` sizes an operator's *working set*, so a plan whose breakers are all
        small reads as "fits" while the scan feeding them does not.

        Args:
            input_bytes: A declared `row_count()` times the projected schema width —
                metadata only, so it costs no I/O. `0` means the sources could not size
                themselves, which is not evidence of fitting.

        Returns:
            True when the query should read through the bounded streaming tap instead.
        """
        return input_bytes > 0 and input_bytes > self.hard_budget()

    def resident_total_exceeds_budget(self, input_bytes: int, plan: PhysicalPlan) -> bool:
        """Whether the resident input **plus** the plan's peak operator state overflows the
        envelope.

        `input_exceeds_budget` and `should_spill` are two halves of one total, and each was
        compared against the whole budget on its own. Nothing summed them — yet on the
        in-memory path they are *concurrent*, not alternatives: the sources are resolved to
        Arrow batches before the engine starts and stay resident for the whole execution,
        while the breaker builds its state on top of them. A query whose input is 70% of the
        envelope and whose breaker is 70% of it passes both checks and needs 140%.

        Measured on a 24 M-row group-by under a 537 MB envelope: input 384 MB, live partial
        state 384 MB, neither over the budget alone, both over it together — and the query
        stayed on the in-memory path and peaked at 2.4 GB.

        Summing is the right reading of the in-memory path specifically, and the double-count
        worry does not apply: `peak_bytes` is an operator's *working set*, so a sort's peak is
        its output and an aggregate's is its partials, in both cases memory that lives
        alongside the input rather than replacing it.

        A `0` input stays "no evidence" rather than "fits", as it is for
        `input_exceeds_budget`: an unsizable source must not be read as a small one. The other
        signals (`should_spill`, live pressure) still apply in that case.

        Args:
            input_bytes: Metadata-only estimate of the resident input, or `0` for unknown.
            plan: The physical plan about to run.

        Returns:
            True when the two together do not fit, so the query should go out of core.
        """
        if input_bytes <= 0:
            return False
        widest = self._widest_intermediate(input_bytes, plan)
        # `2 *`, because the materializing path holds two of these at once, and for the same
        # reason on both terms. A mergeable operator holds its partials *and* the merged
        # result while `combine` runs, so its high-water mark is about twice the state
        # `peak_bytes` reports -- which is the final state. A row-wise operator holds its
        # input while it builds its output, so at the hand-off both are resident. Neither is
        # a safety factor; both are facts about how the executor runs.
        #
        # It matters because the decision is irreversible. Once the sources are resolved and
        # the engine is running there is no route back, so an estimate that lands 10% low is
        # not a slow query, it is a dead process. TPC-H q18's first staged sub-plan -- a
        # `GROUP BY l_orderkey` over 600 M rows -- estimated 8.94 GiB of input and 8.94 GiB of
        # state against a 19.92 GiB budget, read 17.88, said "fits", and was OOM-killed at
        # 19.34 GiB. Doubled, it routes out of core and returns its 100 rows at **7.95 GiB**.
        #
        # TPC-H q13 is the row-wise half of the same story. Its `Filter` and `Project` over
        # `orders` each carry 142.5 M rows and are both live at the hand-off between them;
        # counting one read 13.75 GiB against a 14.41 GiB budget and fit by 0.66 GiB, where
        # the query really peaks at 18.50 GiB. Counting both, it routes out of core.
        #
        # The cost is real and worth stating: this is the trade the module already names --
        # over-estimating costs latency, under-estimating costs the process -- taken
        # deliberately rather than by accident.
        return input_bytes + 2 * max(0, self.peak_bytes(plan), widest) > self.hard_budget()

    def _widest_intermediate(self, input_bytes: int, plan: PhysicalPlan) -> int:
        """Bytes of the largest operator *output* the materializing path holds resident.

        `peak_bytes` is the dominant **breaker state**, and that is not the dominant resident
        term. The in-memory path runs the materializing executor, which holds every
        operator's full output -- and Kyber sizes the row-wise operators at `m_max_bytes = 0`,
        correctly, because a `Filter` and a `Project` retain no state. They retain no state
        and still occupy memory, because their output is materialized before the operator
        above reads it. So the plan's largest resident object is routinely one that nothing
        in the envelope arithmetic mentions.

        TPC-H q4 at sf100 is that shape. Its `Filter` and `Project` over `lineitem` carry
        292,422,301 rows and are both sized `0.00G`; the `Join` above them is sized 2.18 GiB
        and is the whole of `peak_bytes`. Against a 15.65 GiB input and a 19.02 GiB budget the
        gate read 17.83 GiB, said "fits", and the query was OOM-killed at 21.26 GiB. Forced
        out of core the same query peaks at **1.07 GiB** -- so the mistake cost the process,
        and avoiding it costs a query that was already at the edge of the envelope a run out
        of core.

        Sized from the *measured* input rather than from `row_size`, which is the estimate
        that cannot be used here: `row_size` is the operator's unprojected row width (292
        bytes for that `Filter`, against the ~22 bytes actually read), so multiplying by it
        over-reads by an order of magnitude and would push every large scan out of core.
        `input_bytes` already reflects pushed projections, so bytes-per-scanned-row derived
        from it is the width the reader will really produce.

        It can only ever *add* a spill: the caller takes a `max` against the existing term, so
        a plan this does not fire on decides exactly as it did before.

        Args:
            input_bytes: The projected resident input, the same figure the caller sums.
            plan: The annotated physical plan.

        Returns:
            The widest intermediate in bytes, or `0` when the plan carries no usable row
            estimates -- where an absent number must not read as a small one.
        """
        scans = [
            op.properties.est_rows
            for op in plan.ops
            if op.kind.lower() == "scan" and op.properties.est_rows == op.properties.est_rows
        ]
        # Scans are excluded from the max, and that is not a detail: a scan's output *is*
        # the resident input, which the caller has already counted, so including it charges
        # the input twice and reports the widest intermediate as the largest table in the
        # query. On q4 that read 13.4 GiB where the real widest is 6.5 GiB -- right verdict,
        # wrong reason, and wrong on any plan where the double-count is what tipped it.
        above = [
            op.properties.est_rows
            for op in plan.ops
            if op.kind.lower() != "scan" and op.properties.est_rows == op.properties.est_rows
        ]
        scan_rows = sum(scans)
        widest_rows = max(above, default=0.0)
        if scan_rows <= 0 or widest_rows <= 0:
            return 0
        return int(widest_rows * (input_bytes / scan_rows))

    def partitions(self, plan: PhysicalPlan) -> int | None:
        """Out-of-core buckets to shard `plan`'s spilled state into, or ``None``.

        Shards by the LEARNED peak (`m_peak_bytes`-blended, not the plan guess). The
        per-bucket target is the *smaller* of the sizing constant and the configured
        `memory.spill_bucket_max_bytes` — the size above which the reduce re-partitions a
        bucket by grace recursion. Sharding above that ceiling produced buckets the reduce
        then had to split again, re-reading and re-writing the whole spilled state for a
        figure that was known before the first partition was written.

        Args:
            plan: The physical plan about to be routed out-of-core.

        Returns:
            The bucket count, or `None` when the plan is un-sized so the caller keeps its
            default.
        """
        peak = self.peak_bytes(plan)
        if peak <= 0:
            return None
        return partitions_for_volume(
            spill_basis(peak, self._spill_volume(plan)), self._bucket_target_bytes()
        )

    def partitions_for_bounds(self, plan: PhysicalPlan, bounds: ResourceBounds | None) -> int:
        """Fewest spill buckets that make each bucket fit `bounds`, or ``0`` if unconstrained.

        The **return leg** of the Kyber↔Carbonite contract. When admission refuses a plan it
        does not merely say "no": it attaches a `suggested_bounds` counter-offer naming the
        per-operator byte envelope the plan *would* fit in. Sharding by that envelope is
        what makes the counter-offer binding, instead of sharding by a fixed constant that
        knows nothing about the machine's budget — which on a memory-tight host produces
        buckets that individually still do not fit, the exact failure admission diagnosed.

        Args:
            plan: The physical plan about to be routed out-of-core.
            bounds: Carbonite's counter-offer, or `None` when admission raised no objection.

        Returns:
            The minimum bucket count, or `0` when there is no bound or the plan is un-sized.
        """
        if bounds is None:
            return 0
        basis = spill_basis(self.peak_bytes(plan), self._spill_volume(plan))
        parts = partitions_for_envelope(basis, bounds.m_max_bytes)
        self._warn_if_buckets_will_not_fit(basis, bounds.m_max_bytes, parts)
        return parts

    @staticmethod
    def _warn_if_buckets_will_not_fit(basis: int, envelope_bytes: int, parts: int) -> None:
        """Say so when the bucket count saturates and each bucket is still over the envelope.

        `partitions_for_envelope` clamps at `MAX_SPILL_PARTITIONS`, so past
        `4,096 x envelope` of state its own promise — "each bucket fits" — quietly stops
        holding. `envelope_shortfall` has existed to report that and nothing consulted it, so
        the miss was computable and never computed. It is not a failure (the reduce splits an
        over-large bucket by grace recursion), but it *is* the reason an out-of-core query
        pays an extra write and re-read of its whole spilled state, and that is otherwise
        indistinguishable from the spill simply being large.

        One line per spilling query, at INFO for the same reason the admission verdict is:
        an operator wants to see it without opting into per-phase timing.
        """
        shortfall = envelope_shortfall(basis, envelope_bytes)
        if shortfall <= 0:
            return
        log_kv(
            get_logger("carbonite.spill"),
            logging.INFO,
            "spill buckets exceed the offered envelope",
            partitions=parts,
            envelope_bytes=envelope_bytes,
            per_bucket_bytes=envelope_bytes + shortfall,
            shortfall_bytes=shortfall,
        )

    def compression(self, plan: PhysicalPlan) -> bool | None:
        """Whether spilling `plan` should compress its buckets, from the learned peak.

        A large out-of-core state is IO-bound (disk / object store), so above
        `SPILL_COMPRESS_ABOVE` of measured peak, trading CPU for fewer bytes pays; below it
        the CPU is not worth it. Compression is lossless, so this is a pure throughput lever.

        The size rule is only half of it: whether the trade pays is also a question about the
        *device*. On local flash the codec is the bottleneck; on a network volume at a tenth
        of that bandwidth every byte not written is time not spent, and a state well under the
        size threshold is still worth compressing. The device's measured class supplies that
        half.

        Args:
            plan: The physical plan about to be spilled.

        Returns:
            The decision, or `None` for an un-sized plan (keep the configured default).
        """
        return should_compress(self.peak_bytes(plan), self._spill_device_factor())

    def _spill_device_factor(self) -> float:
        """What a byte costs on the device this query will spill to, against local flash.

        The directory comes from `site.spill_scratch_dir`, the one resolution every spill
        path shares, so the policy and the write agree about which disk is being reasoned
        about. Spelling the three steps out here instead is how the cost model came to price
        a container's overlay while the spill landed on the node's NVMe. `1.0` on anything
        unidentified, which is the size-only behaviour this had before.
        """
        from batcher._internal.hardware.storage import device_cost_factor
        from batcher._internal.site import spill_scratch_dir

        try:
            return device_cost_factor(spill_scratch_dir())
        except Exception as exc:  # pragma: no cover - a probe must never break spilling
            note_suppressed("carbonite", "read the spill device class", exc)
            return 1.0

    def soft_budget(self) -> int:
        """Bytes a query aims to stay under (the admission/throttle threshold)."""
        return int(self._envelope * self._config.memory.soft_limit * self._share)

    def hard_budget(self) -> int:
        """Bytes *this query* may hold before it must spill (the spill/reserve cap).

        Every out-of-core decision uses this one figure, derived from the once-sampled
        envelope, so they can never disagree.

        Scaled by the query's concurrency share. The **pool** stays sized to the whole
        envelope — it is the process's one real budget, and shrinking it would make a
        concurrent query's already-granted reservation retroactively unaffordable. What the
        share bounds is the amount this query *plans* to hold, which is the figure that has
        to shrink when it is one of N.
        """
        return int(self._envelope * self._config.memory.hard_limit * self._share)

    def _spill_volume(self, plan: PhysicalPlan) -> int:
        """Bytes this plan's family is predicted to actually write to disk, or 0."""
        return self._model.predicted_spill_bytes(plan.ops)

    def _bucket_target_bytes(self) -> int:
        """Bytes to aim at per spill bucket, held under the grace-recursion ceiling."""
        ceiling = int(getattr(self._config.memory, "spill_bucket_max_bytes", 0) or 0)
        if ceiling <= 0:
            return SPILL_BYTES_PER_PARTITION
        return min(SPILL_BYTES_PER_PARTITION, ceiling)
