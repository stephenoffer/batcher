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

from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed
from batcher.carbonite.memory.pressure import PressureLevel
from batcher.carbonite.policies.spill_shape import (
    SPILL_BYTES_PER_PARTITION,
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
    ) -> None:
        self._config = config
        self._ctx = ctx
        self._estimator = estimator
        self._model = model
        self._pressure = pressure
        self._envelope = envelope_bytes
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
        return None

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
        return input_bytes + max(0, self.peak_bytes(plan)) > self.hard_budget()

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
        return partitions_for_envelope(basis, bounds.m_max_bytes)

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

        Resolved the same three ways the spill paths resolve their directory — configured
        root, measured local scratch, system tempdir — so the policy and the write agree about
        which disk is being reasoned about. `1.0` on anything unidentified, which is the
        size-only behaviour this had before.
        """
        import tempfile

        from batcher._internal.hardware.storage import device_cost_factor
        from batcher._internal.site import local_scratch_root

        try:
            target = self._config.memory.spill_dir or local_scratch_root() or tempfile.gettempdir()
            return device_cost_factor(target)
        except Exception as exc:  # pragma: no cover - a probe must never break spilling
            note_suppressed("carbonite", "read the spill device class", exc)
            return 1.0

    def soft_budget(self) -> int:
        """Bytes a query aims to stay under (the admission/throttle threshold)."""
        return int(self._envelope * self._config.memory.soft_limit)

    def hard_budget(self) -> int:
        """Bytes a query may hold before it must spill (the spill/reserve cap).

        Both the spill decision and the reservation use this one figure, derived from the
        once-sampled envelope, so the two can never disagree.
        """
        return int(self._envelope * self._config.memory.hard_limit)

    def _spill_volume(self, plan: PhysicalPlan) -> int:
        """Bytes this plan's family is predicted to actually write to disk, or 0."""
        return self._model.predicted_spill_bytes(plan.ops)

    def _bucket_target_bytes(self) -> int:
        """Bytes to aim at per spill bucket, held under the grace-recursion ceiling."""
        ceiling = int(getattr(self._config.memory, "spill_bucket_max_bytes", 0) or 0)
        if ceiling <= 0:
            return SPILL_BYTES_PER_PARTITION
        return min(SPILL_BYTES_PER_PARTITION, ceiling)
