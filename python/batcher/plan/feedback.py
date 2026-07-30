"""Execution feedback contract: Core → Kyber.

After each operator runs, Core reports what actually happened. Kyber's learned
cardinality/cost correction consumes this (via the MetadataHub) to improve future
plans. Writes are non-blocking and must never raise into the hot path — a
`FeedbackSink` that fails logs and drops.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Protocol, runtime_checkable

from batcher._internal.hardware import fingerprint
from batcher.plan.ids import OpId

__all__ = [
    "FeedbackSink",
    "OperatorFeedback",
    "cpu_utilization",
    "oversubscribed",
    "preemption_rate",
]

# Involuntary context switches per core-second above which the operator is judged to have been
# competing for its cores rather than owning them. A CPU-bound thread on an uncontended box is
# preempted only by timer and kernel-thread activity, which lands in the low tens per second;
# once another runnable thread wants the core, the scheduler swaps at its timeslice granularity
# and the rate rises by an order of magnitude. The threshold sits between those regimes rather
# than at either edge, so a busy-but-uncontended run is not misread as contended.
CONTENDED_PREEMPTIONS_PER_CORE_SECOND = 200.0

# Major page faults per core-second above which the box is judged to be paging against the
# query rather than warming up. A major fault is a disk read, so a handful over a whole
# operator is the ordinary first-touch of a mapped file and says nothing; a sustained rate is
# the working set not fitting, and the fix for that is fewer concurrent tasks per node, never
# more. The threshold is deliberately low in absolute terms — even a few disk faults per
# core-second is milliseconds of stall per second — but non-zero, so first-touch never trips it.
PAGING_FAULTS_PER_CORE_SECOND = 5.0


def cpu_utilization(
    cpu_ns: float, elapsed_ns: float, threads: int, wall_span_ns: float = 0.0
) -> float:
    """Mean fraction of allocated cores kept busy, clamped to [0, 1].

    `cpu_ns` is CPU-time summed across all worker threads during the operator; dividing by
    ``interval x threads`` (the engine's *actual* live thread count, not a guessed host core
    count — which is wrong under a cgroup CPU quota) gives the per-core busy fraction. 0.0 when
    the engine reported no CPU time (older build), no interval, or no thread count.

    The interval is `wall_span_ns` when the engine reported one and `elapsed_ns` otherwise.
    The two tiers differ here and getting it wrong is not a small error: on a materializing
    executor an operator runs alone so `elapsed_ns` *is* its wall interval, but on the
    streaming executor operators interleave and `elapsed_ns` is transform time summed over
    every morsel and worker. Dividing that summed work by itself yields exactly ``1 / threads``
    for every operator of every query.

    Args:
        cpu_ns: CPU time summed across the operator's worker threads.
        elapsed_ns: The operator's own elapsed time.
        threads: Worker threads the operator ran across.
        wall_span_ns: The wall interval the operator occupied, when the engine tracks one.

    Returns:
        The per-core busy fraction in [0, 1], or `0.0` when unmeasured.
    """
    interval = wall_span_ns if wall_span_ns > 0 else elapsed_ns
    if cpu_ns <= 0 or interval <= 0 or threads <= 0:
        return 0.0
    return min(1.0, cpu_ns / (interval * threads))


def preemption_rate(invol_ctx_switches: int, elapsed_ms: float, threads: int) -> float:
    """Involuntary context switches per core-second — how hard the operator fought for cores.

    An involuntary switch is the scheduler taking a CPU away, which happens only when another
    runnable thread wants it. That makes this the one per-operator signal that separates the
    two indistinguishable causes of poor scaling: an operator that failed to parallelize, and
    an operator whose threads were repeatedly evicted by something else on the box. Utilization
    reports both as low, and they have opposite fixes — widen the fan-out, or narrow it.

    Normalized per core-second rather than per operator so the figure is comparable across
    operators of wildly different durations and widths.

    Args:
        invol_ctx_switches: Involuntary switches measured during the operator.
        elapsed_ms: The operator's wall-clock duration in milliseconds.
        threads: Worker threads the operator ran across.

    Returns:
        Switches per core-second, or `0.0` when nothing was measured.
    """
    core_seconds = (elapsed_ms / 1000.0) * max(0, threads)
    if invol_ctx_switches <= 0 or core_seconds <= 0:
        return 0.0
    return invol_ctx_switches / core_seconds


@dataclass(frozen=True, slots=True)
class OperatorFeedback:
    """Observed outcome of executing one physical operator."""

    op_id: OpId
    kind: str
    n_actual: int  # actual output rows
    t_op_ms: float  # wall-clock time
    m_peak_bytes: int  # observed peak working set (materialized input + result)
    selectivity: float  # n_out / n_in  (1.0 when not applicable)
    batch_size: int  # morsel size used
    backend: str = "interp"  # execution tier/backend that ran it
    algorithm: str = ""  # chosen algorithm arm, if any
    # Mean fraction of allocated cores the operator kept busy (CPU-time / (wall x
    # threads)), in [0, 1]. The CPU analog of GPU utilization: a CPU-bound op nears
    # 1.0, an IO-bound one stays low. 0.0 means unmeasured (an older engine that
    # reports no `cpu_ns`), which the adaptive CPU-share loop treats as "no signal".
    cpu_utilization: float = 0.0
    # Worker threads the operator actually ran across (rayon's live count; 1 for the
    # sequential oracle). `cpu_utilization` folds this into a per-core fraction, but the raw
    # count is what a per-task `num_cpus` sizing loop needs to tell a fully-busy 8-core
    # breaker (needs ~8 cores) from a fully-busy 1-core one — the fraction alone caps at 1.
    # 0 means unmeasured (an older engine that reports no thread count).
    threads: int = 0
    # Actual *input* rows the operator consumed. Cost calibration fits the per-row
    # coefficients of the input-bound families (filter, distinct, aggregate,
    # hash_join) against this, not `n_actual` (output rows) — a selective filter's
    # cost scales with what it read, not what it kept. 0 means unmeasured (an older
    # engine, or a source op with no input), in which case calibration reconstructs
    # it from `n_actual / selectivity`.
    n_input: int = 0
    # Rows fed into a join's *build* side (the hashed input). 0 for every other operator.
    # A join's memory scales with the side it hashes, not with the side it probes, so the
    # learned memory model divides `m_peak_bytes` by this rather than by `n_input`.
    n_build: int = 0
    # Bytes of this operator's *result* alone. `m_peak_bytes` is now the true peak working
    # set (a breaker's materialized input plus its result); this is what that field used to
    # hold, kept so the profiler can still report output size honestly.
    result_bytes: int = 0
    # Logical bytes this operator routed to its out-of-core spill path (0 when it ran in
    # memory, or when the spill volume was not measured). The magnitude `algorithm="spill"`
    # cannot carry: Carbonite sizes spill scratch, disk bandwidth, and partition counts from
    # a measured 1 GB-vs-100 GB spill, and Kyber can cost the grace path from real volume.
    spill_bytes: int = 0
    # Measured growth in the process's peak resident set (bytes) during this operator, from
    # `getrusage(ru_maxrss)`. The ground-truth memory high-water that `m_peak_bytes` (an
    # Arrow-size estimate) cannot see: transient scratch, allocator fragmentation, off-pool
    # buffers. 0 means the op set no new high-water or the platform can't report RSS.
    peak_rss_bytes: int = 0
    # The operator's structural plan signature — a stable identity across executions
    # (`op_id` is only a position in one plan's walk). Empty when the reporter has no
    # plan to correlate against: a distributed worker runs a sub-plan whose `op_id`s
    # live in their own space, so it reports no signature and contributes only to the
    # per-`kind` calibration.
    signature: str = ""
    # The rows Kyber estimated for this operator **before** applying any learned
    # correction. Paired with `n_actual`, this is the raw q-error `n_actual /
    # n_estimated` — the measure of the *structural* estimator's error. Reporting the
    # already-corrected estimate instead would make a converged correction look
    # error-free and decay it back to 1.0. 0.0 means unestimated.
    n_estimated: float = 0.0
    # Per-row cost of the expressions this operator evaluated, relative to a plain
    # comparison (1.0 when it evaluates none). Cost calibration divides it out of
    # `t_op_ms`, so the fitted per-row coefficients describe the *engine* rather than
    # whichever expressions the workload happened to contain. Paired with `backend`
    # (the tier that ran it), it is also what makes the JIT speedup measurable.
    expr_factor: float = 1.0
    # --- Measured hardware consumption. Every field is 0 when unmeasured, never "none". ---
    # Page faults served without disk I/O: the operator committing new memory. Times the page
    # size (`faulted_bytes`) this is the *measured* working set, against which `m_peak_bytes`
    # is only an Arrow-size model — the two diverge on fragmentation and off-pool scratch.
    minor_faults: int = 0
    # Page faults that required disk I/O. Any material count means the box was paging against
    # the query: memory the operator believed it held was being fetched back from storage.
    # This is invisible in every other field, and it is the one condition under which adding
    # parallelism strictly makes things worse.
    major_faults: int = 0
    # Times the operator blocked and yielded the CPU. High against low `cpu_utilization` marks
    # a genuinely I/O- or lock-bound operator, as opposed to one that simply failed to fan out.
    vol_ctx_switches: int = 0
    # Times the scheduler evicted the operator from a CPU. The direct measurement of core
    # contention — see `preemption_rate` for the normalized form the sizing loops read.
    invol_ctx_switches: int = 0
    # Bytes actually fetched from a block device, page-cache hits excluded. A warm and a cold
    # scan of the same file differ by two orders of magnitude and are otherwise identical in
    # every field here, so an I/O cost coefficient calibrated without this is fitted across two
    # populations at once and describes neither.
    io_read_bytes: int = 0
    # Bytes actually written to a block device, spill included. The measured counterpart to
    # `spill_bytes`, which is the volume the operator *decided* to route to disk.
    io_write_bytes: int = 0
    # The fingerprint of the machine that measured this row (`_internal.hardware.fingerprint`).
    # Every field above measured in machine units — times, bytes, faults, switches — is a
    # statement about *this hardware*, and none of them transfers to a different one. Without
    # this, a metadata store shared across a heterogeneous cluster, an autoscaling group that
    # mixes instance generations, or a laptop and CI, fits one model across several machines
    # and gets a model that is wrong for all of them, silently.
    #
    # Defaults to *this* machine, because a feedback row is constructed where the measurement
    # was taken: a caller who does not say which machine measured it is, by construction,
    # saying "here". The one case that is not true is a distributed worker's row, which the
    # driver records on the worker's behalf — so `dist` stamps the worker's own fingerprint
    # into the metrics document before it travels, and the transcription reads that.
    #
    # A stored row can still carry `""`: one written before this field existed. That is
    # "measured on an unknown machine", which is not evidence about this one, so a
    # hardware-scoped consumer excludes it rather than adopting it.
    hw_fingerprint: str = field(default_factory=fingerprint)


@runtime_checkable
class FeedbackSink(Protocol):
    """Anything that can absorb operator feedback (the MetadataHub, a test spy)."""

    def record(self, feedback: OperatorFeedback) -> None: ...


def oversubscribed(rows: Iterable[Mapping[str, Any]]) -> bool:
    """Whether a family's measured history says the *box* was oversubscribed, not the family idle.

    This is the disambiguator the CPU-share loops need and did not have. They size a per-task
    `num_cpus` from measured `cpu_utilization`, shrinking the reservation when a family's cores
    sat idle — which is right for an IO- or GPU-bound family and exactly backwards for a
    contended one. Both read as low utilization, and they have opposite fixes.

    Getting it backwards is not merely a missed improvement, it is a loop that feeds itself:
    contention lowers utilization, the lower utilization shrinks the reservation, the smaller
    reservation lets the scheduler pack more tasks onto the same cores, and that raises
    contention again. Nothing in the measurement can break the cycle, because every step of it
    looks like a family that needed fewer cores.

    Two independent measurements say "the box, not the family", and either is sufficient:

    * **Preemption.** An involuntary context switch is the scheduler taking a CPU away, which
      happens only when another runnable thread wants it. The *median* rate is used rather than
      the mean, so one contended run inside an otherwise clear history does not latch the
      family — and the maximum is not used for the same reason.
    * **Major faults.** A page fault served from disk means memory the operator believed it
      held was being fetched back from storage. Packing more tasks per node brings more memory
      with them, so this is the one condition under which the shrink is unambiguously wrong. A
      single such fault in a whole history is noise (a first-touch of a mapped file), so this
      asks for a material rate rather than a nonzero count.

    Args:
        rows: Stored feedback rows for one operator family.

    Returns:
        `True` when the family's low utilization is explained by contention for the machine.
        `False` when nothing was measured, which keeps every caller's prior behavior.
    """
    rates: list[float] = []
    faulting_core_seconds = 0.0
    faults = 0.0
    for row in rows:
        elapsed_ms = float(row.get("t_op_ms", 0.0) or 0.0)
        threads = int(row.get("threads", 0) or 0)
        core_seconds = (elapsed_ms / 1000.0) * max(0, threads)
        if core_seconds <= 0.0:
            continue  # an older engine that reported no thread count: no evidence either way
        rates.append(
            preemption_rate(int(row.get("invol_ctx_switches", 0) or 0), elapsed_ms, threads)
        )
        faults += float(row.get("major_faults", 0) or 0)
        faulting_core_seconds += core_seconds
    if not rates:
        return False
    if median(rates) > CONTENDED_PREEMPTIONS_PER_CORE_SECOND:
        return True
    return (
        faulting_core_seconds > 0.0
        and faults / faulting_core_seconds > PAGING_FAULTS_PER_CORE_SECOND
    )
