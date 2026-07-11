"""Execution feedback contract: Core → Kyber.

After each operator runs, Core reports what actually happened. Kyber's learned
cardinality/cost correction consumes this (via the MetadataHub) to improve future
plans. Writes are non-blocking and must never raise into the hot path — a
`FeedbackSink` that fails logs and drops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from batcher.plan.ids import OpId

__all__ = ["FeedbackSink", "OperatorFeedback", "cpu_utilization"]


def cpu_utilization(cpu_ns: float, elapsed_ns: float, threads: int) -> float:
    """Mean fraction of allocated cores kept busy, clamped to [0, 1].

    `cpu_ns` is CPU-time summed across all worker threads during the operator;
    dividing by ``elapsed_ns x threads`` (the engine's *actual* live thread count,
    not a guessed host core count — which is wrong under a cgroup CPU quota) gives
    the per-core busy fraction. 0.0 when the engine reported no CPU time (older
    build), no wall time, or no thread count.
    """
    if cpu_ns <= 0 or elapsed_ns <= 0 or threads <= 0:
        return 0.0
    return min(1.0, cpu_ns / (elapsed_ns * threads))


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


@runtime_checkable
class FeedbackSink(Protocol):
    """Anything that can absorb operator feedback (the MetadataHub, a test spy)."""

    def record(self, feedback: OperatorFeedback) -> None: ...
