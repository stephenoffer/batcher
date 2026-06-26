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
    m_peak_bytes: int  # observed peak memory
    selectivity: float  # n_out / n_in  (1.0 when not applicable)
    batch_size: int  # morsel size used
    backend: str = "interp"  # execution tier/backend that ran it
    algorithm: str = ""  # chosen algorithm arm, if any
    # Mean fraction of allocated cores the operator kept busy (CPU-time / (wall x
    # threads)), in [0, 1]. The CPU analog of GPU utilization: a CPU-bound op nears
    # 1.0, an IO-bound one stays low. 0.0 means unmeasured (an older engine that
    # reports no `cpu_ns`), which the adaptive CPU-share loop treats as "no signal".
    cpu_utilization: float = 0.0


@runtime_checkable
class FeedbackSink(Protocol):
    """Anything that can absorb operator feedback (the MetadataHub, a test spy)."""

    def record(self, feedback: OperatorFeedback) -> None: ...
