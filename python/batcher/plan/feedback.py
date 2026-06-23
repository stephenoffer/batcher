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

__all__ = ["FeedbackSink", "OperatorFeedback"]


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


@runtime_checkable
class FeedbackSink(Protocol):
    """Anything that can absorb operator feedback (the MetadataHub, a test spy)."""

    def record(self, feedback: OperatorFeedback) -> None: ...
