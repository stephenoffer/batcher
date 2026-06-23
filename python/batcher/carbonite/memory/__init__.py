"""Carbonite memory governance: the buffer pool, pressure sensing, estimation.

Groups the three memory concerns the resource manager composes — the
reserve-before-allocate `BufferPool`, the `PressureMonitor` that reads live RAM,
and the `OperatorMemoryEstimator` that sizes a plan's envelope from Kyber's
per-operator bounds. Re-exports only; the logic lives in the sibling modules.
"""

from __future__ import annotations

from batcher.carbonite.memory.estimator import OperatorMemoryEstimator, peak_operator_bytes
from batcher.carbonite.memory.pool import BufferPool, process_pool
from batcher.carbonite.memory.pressure import PressureLevel, PressureMonitor

__all__ = [
    "BufferPool",
    "OperatorMemoryEstimator",
    "PressureLevel",
    "PressureMonitor",
    "peak_operator_bytes",
    "process_pool",
]
