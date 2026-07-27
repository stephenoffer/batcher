"""Carbonite — the resource manager. **Resources, memory, and flow control only.**

Responsibility boundary (enforced by the layer-import contract):
  * Carbonite owns the buffer pool and memory envelopes, credit-based flow
    control and backpressure, spill decisions and the tiered spill store, the
    result cache, the Flight shuffle transport, and fault-tolerance policy. It
    validates a `PhysicalPlan` against available resources (`FeasibilityVerdict`)
    and hands Core blocking allocation primitives.
  * Carbonite does NOT choose plans or algorithms (that is Kyber) and does NOT
    run operators (that is Core). It consumes Kyber's `ResourceBounds` and exposes
    allocation policy to Core, but never imports `kyber` or `core`.

Nothing here can change a result. Every decision — how much to reserve, when to go
out of core, how wide to shard a spill, how many credits a channel gets, how big a
morsel is — is a memory-safety or throughput lever over a relation whose value is
already fixed. That is what makes this subsystem's bugs quiet: being wrong costs
latency or the process, never an incorrect row, so no differential test can catch
one. It is also why the module docstrings here argue their reasoning at length
rather than merely describing behavior.
"""

from __future__ import annotations

from batcher.carbonite.cache import (
    CacheStore,
    current_result_cache,
    reset_result_cache,
    result_cache,
)
from batcher.carbonite.manager import ResourceManager

__all__ = [
    "CacheStore",
    "ResourceManager",
    "current_result_cache",
    "reset_result_cache",
    "result_cache",
]
