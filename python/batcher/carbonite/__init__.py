"""Carbonite — the resource manager. **Resources, memory, and flow control only.**

Responsibility boundary (enforced by the layer-import contract):
  * Carbonite owns the buffer pool and memory envelopes, credit-based flow
    control and backpressure, spill decisions, and the result cache. It validates
    a `PhysicalPlan` against available resources (`FeasibilityVerdict`) and hands
    Core blocking allocation primitives.
  * Carbonite does NOT choose plans or algorithms (that is Kyber) and does NOT
    run operators (that is Core). It consumes Kyber's `ResourceBounds` and exposes
    allocation policy to Core, but never imports `kyber` or `core`.

The bootstrap manager validates trivially (single-node, in-memory) and reserves
nothing; the buffer pool, envelopes, AIMD, and spill logic land behind this seam.
"""

from __future__ import annotations

from batcher.carbonite.manager import ResourceManager

__all__ = ["ResourceManager"]
