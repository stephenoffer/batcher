"""Adaptive (intra-query) execution: stage-boundary re-optimization — package façade.

A static optimizer plans the whole query once against cardinality *estimates*.
The adaptive executor instead materializes the plan one pipeline breaker at a
time and re-optimizes the remaining plan with that breaker's **exact** output
cardinality fed back as a known-size source. Downstream decisions — notably join
build-side — therefore use *measured* sizes (provenance `exact`) rather than
guesses, even when the estimate would have been badly wrong (e.g. a very
selective filter feeding a join). This is the metadata-driven moat that static
engines (DuckDB) and stage-plan-only adapters can't match.

Mechanism: find the lowest breaker whose inputs are all breaker-free, execute it
through the normal optimize→engine path, replace its subtree with a `Scan` over
an in-memory source holding the result (whose `row_count` is now exact), and
repeat. Each stage is optimized with its inputs already materialized, so a join
over two aggregates picks its build side from the two real sizes.

Split by responsibility: `gating` decides *whether* to be adaptive and whether an
estimate held, `staging` runs the stage loop and owns its resources, and
`plan_surgery` walks and rewrites the plan tree. All three are re-exported so
`from batcher.api.adaptive import X` keeps working across the split.
"""

from __future__ import annotations

from batcher.api.adaptive.gating import _estimate_accurate, resolve_adaptive
from batcher.api.adaptive.staging import AdaptiveResult, _stage_row_count, execute_adaptive

__all__ = [
    "AdaptiveResult",
    "_estimate_accurate",
    "_stage_row_count",
    "execute_adaptive",
    "resolve_adaptive",
]
