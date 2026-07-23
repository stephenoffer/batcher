"""The Kyber optimizer entry point.

Kyber turns a `LogicalPlan` into a `PhysicalPlan` by running its rules **phase by
phase** (`rule.Phase`). Each phase holds a set of `Rule`s; rewrite phases iterate
to a fixpoint (confluent rules), the cost-based/physical phases run once. Adding an
optimization means registering a `Rule` (drop a decorated function, or
`registry.add(...)`) — never editing this driver.

The driver stays fast as the rule set grows because it **pattern-indexes**: before
running a phase it computes the set of node types present in the plan and skips
every rule whose `matches` set is disjoint from it. So a plan with no `Join` never
pays for the hundred join rules. On top of that it fuses node rules into one plan
traversal, fuses leaf expression rules into one *expression* traversal, and memoizes
the rules that proved themselves no-ops — the three devices that keep three hundred
rules affordable to plan with. See `driver`.

Cardinality and cost estimates feeding the cost-based phases sharpen across
executions via the MetadataHub (learned selectivities / join sizes), so the plan a
query gets *improves the more it runs* — Core collects the metadata, Kyber decides
with it.

Split into a package on its natural seam: the rule-application engine (`driver`) and
the `Optimizer` façade (`facade`). The import path is unchanged.
"""

from __future__ import annotations

from batcher.kyber.optimizer.facade import (
    Optimizer,
    optimize,
    optimize_full,
    optimize_logical,
    optimize_traced,
)

__all__ = [
    "Optimizer",
    "optimize",
    "optimize_full",
    "optimize_logical",
    "optimize_traced",
]
