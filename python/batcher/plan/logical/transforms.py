"""Plan transforms and predicates over `LogicalPlan` trees.

`remap_sources` shifts every `Scan.source_id` (used when appending a right side's
sources after the left's); `is_streamable` reports whether a plan is
partition-independent (only row-wise operators, no pipeline breaker).
"""

from __future__ import annotations

from batcher.plan.logical.base import LogicalPlan
from batcher.plan.logical.relational import (
    Filter,
    MapBatches,
    Project,
    Sample,
    Scan,
    Unnest,
    Unpivot,
)

__all__ = ["is_streamable", "remap_sources"]


def remap_sources(plan: LogicalPlan, offset: int) -> LogicalPlan:
    """Return a copy of `plan` with every `Scan.source_id` shifted by `offset`.

    Used when joining two datasets: the right side's sources are appended after
    the left's, so its scans must point past them.

    Only `Scan` carries a `source_id`; every other node is rebuilt generically with
    its remapped children by `transform_up`, so a new node type needs no edit here.
    The import is function-local because `plan.visitor` imports this module.
    """
    from batcher.plan.visitor import transform_up

    def shift(node: LogicalPlan) -> LogicalPlan:
        if isinstance(node, Scan):
            return Scan(node.source_id + offset, node.schema)
        return node

    return transform_up(plan, shift)


def is_streamable(plan: LogicalPlan) -> bool:
    """Whether `plan` can be executed one source batch at a time in bounded memory.

    True iff the plan contains only row-wise / per-partition operators —
    `Scan`, `Filter`, `Project`, `MapBatches`, `Unnest` — and no pipeline breaker
    (aggregate, sort, join, distinct, union, window, limit) that must see the
    whole input. Such plans are partition-independent, so running them per source
    batch yields exactly the same result as running them over the whole input.
    """
    if isinstance(plan, Scan):
        return True
    if isinstance(plan, (Filter, Project, MapBatches, Unnest, Unpivot, Sample)):
        return is_streamable(plan.input)
    return False
