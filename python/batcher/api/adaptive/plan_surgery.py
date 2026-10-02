"""Plan-tree traversal and rewriting for the adaptive loop (control plane, `api`).

The seam: this module is pure *structure*. It knows how to walk a `LogicalPlan`,
find the next pipeline breaker that is ready to run, and splice a materialized
`Scan` in place of an executed subtree — and nothing else. It never estimates a
cardinality, never decides whether to be adaptive, and never executes anything, so
the stage loop (`staging`) and the gate (`gating`) can both depend on it without
depending on each other.
"""

from __future__ import annotations

from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Join,
    Limit,
    LogicalPlan,
    RangeJoin,
    RowId,
    Sort,
    Union,
    Window,
    is_streamable,
)
from batcher.plan.visitor import children, walk, with_children

__all__ = [
    "BREAKERS",
    "STRUCTURAL_BREAKERS",
    "children",
    "joins",
    "lowest_breaker",
    "replace",
    "walk",
]

BREAKERS = (Aggregate, Sort, Distinct, Window, Limit, Join, Union)

#: The cut points when distribution *forces* staging. A row index is not a pipeline breaker,
#: but it numbers the whole input in one global order, and the distributed dispatcher can
#: compute that only as the top of a stage (a source-ordered map, numbered on the driver). So
#: a breaker above one, such as the aggregate `group_by(maintain_order=True)` lowers to, has
#: no one-shot path, and cutting at the row index is what lets it run at all. Only the
#: structural path uses this; single-node staging keeps `BREAKERS`, so its gate and cut
#: count are unchanged.
#:
#: A range join is the same case. Its one distributed form, a broadcast probe, applies only
#: at the top of a stage, and beneath another join it has none: TPC-H q22's
#: `c_acctbal > (SELECT avg(c_acctbal) ...)` under its `NOT EXISTS` raised "did not stage"
#: with `distributed=True`. Cut there, the range join is its own stage and the anti join and
#: aggregate above it read its result.
STRUCTURAL_BREAKERS = (*BREAKERS, RowId, RangeJoin)


def joins(node: LogicalPlan) -> list[Join]:
    """Every `Join` node in the plan (pre-order)."""
    out: list[Join] = [node] if isinstance(node, Join) else []
    for child in children(node):
        out.extend(joins(child))
    return out


def lowest_breaker(node: LogicalPlan, accept=None, breakers: tuple[type, ...] = BREAKERS):
    """A breaker whose inputs are all breaker-free (so it can run now).

    `accept`, when given, filters *which* runnable breaker qualifies. A rejected
    breaker is not skipped over in the plan; it stays where it is and gets executed
    inside whatever larger subplan is staged above it, which is the point. Staging a
    breaker costs a materialization and buys a measurement, so a breaker whose size is
    already known exactly is worth executing inline, fused with its neighbours, rather
    than on its own. `breakers` names the node types that count as a cut point.
    """
    for child in children(node):
        found = lowest_breaker(child, accept, breakers)
        if found is not None:
            return found
    runnable = isinstance(node, breakers) and all(is_streamable(c) for c in children(node))
    if runnable and (accept is None or accept(node)):
        return node
    return None


def replace(node: LogicalPlan, target: LogicalPlan, repl: LogicalPlan) -> LogicalPlan:
    """`node` with the subtree `target` (by identity) swapped for `repl`."""
    if node is target:
        return repl
    return with_children(node, [replace(c, target, repl) for c in children(node)])
