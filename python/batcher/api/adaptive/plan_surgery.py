"""Plan-tree traversal and rewriting for the adaptive loop (control plane, `api`).

The seam: this module is pure *structure*. It knows how to walk a `LogicalPlan`,
find the next pipeline breaker that is ready to run, and splice a materialized
`Scan` in place of an executed subtree — and nothing else. It never estimates a
cardinality, never decides whether to be adaptive, and never executes anything, so
the stage loop (`staging`) and the gate (`gating`) can both depend on it without
depending on each other.
"""

from __future__ import annotations

import dataclasses

from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Join,
    Limit,
    LogicalPlan,
    Sort,
    Union,
    Window,
    is_streamable,
)

__all__ = ["BREAKERS", "children", "joins", "lowest_breaker", "replace", "walk"]

BREAKERS = (Aggregate, Sort, Distinct, Window, Limit, Join, Union)


def children(node: LogicalPlan) -> list[LogicalPlan]:
    if isinstance(node, Join):
        return [node.left, node.right]
    if isinstance(node, Union):
        return list(node.inputs)
    if hasattr(node, "input"):
        return [node.input]
    return []


def walk(node: LogicalPlan):
    """Pre-order walk over the plan tree (local helper, no visitor import cycle).

    Recursive for the reason `plan.visitor.walk` documents: at real plan depths the stack
    bookkeeping an iterative walk needs costs more than the `yield from` re-entry it
    removes. Measured, not assumed.
    """
    yield node
    for child in children(node):
        yield from walk(child)


def joins(node: LogicalPlan) -> list[Join]:
    """Every `Join` node in the plan (pre-order)."""
    out: list[Join] = [node] if isinstance(node, Join) else []
    for child in children(node):
        out.extend(joins(child))
    return out


def lowest_breaker(node: LogicalPlan):
    """A breaker whose inputs are all breaker-free (so it can run now)."""
    for child in children(node):
        found = lowest_breaker(child)
        if found is not None:
            return found
    if isinstance(node, BREAKERS) and all(is_streamable(c) for c in children(node)):
        return node
    return None


def replace(node: LogicalPlan, target: LogicalPlan, repl: LogicalPlan) -> LogicalPlan:
    if node is target:
        return repl
    if isinstance(node, Join):
        return Join(
            replace(node.left, target, repl),
            replace(node.right, target, repl),
            node.left_keys,
            node.right_keys,
            node.join_type,
            node.output,
            node.strategy,
        )
    if isinstance(node, Union):
        return Union(tuple(replace(i, target, repl) for i in node.inputs), node.distinct)
    if hasattr(node, "input"):
        return dataclasses.replace(node, input=replace(node.input, target, repl))
    return node
