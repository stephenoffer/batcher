"""Shared traversal for `LogicalPlan` trees.

Every pass, analysis, and rewrite walks the same immutable node tree. Without a
shared traversal each one re-implements the per-node-type `isinstance` ladder
(see how predicate pushdown, source remapping, and cardinality estimation each
hand-roll it) — which means every new node type has to be added in N places.

This module centralizes the structural recursion so the rest of the codebase
expresses *what* to do at a node, not *how* to find its children. A new node type
is handled here once (generically, via dataclass fields), and adding the
hundredth optimization rule never means touching a tree-walk again.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from batcher.plan.logical import LogicalPlan

__all__ = [
    "children",
    "transform_down",
    "transform_up",
    "walk",
    "with_children",
]

# Per-type field-name cache. A node's dataclass fields are fixed by its *type*, but
# `dataclasses.fields(node)` rebuilds the field tuple on every call — and the optimizer
# walks the tree thousands of times per query. Caching the names once per type turns the
# hot `children`/`with_children` traversal into plain `getattr`s. Keyed by the node class,
# which is process-stable, so the cache never invalidates.
_FIELD_NAMES: dict[type, tuple[str, ...]] = {}


def _field_names(node: LogicalPlan) -> tuple[str, ...]:
    names = _FIELD_NAMES.get(type(node))
    if names is None:
        names = tuple(f.name for f in dataclasses.fields(node))
        _FIELD_NAMES[type(node)] = names
    return names


def children(node: LogicalPlan) -> list[LogicalPlan]:
    """The direct child plans of `node`, left-to-right.

    Discovered generically from the node's dataclass fields: any field that is a
    `LogicalPlan`, or a tuple containing them, contributes children. This is why
    new node types need no edit here.
    """
    out: list[LogicalPlan] = []
    for name in _field_names(node):
        value = getattr(node, name)
        if isinstance(value, LogicalPlan):
            out.append(value)
        elif isinstance(value, tuple):
            out.extend(v for v in value if isinstance(v, LogicalPlan))
    return out


def with_children(node: LogicalPlan, new_children: list[LogicalPlan]) -> LogicalPlan:
    """Rebuild `node` with its child plans replaced, in the order `children` yields.

    Non-plan fields (predicates, keys, schemas) are preserved. The number of
    `new_children` must match `len(children(node))`.

    **Structural sharing:** when every replacement child is the *same object* (`is`)
    as the original, no allocation happens — `node` itself is returned. This lets an
    unchanged subtree keep its identity through a `transform_up`, which the optimizer
    relies on for O(1) fixpoint detection (`updated is plan`) and a higher
    estimator-memo hit rate. Comparison is element-wise `is` (a rebuilt tuple is
    always a fresh object, so comparing the tuple itself would never share).
    """
    it = iter(new_children)
    changes: dict[str, object] = {}
    for name in _field_names(node):
        value = getattr(node, name)
        if isinstance(value, LogicalPlan):
            replacement = next(it)
            if replacement is not value:
                changes[name] = replacement
        elif isinstance(value, tuple) and any(isinstance(v, LogicalPlan) for v in value):
            rebuilt = tuple(next(it) if isinstance(v, LogicalPlan) else v for v in value)
            if any(a is not b for a, b in zip(rebuilt, value, strict=True)):
                changes[name] = rebuilt
    return node if not changes else dataclasses.replace(node, **changes)


def transform_up(node: LogicalPlan, fn: Callable[[LogicalPlan], LogicalPlan]) -> LogicalPlan:
    """Bottom-up rewrite: transform children first, then apply `fn` to the rebuilt
    node. The post-order shape most rewrites want (children are already final when
    a node is visited)."""
    rebuilt = with_children(node, [transform_up(c, fn) for c in children(node)])
    return fn(rebuilt)


def transform_down(node: LogicalPlan, fn: Callable[[LogicalPlan], LogicalPlan]) -> LogicalPlan:
    """Top-down rewrite: apply `fn` to `node`, then recurse into the result's
    children. Use when a rule reshapes a node before its children are visited."""
    transformed = fn(node)
    return with_children(transformed, [transform_down(c, fn) for c in children(transformed)])


def walk(node: LogicalPlan):
    """Yield every node in the tree, pre-order (parents before children). For
    read-only analyses (counting ops, collecting scans, validation)."""
    yield node
    for child in children(node):
        yield from walk(child)
