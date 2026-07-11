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

# Per-type child-field cache. A node's dataclass fields are fixed by its *type*, and only a
# couple ever hold child plans (a `Filter` has one `input`; the rest are predicates, keys,
# schemas). `dataclasses.fields(node)` rebuilds the whole field tuple on every call, and the
# generic walk then does a `getattr` + `isinstance` on *every* field — yet the optimizer
# walks the tree thousands of times per query. So classify each type once, on its first node,
# keeping only the child-bearing fields *and how each carries children*; the hot traversal
# then touches nothing but those and dispatches with no per-call `isinstance` ladder. Keyed
# by the node class, which is process-stable, so the cache never invalidates.
#
# Each kept field carries a `_Kind`. The classification is by the field's shape, which is
# fixed by its dataclass type (a `tuple[...]` field is always a tuple; a `LogicalPlan` field
# is never one), so it is stable across instances — *except* a field first seen as `None`,
# whose type could be `LogicalPlan | None` or `tuple[...] | None`; that stays `AMBIGUOUS` and
# is re-examined by value on every call, exactly the old conservative behavior. The plan
# nodes today have no optional/None child field, so `AMBIGUOUS` never occurs and every kept
# field dispatches straight to its branch — the guard costs nothing now and stays correct if
# an optional child is ever added.
_KIND_PLAN = 0  # a single `LogicalPlan` slot
_KIND_TUPLE = 1  # a tuple that holds (some) plans, positionally
_KIND_AMBIGUOUS = 2  # first seen `None` — could hold a plan later; dispatch by value

_ChildSpec = tuple[tuple[str, int], ...]
_CHILD_SPEC: dict[type, _ChildSpec] = {}


def _classify(value: object) -> int | None:
    """The `_Kind` of a field holding `value`, or `None` if it can never hold a child."""
    if isinstance(value, LogicalPlan):
        return _KIND_PLAN
    if isinstance(value, tuple):
        # Empty → ambiguous shape but definitely a *tuple* field, so still dispatch as a
        # tuple (it may fill with plans later). Non-empty and plan-free → a scalar tuple
        # (keys, schema) that never holds a child.
        if not value:
            return _KIND_TUPLE
        return _KIND_TUPLE if any(isinstance(v, LogicalPlan) for v in value) else None
    return None if value is not None else _KIND_AMBIGUOUS


def _child_spec(node: LogicalPlan) -> _ChildSpec:
    """`((field_name, kind), ...)` for `node`'s child-bearing fields, cached per type."""
    spec = _CHILD_SPEC.get(type(node))
    if spec is None:
        pairs = []
        for f in dataclasses.fields(node):
            kind = _classify(getattr(node, f.name))
            if kind is not None:
                pairs.append((f.name, kind))
        spec = tuple(pairs)
        _CHILD_SPEC[type(node)] = spec
    return spec


def children(node: LogicalPlan) -> list[LogicalPlan]:
    """The direct child plans of `node`, left-to-right.

    Discovered generically from the node's dataclass fields: any field that is a
    `LogicalPlan`, or a tuple containing them, contributes children. This is why
    new node types need no edit here.
    """
    out: list[LogicalPlan] = []
    for name, kind in _child_spec(node):
        value = getattr(node, name)
        if kind == _KIND_PLAN:
            out.append(value)
        elif kind == _KIND_TUPLE:
            out.extend(v for v in value if isinstance(v, LogicalPlan))
        elif isinstance(value, LogicalPlan):  # AMBIGUOUS: dispatch by value
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
    for name, kind in _child_spec(node):
        value = getattr(node, name)
        if kind == _KIND_PLAN or (kind == _KIND_AMBIGUOUS and isinstance(value, LogicalPlan)):
            replacement = next(it)
            if replacement is not value:
                changes[name] = replacement
        elif kind == _KIND_TUPLE or (kind == _KIND_AMBIGUOUS and isinstance(value, tuple)):
            rebuilt = tuple(next(it) if isinstance(v, LogicalPlan) else v for v in value)
            if any(a is not b for a, b in zip(rebuilt, value, strict=True)):
                changes[name] = rebuilt
        # else: an ambiguous slot that is currently `None` consumed no child, so the
        # `new_children` cursor is not advanced and the field is left untouched.
    return node if not changes else dataclasses.replace(node, **changes)


def transform_up(node: LogicalPlan, fn: Callable[[LogicalPlan], LogicalPlan]) -> LogicalPlan:
    """Bottom-up rewrite: transform children first, then apply `fn` to the rebuilt
    node. The post-order shape most rewrites want (children are already final when
    a node is visited).

    Fused into a *single* pass over the child fields — recurse into each child and
    record only the replacements that actually differ — instead of building a child
    list (`children`) and mapping it back (`with_children`), which scanned the fields
    twice and allocated an intermediate list per node. Semantics and structural sharing
    are identical (an unchanged subtree returns the same object).
    """
    changes: dict[str, object] | None = None
    for name, kind in _child_spec(node):
        value = getattr(node, name)
        if kind == _KIND_PLAN or (kind == _KIND_AMBIGUOUS and isinstance(value, LogicalPlan)):
            new = transform_up(value, fn)
            if new is not value:
                if changes is None:
                    changes = {}
                changes[name] = new
        elif kind == _KIND_TUPLE or (kind == _KIND_AMBIGUOUS and isinstance(value, tuple)):
            rebuilt: list[object] | None = None
            for idx, v in enumerate(value):
                if isinstance(v, LogicalPlan):
                    nv = transform_up(v, fn)
                    if nv is not v:
                        if rebuilt is None:
                            rebuilt = list(value)
                        rebuilt[idx] = nv
            if rebuilt is not None:
                if changes is None:
                    changes = {}
                changes[name] = tuple(rebuilt)
        # else: an ambiguous slot currently `None` — no child to recurse into.
    rebuilt_node = node if changes is None else dataclasses.replace(node, **changes)
    return fn(rebuilt_node)


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
