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

from batcher.plan.expr_ir import Col
from batcher.plan.logical import Aggregate, Join, LogicalPlan, Project, Scan

__all__ = [
    "children",
    "reparent_unvalidated",
    "scanned_source_ids",
    "transform_down",
    "transform_up",
    "walk",
    "walk_with_base_names",
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


def scanned_source_ids(node: LogicalPlan) -> set[int]:
    """The `source_id`s that `node`'s subtree actually reads.

    Sources are addressed positionally across the FFI boundary, so a caller has to keep
    every source at its own index while reading only the ones the plan reaches. Both the
    single-node UDF path and the distributed stage splitter need exactly that set, and
    they live in `core` and `dist`, which cannot import each other — so it belongs here,
    in the neutral layer both already depend on, rather than being written twice.

    Args:
        node: The root of the subtree to inspect.

    Returns:
        The set of `Scan.source_id` values reachable from `node`.
    """
    if isinstance(node, Scan):
        return {node.source_id}
    ids: set[int] = set()
    for child in children(node):
        ids |= scanned_source_ids(child)
    return ids


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


def reparent_unvalidated(node: LogicalPlan, **changes: object) -> LogicalPlan:
    """`node` with `changes` applied, *without* re-running its `__post_init__` validation.

    `dataclasses.replace` is the normal way to do this and is right almost everywhere: an
    optimizer rewrite rebuilds a node onto a real input, so re-validating catches a rule that
    produced a plan referencing a column its input does not have.

    It is wrong in exactly one place — rebuilding a node onto a **stage boundary**. When the
    distributed executor splits a plan into resource stages, each later stage is re-parented
    onto a stand-in `Scan` representing the upstream stage's published morsels, and that scan
    carries an empty schema because a `MapBatches` cannot report its output *types* through an
    opaque `fn` (`available_schema()` returns `None` by design). Re-validating a `Project` or
    `Filter` against that stand-in asks a question it was never able to answer, and the node
    fails with ``references unknown column(s) ... available: []``. So any
    ``map_batches(...).select(...)`` or ``.filter(...)`` — score then narrow, the ordinary
    shape of batch inference — raised under ``distributed=True`` while working single-node.

    Skipping the check is sound, not a workaround: these nodes were validated against their
    real input when the plan was built, and every `__post_init__` on them is validation-only,
    so nothing is normalized away by not running it. Giving the boundary scan the upstream's
    column *names* was the alternative and is worse: that schema is also the one consulted when
    the upstream yields no rows, so inventing types to pair with the names is the silent
    wrong-column-type defect the device tier documents.

    Use this only where the new input is a stand-in. Everywhere else `dataclasses.replace` and
    `with_children` are correct, and their validation is worth keeping.

    Args:
        node: The node to rebuild, already validated against its original input.
        changes: Fields to replace, by name.

    Returns:
        A new node of the same type with `changes` applied and every other field copied.
    """
    clone = object.__new__(type(node))
    for field in dataclasses.fields(node):
        value = changes[field.name] if field.name in changes else getattr(node, field.name)
        object.__setattr__(clone, field.name, value)
    return clone


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
    read-only analyses (counting ops, collecting scans, validation).

    Recursive on purpose, and measured. `yield from` re-enters one generator per nesting
    level for each value it forwards, which argues for an explicit stack — but a plan is
    a handful of nodes deep, and at that size the `children()` list plus the stack
    bookkeeping costs more than the re-entry it removes (3-node plan: 1.14 us recursive
    against 1.45 us iterative; an 11-node plan, 4.24 against 4.90). Don't "fix" this
    without re-measuring on a real plan.
    """
    yield node
    for child in children(node):
        yield from walk(child)


def walk_with_base_names(node: LogicalPlan) -> list[tuple[LogicalPlan, dict[str, str]]]:
    """Every node, paired with the map from the names it references to their base columns.

    A plan's column names are not a source's column names. The SQL front-end disambiguates
    `date_dim d1, date_dim d2` by projecting every column of each scan to `d1__<name>` /
    `d2__<name>`, and a `SELECT a AS b` renames one. So an analysis that reads names *off a
    plan* and then looks them up in a **source's schema** — which is what deciding "sketch
    this column's distinct count" and "fetch this column's min/max" both do — matches
    nothing for any aliased table, and goes silently blind rather than failing.

    That is not a corner case: it cost TPC-DS q17 its whole join order.
    `d1.d_quarter_name = '2001Q1'` keeps 91 of `date_dim`'s 73,049 rows, but with no distinct
    count the estimator used the flat equality default and predicted 7,305 — so the plan put
    `store_sales ⋈ item` (a 498 MB intermediate) ahead of the 91-row filter that reduces the
    fact table 20x.

    Each pair's map is keyed by the names visible *at that node's input*, so a caller
    translates a name with `mapping.get(name, name)` — an absent name stands for itself. Only
    a chain of plain `Col` projections and join pass-throughs is followed; a column computed
    by an expression has no single base column and is dropped.

    A join's two sides are merged into one map. Their *inputs* may share a column name (only
    the join's *output* aliases are guaranteed distinct), so a collision resolves to one side
    arbitrarily. That is sound for every caller here, which unions the results and matches
    them against each source's own schema by name — a name that belongs to neither side's
    source simply matches nothing.

    Args:
        node: The plan to walk.

    Returns:
        `(node, name → base name)` for every node, children before parents.
    """
    pairs: list[tuple[LogicalPlan, dict[str, str]]] = []
    _base_names(node, pairs)
    return pairs


#: The map a leaf reports: a `Scan` reads its source's own column names, so every name it
#: sees stands for itself. Shared and never written to, like every map this walk hands up.
_NO_NAMES: dict[str, str] = {}


def _base_names(
    node: LogicalPlan, pairs: list[tuple[LogicalPlan, dict[str, str]]]
) -> dict[str, str]:
    """Record `node`'s input-name map into `pairs` and return its *output* name map."""
    kids = children(node)
    maps = [_base_names(child, pairs) for child in kids]
    # One child hands its map straight up rather than into a copy of itself, and that is
    # most of this walk's cost: a plan is mostly unary (filter, project, sort, limit), so
    # copying at every level makes the work quadratic in the depth of a chain that renames
    # nothing. Every map here is built fresh by the branches below and only ever read
    # afterwards, so sharing one is safe — nothing mutates a map it did not build.
    if len(maps) == 1:
        merged = maps[0]
    elif not maps:
        merged = _NO_NAMES
    else:
        merged = {}
        for m in maps:
            merged.update(m)
    pairs.append((node, merged))

    if isinstance(node, Project):
        return {
            item.alias: merged.get(item.expr.name, item.expr.name)
            for item in node.items
            if isinstance(item.expr, Col)
        }
    if isinstance(node, Join):
        left, right = maps[0], maps[1]
        return {
            o.alias: (left if o.side == "left" else right).get(o.name, o.name) for o in node.output
        }
    if isinstance(node, Aggregate):
        return {
            key.alias: merged.get(key.expr.name, key.expr.name)
            for key in node.group_keys
            if isinstance(key.expr, Col)
        }
    # Everything else either shapes rows without renaming columns (filter, sort, limit,
    # distinct) or combines relations that already agree on their names (union), so the
    # names its consumers see are the ones its input carried.
    return merged
