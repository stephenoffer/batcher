"""Set-operation rewrites — UNION / DISTINCT structural simplifications.

Small, node-local, unconditionally semantics-preserving rewrites over `Union` and
`Distinct`. Each is registered with `@rule`; the driver supplies bottom-up traversal
and fixpoint iteration and pattern-indexes on the declared `matches`.

The one hazard these rules navigate is **bag vs set semantics**: `UNION ALL`
concatenates multisets (duplicates are significant) while `UNION`/`Distinct` collapse
to a set. A rewrite that is valid for a distinct union (where an outer dedup dominates)
is frequently *invalid* for `UNION ALL` — so every rule that could drop or reorder
duplicate-bearing rows is gated on the `distinct` flag. NULLs need no special care
here: these rewrites never compare values, they only move/merge whole rows, and every
node preserves each row (and its nulls) verbatim.

SQL `INTERSECT` / `EXCEPT` are **pre-lowered** by the front end (`Dataset.intersect` /
`Dataset.except_` in `api`) into a tagged `Union` + group-by-`bool_or` + `Filter`
shape — there is no `Intersect`/`Except` logical node — so the classic empty-operand /
identical-operand set-op rewrites do not apply to a dedicated node here and are omitted.
"""

from __future__ import annotations

import json

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.plan.logical import (
    Distinct,
    Filter,
    Limit,
    LogicalPlan,
    Project,
    Sort,
    Union,
)

__all__ = [
    "dedup_distinct_union_branches",
    "drop_distinct_in_distinct_union",
    "eliminate_sort_before_distinct",
    "eliminate_sort_in_distinct_union_branch",
    "flatten_nested_union",
    "fold_distinct_union_all",
    "prune_distinct_of_empty",
    "prune_empty_union_branch",
    "push_filter_through_distinct",
    "push_project_through_union",
    "simplify_singleton_union",
]


def _ir_key(node: LogicalPlan) -> str:
    """A hashable structural identity for a plan node (its IR rendered canonically)."""
    return json.dumps(node.to_ir(), sort_keys=True)


def _is_empty(node: LogicalPlan) -> bool:
    """Whether `node` structurally produces zero rows — a `Limit` capped at 0.

    Deliberately narrow: only a syntactically zero-row cap is treated as empty (never
    an estimate), so a branch is dropped only when it *provably* contributes nothing.
    """
    return isinstance(node, Limit) and node.n == 0


def _flatten_branches(branch: LogicalPlan, outer_distinct: bool) -> list[LogicalPlan]:
    """Recursively splice a mergeable child `Union` into the parent's branch list.

    A child union merges into its parent iff the parent's rows form a superset-safe
    concatenation of the child's rows. That holds when the parent is a `UNION ALL` and
    the child is *also* `UNION ALL` (associativity of multiset concatenation), OR when
    the parent is a distinct union (its final dedup dominates *any* nested union,
    distinct or not, all the way down). Anything else stops the recursion and stays a
    branch — merging a distinct child into a `UNION ALL` parent would silently drop the
    duplicates the parent must keep.
    """
    if isinstance(branch, Union) and (outer_distinct or not branch.distinct):
        merged: list[LogicalPlan] = []
        for child in branch.inputs:
            merged.extend(_flatten_branches(child, outer_distinct))
        return merged
    return [branch]


@rule(name="flatten_nested_union", phase=Phase.REWRITE, matches=(Union,))
def flatten_nested_union(node: Union, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Flatten a nested same-kind union into one wide union.

    `Union(Union(a, b), c)` → `Union(a, b, c)`. A `UNION ALL` parent absorbs only
    `UNION ALL` children (multiset concatenation is associative); a distinct-union
    parent absorbs *any* nested union — its final dedup dominates every inner union,
    so `Distinct-Union(UnionAll(a, b), c)` = `Distinct-Union(a, b, c)`. Merging a
    distinct child into a `UNION ALL` parent is refused (it would drop duplicates the
    parent must keep). The flatten is exhaustive in one shot, so re-applying is a no-op.
    """
    new_inputs: list[LogicalPlan] = []
    for inp in node.inputs:
        new_inputs.extend(_flatten_branches(inp, node.distinct))
    if len(new_inputs) == len(node.inputs) and all(
        a is b for a, b in zip(new_inputs, node.inputs, strict=True)
    ):
        return None
    return Union(tuple(new_inputs), node.distinct)


@rule(name="simplify_singleton_union", phase=Phase.REWRITE, matches=(Union,))
def simplify_singleton_union(node: Union, _ctx: OptimizerContext) -> LogicalPlan | None:
    """A union of exactly one branch is that branch (a `Distinct` if it was distinct).

    `Union((a,), all)` → `a`; `Union((a,), distinct)` → `Distinct(a)` (the union of one
    relation with itself-nothing is the relation, and a distinct union still dedups).
    """
    if len(node.inputs) != 1:
        return None
    only = node.inputs[0]
    return Distinct(only) if node.distinct else only


@rule(name="prune_empty_union_branch", phase=Phase.REWRITE, matches=(Union,))
def prune_empty_union_branch(node: Union, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop provably-empty branches (`Limit(x, 0)`) from a union.

    An empty branch contributes no rows to either a `UNION ALL` or a distinct union, so
    removing it changes neither the multiset nor the set. If one branch remains it
    replaces the union (wrapped in `Distinct` when the union was distinct); if every
    branch is empty, one empty branch is kept so the result stays the empty relation.
    """
    kept = [b for b in node.inputs if not _is_empty(b)]
    if len(kept) == len(node.inputs):
        return None
    if not kept:
        kept = [node.inputs[0]]
    if len(kept) == 1:
        return Distinct(kept[0]) if node.distinct else kept[0]
    return Union(tuple(kept), node.distinct)


@rule(name="dedup_distinct_union_branches", phase=Phase.REWRITE, matches=(Union,))
def dedup_distinct_union_branches(node: Union, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop structurally-identical branches of a DISTINCT union.

    `a UNION b UNION a` → `a UNION b`: the outer dedup makes a repeated branch redundant
    (`distinct(concat(a, b, a))` = `distinct(concat(a, b))`). Restricted to distinct
    unions — for `UNION ALL` a repeated branch genuinely doubles those rows (bag
    semantics) and must be kept. Identity is structural (canonical IR), so only provably
    identical branches are collapsed.
    """
    if not node.distinct:
        return None
    seen: set[str] = set()
    kept: list[LogicalPlan] = []
    for branch in node.inputs:
        key = _ir_key(branch)
        if key in seen:
            continue
        seen.add(key)
        kept.append(branch)
    if len(kept) == len(node.inputs):
        return None
    if len(kept) == 1:
        return Distinct(kept[0])
    return Union(tuple(kept), True)


@rule(name="drop_distinct_in_distinct_union", phase=Phase.REWRITE, matches=(Union,))
def drop_distinct_in_distinct_union(node: Union, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Strip a redundant per-branch `Distinct` under a DISTINCT union.

    `Distinct-Union(Distinct(x), y)` → `Distinct-Union(x, y)`: the union's own dedup
    re-deduplicates everything, so a branch-level `Distinct` is pure overhead
    (`distinct(concat(distinct(x), y))` = `distinct(concat(x, y))`). Only for distinct
    unions; stacked branch `Distinct`s are all removed at once so re-applying is a no-op.
    """
    if not node.distinct:
        return None
    new_inputs: list[LogicalPlan] = []
    changed = False
    for branch in node.inputs:
        stripped = branch
        while isinstance(stripped, Distinct):
            stripped = stripped.input
        if stripped is not branch:
            changed = True
        new_inputs.append(stripped)
    if not changed:
        return None
    return Union(tuple(new_inputs), True)


@rule(name="eliminate_sort_in_distinct_union_branch", phase=Phase.REWRITE, matches=(Union,))
def eliminate_sort_in_distinct_union_branch(
    node: Union, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """Drop an order-only `Sort` inside a branch of a DISTINCT union.

    A distinct union produces a set, so its rows carry no meaningful order — a `Sort`
    (without a top-N `limit`) feeding a branch is wasted work whose ordering the dedup
    discards. Skipped for `UNION ALL` (whose branch order is observable) and for a sort
    carrying a `limit` (a top-N selects *which* rows). Stacked sorts collapse at once.
    """
    if not node.distinct:
        return None
    new_inputs: list[LogicalPlan] = []
    changed = False
    for branch in node.inputs:
        stripped = branch
        while isinstance(stripped, Sort) and stripped.limit is None:
            stripped = stripped.input
        if stripped is not branch:
            changed = True
        new_inputs.append(stripped)
    if not changed:
        return None
    return Union(tuple(new_inputs), True)


@rule(name="push_project_through_union", phase=Phase.PUSHDOWN, matches=(Project,))
def push_project_through_union(node: Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Project(UNION ALL(a, b), items)` → `UNION ALL(Project(a), Project(b))`.

    Projection distributes over multiset concatenation, so evaluating it per branch
    lets column pruning and further pushdown continue independently into each branch.
    Restricted to `UNION ALL`: pushing a projection below a *distinct* union would run
    the union's dedup on the projected (fewer/derived) columns instead of the original
    rows, which can change the result. Union branches share an identical schema, so the
    same `items` apply to each unchanged.
    """
    inner = node.input
    if not isinstance(inner, Union) or inner.distinct:
        return None
    pushed = tuple(Project(branch, node.items) for branch in inner.inputs)
    return Union(pushed, distinct=False)


@rule(name="fold_distinct_union_all", phase=Phase.REWRITE, matches=(Distinct,))
def fold_distinct_union_all(node: Distinct, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Distinct(UNION ALL(...))` → `UNION(..., distinct=True)`.

    Deduplicating a `UNION ALL` is exactly a distinct union, so the two operators fold
    into one node that dedups as it concatenates. Complements `remove_redundant_distinct`
    (which drops a `Distinct` over an *already*-distinct union); this handles the
    `UNION ALL` case that rule leaves alone.
    """
    inner = node.input
    if isinstance(inner, Union) and not inner.distinct:
        return Union(inner.inputs, distinct=True)
    return None


@rule(name="push_filter_through_distinct", phase=Phase.PUSHDOWN, matches=(Filter,))
def push_filter_through_distinct(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Filter(Distinct(x), p)` → `Distinct(Filter(x, p))`.

    A row-wise predicate commutes with dedup — the distinct rows satisfying `p` equal
    the dedup of the rows satisfying `p` — so filtering first shrinks the input the
    dedup must carry. The predicate references `Distinct`'s (pass-through) columns, so it
    transfers unchanged.
    """
    inner = node.input
    if isinstance(inner, Distinct):
        return Distinct(Filter(inner.input, node.predicate))
    return None


@rule(name="eliminate_sort_before_distinct", phase=Phase.REWRITE, matches=(Distinct,))
def eliminate_sort_before_distinct(node: Distinct, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Distinct(Sort(x))` → `Distinct(x)`. Dedup is order-independent, so an order-only
    sort feeding it is wasted work. Skipped when the sort carries a `limit` (a top-N
    changes *which* rows reach the dedup). Stacked limitless sorts collapse at once."""
    inner = node.input
    if not (isinstance(inner, Sort) and inner.limit is None):
        return None
    while isinstance(inner, Sort) and inner.limit is None:
        inner = inner.input
    return Distinct(inner)


@rule(name="prune_distinct_of_empty", phase=Phase.REWRITE, matches=(Distinct,))
def prune_distinct_of_empty(node: Distinct, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Distinct(Limit(x, 0))` → `Limit(x, 0)`. Deduplicating a provably-empty relation
    yields the same empty relation, so the dedup is pure overhead."""
    inner = node.input
    if isinstance(inner, Limit) and inner.n == 0:
        return inner
    return None
