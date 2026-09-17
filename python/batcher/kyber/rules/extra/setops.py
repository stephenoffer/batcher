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
The one rule that does read that shape, `set_membership_to_join`, swaps it for a semi or
anti join where statistics prove NULLs cannot make the two disagree.
"""

from __future__ import annotations

import json

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.plan.expr_ir import Binary, Col, Lit, Not
from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Filter,
    Join,
    JoinOutputCol,
    Limit,
    LogicalPlan,
    Project,
    Projection,
    Union,
)
from batcher.plan.logical._setops import (
    MEMBERSHIP_IN_LEFT,
    MEMBERSHIP_IN_RIGHT,
    MEMBERSHIP_LEFT_TAG,
    MEMBERSHIP_RIGHT_TAG,
)

__all__ = [
    "dedup_distinct_union_branches",
    "drop_distinct_in_distinct_union",
    "flatten_nested_union",
    "fold_distinct_union_all",
    "prune_distinct_of_empty",
    "prune_empty_union_branch",
    "push_filter_through_distinct",
    "push_project_through_union",
    "set_membership_to_join",
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
        # A keyed dedup in a branch is not made redundant by the union's own dedup: the union
        # dedups whole rows, while the branch collapsed rows that differ outside its key.
        while isinstance(stripped, Distinct) and not stripped.keys:
            stripped = stripped.input
        if stripped is not branch:
            changed = True
        new_inputs.append(stripped)
    if not changed:
        return None
    return Union(tuple(new_inputs), True)


# `eliminate_sort_in_distinct_union_branch` and `eliminate_sort_before_distinct` used to
# live here. Both dropped an order-only `Sort` feeding a dedup, reasoning that "a distinct
# produces a set, so its rows carry no meaningful order". The output *multiset* is indeed
# order-independent; the output *row order* is not, and Batcher's `Distinct` preserves its
# input's order. So the rewrite changed observable results — `sort("k").distinct().limit(3)`
# returned `[5, 3, 1]` with the optimizer on and `[1, 2, 3]` with it off — which breaks the
# rule contract that a pass is semantics-preserving (the plan changes, the result does not).
# Removed rather than narrowed: soundness needs to know whether anything downstream observes
# the order, and a local `Distinct`/`Union` rewrite has no view of its parent.


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

    Whole-row dedup only. It does **not** commute with a keyed one: that keeps a chosen row
    per key, so filtering first changes which rows are available to be chosen. `distinct(["k"],
    keep="first", order_by="ts").filter(v > 5)` takes each key's earliest row and keeps it only
    if it clears the predicate; filtering first would instead take the earliest row *among those
    clearing it*, which is a different row and a different number of them.
    """
    inner = node.input
    if isinstance(inner, Distinct) and not inner.keys:
        return Distinct(Filter(inner.input, node.predicate))
    return None


@rule(name="prune_distinct_of_empty", phase=Phase.REWRITE, matches=(Distinct,))
def prune_distinct_of_empty(node: Distinct, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Distinct(Limit(x, 0))` → `Limit(x, 0)`. Deduplicating a provably-empty relation
    yields the same empty relation, so the dedup is pure overhead."""
    inner = node.input
    if isinstance(inner, Limit) and inner.n == 0:
        return inner
    return None


def _membership_branch(
    branch: LogicalPlan, cols: tuple[str, ...], left: bool
) -> LogicalPlan | None:
    """The untagged relation under one tagged union branch, or `None` if it is not one.

    A branch is `Project(x, cols..., tag_l=<left>, tag_r=<not left>)`, exactly as
    `Dataset._set_membership` builds it. Returned as `Project(x, cols...)`, so whatever rules
    have since done to `x` is kept and only the two tag literals are dropped.
    """
    if not isinstance(branch, Project):
        return None
    items = {p.alias: p.expr for p in branch.items}
    want_tags = {MEMBERSHIP_LEFT_TAG: left, MEMBERSHIP_RIGHT_TAG: not left}
    if len(branch.items) != len(cols) + 2 or set(items) != {*cols, *want_tags}:
        return None
    for tag, value in want_tags.items():
        lit = items[tag]
        if not (isinstance(lit, Lit) and lit.value is value):
            return None
    return Project(branch.input, tuple(p for p in branch.items if p.alias in cols))


def _membership_kind(predicate: object) -> str | None:
    """``"semi"`` for the INTERSECT filter `in_l AND in_r`, ``"anti"`` for EXCEPT's
    `in_l AND NOT in_r`, else `None`."""
    if not (
        isinstance(predicate, Binary)
        and predicate.op == "and"
        and isinstance(predicate.left, Col)
        and predicate.left.name == MEMBERSHIP_IN_LEFT
    ):
        return None
    right = predicate.right
    if isinstance(right, Col) and right.name == MEMBERSHIP_IN_RIGHT:
        return "semi"
    if (
        isinstance(right, Not)
        and isinstance(right.input, Col)
        and right.input.name == MEMBERSHIP_IN_RIGHT
    ):
        return "anti"
    return None


@rule(name="set_membership_to_join", phase=Phase.REWRITE, matches=(Project,))
def set_membership_to_join(node: Project, ctx: OptimizerContext) -> LogicalPlan | None:
    """DISTINCT `INTERSECT`/`EXCEPT` → `Distinct(semi/anti Join)` when NULLs cannot differ.

    `Dataset._set_membership` lowers both through a tagged union and a group-by over every
    column, because grouping treats NULL as equal to NULL — SQL set semantics — and a join
    does not. That aggregate hashes *both* inputs whole. A semi or anti join builds only the
    right side and probes the left, which measured 24 ms against 52 ms for TPC-H
    `orders EXCEPT lineitem` on the order key.

    The two agree exactly when **every column is proven null-free on at least one side**,
    from an exact null count. A NULL can only matter where both rows hold it in the same
    column — only then does grouping pair two rows a join would not — and that needs a NULL
    on both sides of that column. Under the condition, a left row carrying a NULL has no equal
    row on the right under either semantics, so the join keeps it out of an INTERSECT and in
    an EXCEPT exactly as the aggregate does. `Distinct` then restores the set semantics the
    aggregate gave by construction; applying it after the join is exact for both kinds, since
    membership is a property of the row value, not of which duplicate carried it.

    The branch column types must also be identical, or the union's type widening and the
    join's key typing would disagree on the output schema. Only the DISTINCT forms are
    matched: the ALL forms group by an extra ordinal, so their group keys differ from the
    projected columns and the pattern does not fire.
    """
    cols = tuple(p.alias for p in node.items)
    if not all(isinstance(p.expr, Col) and p.expr.name == p.alias for p in node.items):
        return None
    filt = node.input
    if not isinstance(filt, Filter):
        return None
    kind = _membership_kind(filt.predicate)
    agg = filt.input
    if kind is None or not isinstance(agg, Aggregate) or agg.watermark is not None:
        return None
    if tuple(g.alias for g in agg.group_keys) != cols or not all(
        isinstance(g.expr, Col) and g.expr.name == g.alias for g in agg.group_keys
    ):
        return None
    tags = {(a.alias, a.agg.func, getattr(a.agg.input, "name", None)) for a in agg.aggregates}
    if tags != {
        (MEMBERSHIP_IN_LEFT, "bool_or", MEMBERSHIP_LEFT_TAG),
        (MEMBERSHIP_IN_RIGHT, "bool_or", MEMBERSHIP_RIGHT_TAG),
    }:
        return None
    union = agg.input
    if not isinstance(union, Union) or union.distinct or len(union.inputs) != 2:
        return None
    left = _membership_branch(union.inputs[0], cols, left=True)
    right = _membership_branch(union.inputs[1], cols, left=False)
    if left is None or right is None:
        return None
    # The union widens differing branch types to a common supertype (`int32` with `int64`
    # yields `int64`), where a join keeps its left side's type or refuses the pair outright.
    # Identical types make the two agree on the output schema as well as on the rows.
    left_schema, right_schema = left.available_schema(), right.available_schema()
    if left_schema is None or right_schema is None:
        return None
    if any(left_schema.field(c).type != right_schema.field(c).type for c in cols):
        return None
    left_nn = ctx.estimator.estimate(left).non_null_columns()
    right_nn = ctx.estimator.estimate(right).non_null_columns()
    if not all(c in left_nn or c in right_nn for c in cols):
        return None
    output = tuple(JoinOutputCol("left", c, c) for c in cols)
    join = Join(left, right, cols, cols, kind, output)
    return Project(Distinct(join), tuple(Projection(c, Col(c)) for c in cols))
