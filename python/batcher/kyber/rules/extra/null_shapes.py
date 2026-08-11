"""Null-check rewrites driven by an expression's *shape* rather than by column nullability.

Split out of `extra/nullability`, which answers "can this column ever be null?" from the
schema. These rules need no such answer: `coalesce`'s own definition says when it is null,
`NOT (x IS NULL)` is `x IS NOT NULL` by the meaning of the operators, and a `count` over a
column the schema marks non-nullable is `count(*)` outright.

Registration order is run order, so this module is imported from `extra/__init__` directly
after `nullability` -- the position these rules held when the two were one file.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.rules.exprs.guards import SchemaNode
from batcher.kyber.rules.extra.boolean_algebra import _rewrite_node
from batcher.kyber.rules.extra.nullability import _EXPR_NODES, _never_null, _non_null_cols
from batcher.plan.expr_ir import AggExpr, Coalesce, Expr, IsNotNull, IsNull, Not
from batcher.plan.expr_rewrite import combine_conjuncts, combine_disjuncts
from batcher.plan.logical import (
    Aggregate,
    AggregateSpec,
    LogicalPlan,
    Sort,
    SortKeySpec,
)

__all__ = [
    "canonicalize_not_null_check",
    "count_of_non_nullable_column_to_count_star",
    "drop_null_ordering_on_non_nullable_sort_key",
    "expand_is_not_null_of_coalesce",
    "expand_is_null_of_coalesce",
]


def _expand_is_null_coalesce(expr: Expr) -> Expr:
    if isinstance(expr, IsNull) and isinstance(expr.input, Coalesce):
        args = expr.input.inputs
        if len(args) >= 2:
            return combine_conjuncts([IsNull(arg) for arg in args])
    return expr


@rule(
    name="expand_is_null_of_coalesce",
    phase=Phase.NORMALIZE,
    matches=_EXPR_NODES,
    expr=_expand_is_null_coalesce,
    expr_matches=(Coalesce, IsNull),
)
def expand_is_null_of_coalesce(node: SchemaNode, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`coalesce(a, b) IS NULL` → `a IS NULL AND b IS NULL`.

    COALESCE is NULL exactly when every argument is NULL, so the two forms agree on every row
    — and the expanded form is per-*column* null checks, which is what the schema rules above
    (and zone-map pruning, and predicate pushdown) can actually reason about: over a NOT NULL
    `a` the conjunction collapses to FALSE. No purity guard is needed — both forms evaluate
    every argument (the engine is vectorized; COALESCE does not short-circuit), so value and
    error behavior are unchanged. The conjunction is balanced, and the output holds no
    `IS NULL(COALESCE(…))`, so re-applying is a no-op.
    """
    return _rewrite_node(node, _expand_is_null_coalesce)


def _expand_is_not_null_coalesce(expr: Expr) -> Expr:
    if isinstance(expr, IsNotNull) and isinstance(expr.input, Coalesce):
        args = expr.input.inputs
        if len(args) >= 2:
            return combine_disjuncts([IsNotNull(arg) for arg in args])
    return expr


@rule(
    name="expand_is_not_null_of_coalesce",
    phase=Phase.NORMALIZE,
    matches=_EXPR_NODES,
    expr=_expand_is_not_null_coalesce,
    expr_matches=(Coalesce, IsNotNull),
)
def expand_is_not_null_of_coalesce(node: SchemaNode, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`coalesce(a, b) IS NOT NULL` → `a IS NOT NULL OR b IS NOT NULL` — the De Morgan dual
    of `expand_is_null_of_coalesce`: a COALESCE is non-null exactly when *some* argument is.
    Both forms evaluate every argument, so no purity guard is needed; over a NOT NULL `a`
    the disjunction collapses to TRUE and the filter disappears entirely."""
    return _rewrite_node(node, _expand_is_not_null_coalesce)


# --- canonical spelling of a negated null check -----------------------------


def _canonicalize_not_null_check(expr: Expr) -> Expr:
    if isinstance(expr, Not):
        if isinstance(expr.input, IsNull):
            return IsNotNull(expr.input.input)
        if isinstance(expr.input, IsNotNull):
            return IsNull(expr.input.input)
    return expr


@rule(
    name="canonicalize_not_null_check",
    phase=Phase.NORMALIZE,
    matches=_EXPR_NODES,
    expr=_canonicalize_not_null_check,
    expr_matches=(IsNotNull, IsNull, Not),
)
def canonicalize_not_null_check(node: SchemaNode, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`NOT (x IS NULL)` → `x IS NOT NULL`, and `NOT (x IS NOT NULL)` → `x IS NULL`.

    Exact under three-valued logic precisely *because* the null checks are total: `IS NULL`
    never yields NULL, so its `NOT` is a plain boolean complement (for an ordinary nullable
    predicate this would fail — `NOT NULL` is NULL). Canonicalizing the two spellings lets
    every rule that pattern-matches `IsNull`/`IsNotNull` see a shape it recognizes instead of
    an opaque `NOT`. The output contains no `NOT(IsNull|IsNotNull)`, so it is idempotent.
    """
    return _rewrite_node(node, _canonicalize_not_null_check)


# --- aggregate / ordering: nullability the operator no longer has to check ---


@rule(name="count_of_non_nullable_column_to_count_star", phase=Phase.REWRITE, matches=(Aggregate,))
def count_of_non_nullable_column_to_count_star(
    node: Aggregate, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`COUNT(x)` → `COUNT(*)` when `x` provably never yields NULL.

    COUNT counts the rows where its argument is non-null; over a NOT NULL column that is every
    row — exactly `COUNT(*)`, which needs no per-row null check and no column at all. It
    generalizes `count_constant_to_count_star` from a non-null *literal* to any never-null
    expression, and holds for grouped and global aggregates alike (both count 0 over empty
    input). Only the unary `count` is touched — `count_distinct` counts values, not rows.
    """
    non_null = _non_null_cols(node.input)
    if not non_null:
        return None
    new: list[AggregateSpec] = []
    changed = False
    for spec in node.aggregates:
        agg = spec.agg
        countable = (
            agg.func == "count"
            and agg.input is not None
            and agg.input2 is None
            and agg.param is None
            and _never_null(agg.input, non_null)
        )
        if countable:
            new.append(AggregateSpec(spec.alias, AggExpr("count_star", None)))
            changed = True
        else:
            new.append(spec)
    if not changed:
        return None
    return Aggregate(node.input, node.group_keys, tuple(new), node.watermark)


@rule(name="drop_null_ordering_on_non_nullable_sort_key", phase=Phase.NORMALIZE, matches=(Sort,))
def drop_null_ordering_on_non_nullable_sort_key(
    node: Sort, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """Reset `nulls_first` to the default on a sort key that provably holds no NULL.

    Where a key can never be NULL, the null-placement flag selects between two orderings that
    are the same ordering — the sort is bit-identical either way, and the *plan* is now in
    canonical form. That matters because the ordering rules compare sort keys structurally:
    `dedupe_sort_keys` and `sort_elimination_from_ordering` read two keys differing only in a
    null placement they cannot observe as *different*, and decline a valid rewrite. Only ever
    relaxes `nulls_first=True` → the default `False`; a nullable key is left as written.
    """
    non_null = _non_null_cols(node.input)
    if not non_null:
        return None
    new_keys: list[SortKeySpec] = []
    changed = False
    for key in node.keys:
        if key.nulls_first and _never_null(key.expr, non_null):
            new_keys.append(SortKeySpec(key.expr, key.descending, False))
            changed = True
        else:
            new_keys.append(key)
    if not changed:
        return None
    return Sort(node.input, tuple(new_keys), node.limit)
