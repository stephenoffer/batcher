"""Merge `CASE` branches that agree, and collapse one that re-tests a decided condition.

Both rules here exist because `CASE` chains are generated, not written. A pivot, a
bucketing expression, or a `_sql` translation of `DECODE`/`IIF` produces a chain whose
branches were independent when emitted and are redundant once the surrounding rules have
folded them.

`merge_case_branches_with_equal_results` is the more delicate of the two, because merging
two conditions into `c1 OR c2` has to reproduce `CASE`'s null handling exactly. It does:
`CASE` treats a `NULL` condition as not-taken, and Kleene `OR` answers `true` whenever
either side is `true`, `false` only when both are `false`, and `NULL` otherwise — which
falls through to the `ELSE`, precisely as a pair of non-taken branches would. The merge is
therefore exact for every combination of `true`, `false` and `NULL` in the two conditions,
and only for *adjacent* branches: a branch in between could otherwise be reached first,
and merging across it would change which value wins.

`drop_nested_case_on_settled_condition` uses the other half of the same fact. Inside a
`CASE`'s `ELSE`, the parent's condition is known to be `false` or `NULL` — never `true` —
so a nested `CASE` that tests it again always falls through to its own `ELSE`.
"""

from __future__ import annotations

from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase

# Imported, not re-implemented: `_replaceable` is the conditional family's shared
# type-preservation + purity guard, and `schema_rule` the schema-threading lift. Both
# modules load before this one, so these cannot move any rule's registration.
from batcher.kyber.rules.exprs.guards import schema_rule
from batcher.kyber.rules.extra.nullability import _replaceable
from batcher.kyber.rules.leaf_rewrite import EXPR_NODES, rewrite_node
from batcher.plan.expr_ir import Binary, Case, Expr
from batcher.plan.expr_rewrite import expr_key
from batcher.plan.logical import LogicalPlan

__all__ = [
    "drop_nested_case_on_settled_condition",
    "merge_case_branches_with_equal_results",
]


def _merge_equal_branches(expr: Expr) -> Expr:
    if not isinstance(expr, Case) or len(expr.branches) < 2:
        return expr
    merged: list[tuple[Expr, Expr]] = []
    changed = False
    for cond, value in expr.branches:
        if merged and expr_key(merged[-1][1]) == expr_key(value):
            previous_cond, previous_value = merged[-1]
            merged[-1] = (Binary("or", previous_cond, cond), previous_value)
            changed = True
        else:
            merged.append((cond, value))
    return Case(merged, expr.otherwise) if changed else expr


@rule(
    name="merge_case_branches_with_equal_results",
    phase=Phase.NORMALIZE,
    matches=EXPR_NODES,
    expr=_merge_equal_branches,
    expr_matches=(Case,),
)
def merge_case_branches_with_equal_results(node: LogicalPlan, _ctx) -> LogicalPlan | None:
    """`CASE WHEN c1 THEN v WHEN c2 THEN v ELSE o END -> CASE WHEN c1 OR c2 THEN v ELSE o END`.

    Only *adjacent* branches merge, and only when their values are structurally identical.
    Adjacency is the correctness condition: a branch between the two could be taken first,
    and merging past it would let a later condition win a row an earlier one had claimed.

    The `OR` reproduces `CASE`'s null handling exactly. A `NULL` condition is not-taken,
    and `NULL OR false` is `NULL`, which is also not-taken — so a row that fell through
    both branches still falls through the merged one. Merging shrinks the chain the engine
    must evaluate condition by condition, and hands `case_drop_duplicate_conditions` and
    the boolean normalizer a shape they can work on.
    """
    return rewrite_node(node, _merge_equal_branches)


def _drop_settled_nested_case(expr: Expr, schema) -> Expr:
    if not isinstance(expr, Case) or not expr.branches:
        return expr
    inner = expr.otherwise
    if not isinstance(inner, Case) or not inner.branches:
        return expr
    outer_conditions = {expr_key(cond) for cond, _ in expr.branches}
    kept, dropped = [], []
    for cond, value in inner.branches:
        (dropped if expr_key(cond) in outer_conditions else kept).append((cond, value))
    if not dropped:
        return expr
    reduced = Case(kept, inner.otherwise) if kept else inner.otherwise
    rewritten = Case(list(expr.branches), reduced)
    # A dead branch is dead by *value*, not by *type*: `CASE` takes its type from the join
    # of every arm, so deleting one can narrow the column even though no row could have
    # taken it. `CASE WHEN c THEN 1 ELSE (CASE WHEN c THEN 2.5 ELSE 3 END) END` is a DOUBLE,
    # and dropping the unreachable `2.5` makes it a BIGINT -- a schema change, not an
    # optimization. `_replaceable` settles both that and the purity of what is discarded,
    # and refuses whenever either type is unknown.
    if not _replaceable(expr, rewritten, [arm for pair in dropped for arm in pair], schema):
        return expr
    return rewritten


@rule(
    name="drop_nested_case_on_settled_condition",
    phase=Phase.NORMALIZE,
    matches=EXPR_NODES,
    expr_schema=_drop_settled_nested_case,
    expr_matches=(Case,),
)
def drop_nested_case_on_settled_condition(node: LogicalPlan, _ctx) -> LogicalPlan | None:
    """Drop a nested `CASE` branch whose condition the enclosing `CASE` has already decided.

    `CASE WHEN c THEN a ELSE (CASE WHEN c THEN b ELSE d END) END -> CASE WHEN c THEN a ELSE d END`.

    Reaching the `ELSE` means every outer condition evaluated to `false` or `NULL` — a
    `true` would have been taken. So an inner branch repeating one of them can never fire,
    and its condition and value are dead weight the engine still evaluates. When every
    inner branch is dead the nested `CASE` collapses to its own `ELSE` entirely.

    This is the shape a chain of translated `IIF`/`DECODE` calls produces, where each level
    re-tests the discriminator the level above it already tested.
    """
    return schema_rule(node, _drop_settled_nested_case, carries=(Case,))
