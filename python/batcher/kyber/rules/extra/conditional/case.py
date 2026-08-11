"""CASE / NULLIF / COALESCE rewrites."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase

# `_key` (structural identity), `_rewrite_node` (leaf Expr rule → rebuilt node, or None) and
# `_safe` (deterministic + non-erroring) are the sibling family's helpers, imported rather than
# re-implemented — copy-paste is the one wrong way to share.
from batcher.kyber.rules.extra.boolean_algebra import _key, _rewrite_node
from batcher.kyber.rules.extra.conditional.shared import (
    _droppable,
    _is_false_lit,
    _is_null_lit,
    _is_true_lit,
    _lit_class,
    _pure,
    _rewrite_typed,
)
from batcher.plan.expr_ir import (
    Case,
    Coalesce,
    Expr,
    IsNotNull,
    IsNull,
    Lit,
    NullIf,
)
from batcher.plan.logical import Filter, LogicalPlan, Project
from batcher.plan.schema import SchemaRef

if TYPE_CHECKING:
    from batcher.kyber.rules.exprs.guards import SchemaNode


def _drop_unreachable(expr: Expr, schema: SchemaRef | None = None) -> Expr:
    if not isinstance(expr, Case):
        return expr
    kept = [b for b in expr.branches if not _is_false_lit(b[0])]
    dropped = [t for c, t in expr.branches if _is_false_lit(c)]
    if not dropped or not _droppable(dropped, [t for _, t in kept] + [expr.otherwise], schema):
        return expr
    return Case(kept, expr.otherwise)


@rule(
    name="case_drop_unreachable_branches",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_drop_unreachable,
    expr_schema=_drop_unreachable,
    expr_matches=(Case,),
)
def case_drop_unreachable_branches(node: SchemaNode, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Delete a `WHEN` whose condition is the literal FALSE: a branch fires only where its condition
    is TRUE, so it fires on no row and contributes nothing but its type. (A constant-NULL condition
    is just as dead — NULL selects no rows either — but no NULL *literal* exists here.) The branch
    goes only when its `then` is pure and its type is already carried by a surviving arm, so neither
    the error behavior nor the CASE's result type moves.

    "Already carried by a surviving arm" is answered schema-free *and*, where the node's schema
    resolves, against the arms' exact Arrow types — without which two `int` columns look like two
    unknowns and the branch is never dropped."""
    return _rewrite_typed(node, _drop_unreachable, carries=(Case,))


def _first_true(expr: Expr, schema: SchemaRef | None = None) -> Expr:
    if not isinstance(expr, Case):
        return expr
    i = next((i for i, (c, _) in enumerate(expr.branches) if _is_true_lit(c)), None)
    if i is None:
        return expr
    head, winner, tail = expr.branches[:i], expr.branches[i][1], expr.branches[i + 1 :]
    dropped = [t for _, t in tail] + [expr.otherwise]
    kept = [t for _, t in head] + [winner]
    if not all(_pure(c) for c, _ in tail) or not _droppable(dropped, kept, schema):
        return expr
    return Case(head, winner)


@rule(
    name="case_first_true_branch_wins",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_first_true,
    expr_schema=_first_true,
    expr_matches=(Case,),
)
def case_first_true_branch_wins(node: SchemaNode, _ctx: OptimizerContext) -> LogicalPlan | None:
    """A `WHEN` whose condition is the literal TRUE becomes the `ELSE`, and every branch after it
    (plus the old `ELSE`) is deleted: it fires on every row that got past the earlier branches, so
    its result *is* the default and nothing below it is reachable. Only a literal TRUE qualifies —
    FALSE or NULL says nothing about the branches beneath. Dropped conditions and results must be
    pure, and the result type must survive."""
    return _rewrite_typed(node, _first_true, carries=(Case,))


def _all_same_result(expr: Expr) -> Expr:
    if not isinstance(expr, Case) or not expr.branches:
        return expr
    results = [t for _, t in expr.branches] + [expr.otherwise]
    if len({_key(r) for r in results}) != 1 or not all(_pure(c) for c, _ in expr.branches):
        return expr
    return results[0]


@rule(
    name="case_all_branches_same_result",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_all_same_result,
    expr_matches=(Case,),
)
def case_all_branches_same_result(node: SchemaNode, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Every arm (each `then` *and* the `otherwise`) is the same expression → that expression.
    Whichever branch a row selects it yields the same value, so the conditional *is* that value; and
    since every arm has the identical type, the join that types the CASE is that type too. The
    conditions are dropped, so they must be pure."""
    return _rewrite_node(node, _all_same_result)


def _no_branches(expr: Expr) -> Expr:
    if isinstance(expr, Case) and not expr.branches:
        return expr.otherwise
    return expr


@rule(
    name="case_no_branches_to_else",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_no_branches,
    expr_matches=(Case,),
)
def case_no_branches_to_else(node: SchemaNode, _ctx: OptimizerContext) -> LogicalPlan | None:
    """A `CASE` with no `WHEN` branches left is its `ELSE`: with nothing to select, every row falls
    through to the default, and the type join over one arm is its own type. This is the collapse the
    other CASE rules feed — once they delete the last unreachable branch, the husk disappears."""
    return _rewrite_node(node, _no_branches)


def _dedup_conditions(expr: Expr, schema: SchemaRef | None = None) -> Expr:
    if not isinstance(expr, Case) or len(expr.branches) < 2:
        return expr
    kept: list[tuple[Expr, Expr]] = []
    dropped: list[Expr] = []
    seen: set[str] = set()
    for cond, then in expr.branches:
        if _key(cond) in seen and _pure(cond):
            dropped.append(then)
            continue
        seen.add(_key(cond))
        kept.append((cond, then))
    if not dropped or not _droppable(dropped, [t for _, t in kept] + [expr.otherwise], schema):
        return expr
    return Case(kept, expr.otherwise)


@rule(
    name="case_drop_duplicate_conditions",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_dedup_conditions,
    expr_schema=_dedup_conditions,
    expr_matches=(Case,),
)
def case_drop_duplicate_conditions(node: SchemaNode, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Delete a `WHEN` whose condition repeats an earlier branch's. Wherever the later condition is
    TRUE the earlier (structurally identical, and required to be pure, hence equal-valued) one was
    TRUE too — and first-true-wins already fired it — so the repeat is unreachable. Its result is
    removed under the usual purity + type guard."""
    return _rewrite_typed(node, _dedup_conditions, carries=(Case,))


def _case_to_coalesce(expr: Expr) -> Expr:
    if not isinstance(expr, Case) or len(expr.branches) != 1:
        return expr
    cond, then = expr.branches[0]
    if isinstance(cond, IsNotNull) and _pure(cond.input) and _key(cond.input) == _key(then):
        return Coalesce([then, expr.otherwise])  # WHEN x IS NOT NULL THEN x ELSE y
    if isinstance(cond, IsNull) and _pure(cond.input) and _key(cond.input) == _key(expr.otherwise):
        return Coalesce([expr.otherwise, then])  # WHEN x IS NULL THEN y ELSE x
    return expr


@rule(
    name="case_to_coalesce",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_case_to_coalesce,
    expr_matches=(Case, IsNotNull, IsNull),
)
def case_to_coalesce(node: SchemaNode, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`CASE WHEN x IS NOT NULL THEN x ELSE y END` → `coalesce(x, y)`, and the mirrored
    `CASE WHEN x IS NULL THEN y ELSE x END`. Both compute "x unless it is null, then y":
    `IS NOT NULL` is total (never NULL), so the branch fires on exactly the non-null rows — exactly
    where COALESCE takes `x`. Both arms survive verbatim, so the type join is unchanged; `x` appears
    twice in the CASE, so it must be pure for the two occurrences to agree."""
    return _rewrite_node(node, _case_to_coalesce)


def _nullif_distinct_literals(expr: Expr) -> Expr:
    if not (
        isinstance(expr, NullIf) and isinstance(expr.left, Lit) and isinstance(expr.right, Lit)
    ):
        return expr
    left, right = expr.left, expr.right
    cls = _lit_class(left.value)
    if cls is None or cls != _lit_class(right.value) or left.value == right.value:
        return expr
    if cls == "float" and (math.isnan(left.value) or math.isnan(right.value)):
        return expr
    return left


@rule(
    name="nullif_distinct_literals",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_nullif_distinct_literals,
    expr_matches=(Lit, NullIf),
)
def nullif_distinct_literals(node: SchemaNode, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`NULLIF(a, b)` over two *distinct* literals of one type → `a`. NULLIF nulls its left operand
    exactly where `left = right`; two unequal constants are never equal, so the result is `a` on
    every row. Guarded three ways: one type class for both (so the result type stays `a`'s, not the
    join `NULLIF(1, 2.5)` would take); NaN is refused (`NaN = NaN` is TRUE in SQL but False in
    Python, so a NaN pair is not "distinct"); and `-0.0 == 0.0` in Python, so a signed-zero pair is
    correctly *not* distinct and does not fire."""
    return _rewrite_node(node, _nullif_distinct_literals)


def _coalesce_flatten(expr: Expr) -> Expr:
    if not isinstance(expr, Coalesce) or not any(isinstance(a, Coalesce) for a in expr.inputs):
        return expr
    flat: list[Expr] = []
    for arg in expr.inputs:
        flat.extend(arg.inputs if isinstance(arg, Coalesce) else [arg])
    return Coalesce(flat)


@rule(
    name="coalesce_flatten_nested",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_coalesce_flatten,
    expr_matches=(Coalesce,),
)
def coalesce_flatten_nested(node: SchemaNode, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`coalesce(a, coalesce(b, c))` → `coalesce(a, b, c)`. "First non-null" is associative: the
    nested call is null exactly when `b` and `c` are both null, which is exactly when the flat form
    moves past them. No argument is added or removed, so the type join and the error behavior are
    identical and no purity guard is needed; bottom-up, one pass splices every level."""
    return _rewrite_node(node, _coalesce_flatten)


def _coalesce_drop_unreachable(expr: Expr, schema: SchemaRef | None = None) -> Expr:
    if not isinstance(expr, Coalesce):
        return expr
    kept = [a for a in expr.inputs if not _is_null_lit(a)]
    # A `Lit` is never null (no NULL literal here), so it wins once reached; the rest is dead code.
    first_lit = next((i for i, a in enumerate(kept) if isinstance(a, Lit)), None)
    if first_lit is not None:
        kept = kept[: first_lit + 1]
    # Nothing to do — or every argument was a typed NULL, whose type *is* the value: leave it.
    if len(kept) == len(expr.inputs) or not kept:
        return expr
    kept_keys = {_key(a) for a in kept}
    if not _droppable([a for a in expr.inputs if _key(a) not in kept_keys], kept, schema):
        return expr
    return Coalesce(kept)


@rule(
    name="coalesce_drop_nulls_after_first_non_null",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_coalesce_drop_unreachable,
    expr_schema=_coalesce_drop_unreachable,
    expr_matches=(Coalesce, Lit),
)
def coalesce_drop_nulls_after_first_non_null(
    node: SchemaNode, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """Delete a `COALESCE` argument that is a constant NULL, and truncate everything after the first
    constant non-NULL. A provably-null argument is skipped on every row and can never be the answer;
    a `Lit` is never null, so it is *always* the answer once reached and all behind it is dead code.
    Dropped arguments must be pure and must not be the sole carrier of the result type: dropping the
    `NULL::double` from `coalesce(int_col, NULL::double)` would narrow DOUBLE to INT."""
    return _rewrite_typed(node, _coalesce_drop_unreachable, carries=(Coalesce, Lit))


def _coalesce_single(expr: Expr) -> Expr:
    if isinstance(expr, Coalesce) and len(expr.inputs) == 1:
        return expr.inputs[0]
    return expr


@rule(
    name="coalesce_single_arg",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_coalesce_single,
    expr_matches=(Coalesce,),
)
def coalesce_single_arg(node: SchemaNode, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`coalesce(x)` → `x`. The first non-null of one argument is that argument (a null `x` yields
    null either way), and the type join over one arm is that arm's type — the collapse the other
    COALESCE rules feed once they have removed the redundant arguments."""
    return _rewrite_node(node, _coalesce_single)


def _coalesce_dedup(expr: Expr) -> Expr:
    if not isinstance(expr, Coalesce) or len(expr.inputs) < 2:
        return expr
    kept: list[Expr] = [expr.inputs[0]]
    for arg in expr.inputs[1:]:
        if not (_key(arg) == _key(kept[-1]) and _pure(arg)):
            kept.append(arg)
    if len(kept) == len(expr.inputs):
        return expr
    return Coalesce(kept)


@rule(
    name="coalesce_dedup_args",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_coalesce_dedup,
    expr_matches=(Coalesce,),
)
def coalesce_dedup_args(node: SchemaNode, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`coalesce(a, a, b)` → `coalesce(a, b)` — drop an argument identical to the one before it.
    COALESCE only advances past an argument that evaluated to null, so the repeat is reached only in
    the rows where it is *itself* null: it can never be the answer. The surviving twin has the same
    type, so the join is untouched; the dropped copy must be pure, so the two provably agree."""
    return _rewrite_node(node, _coalesce_dedup)
