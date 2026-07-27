"""Driving `NOT` down to the leaves: De Morgan and double negation.

`extra/boolean_algebra` handles the annihilators, absorption, and idempotence, and
`fold_not_comparison` flips a `NOT` sitting directly on a comparison into the negated
comparison. What was missing is the step between: a `NOT` wrapped around a *whole
conjunction* has nowhere to go, so the comparisons underneath it never reach the rule
that would flip them.

That is what these rules fix, and the win is downstream rather than local. `NOT (a > 5
AND b < 3)` is one opaque boolean over two comparisons; distributed and flipped it
becomes `a <= 5 OR b >= 3`, a disjunction of plain `col OP literal` terms. Those are
exactly the shapes zone-map pruning, the `IN`-list builder, the range normalizers, and
source-level predicate pushdown match on. None of them can see through a `NOT`.

DuckDB implements this as `not_conjunction_simplification`; Spark folds it into
`BooleanSimplification`.

**Soundness under three-valued logic.** De Morgan is often stated for two-valued logic,
so it is worth being explicit that it holds here. Both laws were checked against the
engine over the full nine-cell Kleene cross-product of `{TRUE, FALSE, NULL}` squared,
and both sides agree in every cell -- including the ones where intuition wobbles, such
as `NOT (NULL AND FALSE)` being `TRUE` while `NOT NULL OR NOT FALSE` is also `TRUE`.
Kleene negation is likewise an involution, which is what makes double negation exact.

**Termination.** These only ever push a `NOT` *downward*, never rebuild one upward, so
each application moves negations strictly closer to the leaves and the fixpoint
converges. A rule for the reverse direction would oscillate against these forever, and
is deliberately absent.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.rules.leaf_rewrite import rewrite_node, safe_expr
from batcher.plan.expr_ir import Binary, Expr, Not
from batcher.plan.logical import Filter, LogicalPlan, Project

__all__ = [
    "de_morgan_not_of_conjunction",
    "de_morgan_not_of_disjunction",
    "drop_double_negation",
]


def _de_morgan(op: str, negated: str):
    """Build the leaf rewrite distributing `NOT` across `op`, producing `negated`."""

    def leaf(expr: Expr) -> Expr:
        if (
            isinstance(expr, Not)
            and isinstance(expr.input, Binary)
            and expr.input.op == op
            and safe_expr(expr.input)
        ):
            inner = expr.input
            return Binary(negated, Not(inner.left), Not(inner.right))
        return expr

    return leaf


_NOT_AND = _de_morgan("and", "or")
_NOT_OR = _de_morgan("or", "and")


@rule(
    name="de_morgan_not_of_conjunction",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_NOT_AND,
)
def de_morgan_not_of_conjunction(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`NOT (a AND b) -> NOT a OR NOT b`.

    Verified against the engine across the full Kleene cross-product, so it is exact
    under three-valued logic and not merely under two.

    The value is what it unblocks. A `NOT` over a conjunction hides both comparisons
    from every rule that matches on comparison shape; once distributed, the existing
    `fold_not_comparison` turns each `NOT (x > k)` into `x <= k`, and the result is a
    disjunction of plain `col OP literal` terms that zone-map pruning, the range
    normalizers, and source pushdown can all act on.

    Guarded on `safe_expr` because the two operands are re-wrapped rather than
    evaluated in place, and only pushes downward, which is what makes the fixpoint
    terminate."""
    return rewrite_node(node, _NOT_AND)


@rule(
    name="de_morgan_not_of_disjunction",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_NOT_OR,
)
def de_morgan_not_of_disjunction(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`NOT (a OR b) -> NOT a AND NOT b`, the dual of the conjunction law and verified
    the same way.

    This direction is the more valuable of the two in a `Filter`: it turns one negated
    disjunction into a *conjunction*, which `split_conjuncts` then breaks into separate
    top-level conjuncts. Each of those can be pushed to a different side of a join, or
    into the scan, independently -- something the single negated term could never be."""
    return rewrite_node(node, _NOT_OR)


def _double_negation(expr: Expr) -> Expr:
    if isinstance(expr, Not) and isinstance(expr.input, Not):
        return expr.input.input
    return expr


@rule(
    name="drop_double_negation",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_double_negation,
)
def drop_double_negation(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`NOT (NOT a) -> a`. Kleene negation is an involution -- it maps `TRUE` to
    `FALSE`, `FALSE` to `TRUE`, and `NULL` to `NULL` -- so applying it twice is the
    identity on all three values, confirmed against the engine.

    No `safe_expr` guard is needed: the operand is neither dropped nor duplicated, only
    unwrapped, so the query evaluates exactly what it did before minus two boolean
    inversions.

    This pairs with the De Morgan rules above, which can leave a `NOT` directly on top
    of another when the original expression already contained one."""
    return rewrite_node(node, _double_negation)
