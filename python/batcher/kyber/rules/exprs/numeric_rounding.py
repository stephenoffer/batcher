"""Rounding calls whose digit argument makes them a different function.

Split out of `exprs/numeric`, which holds the arithmetic identities and the integer folds.
`round(x, 0)` is the one-argument `round`, and saying so lets the rounding-range family in
`math_algebra` recognize it -- which the two-argument spelling blocks.

Registration order is run order, so this module is imported from `exprs/__init__` directly
after `numeric` -- the position this rule held when the two were one file.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.rules.exprs.numeric import _int_lit
from batcher.kyber.rules.leaf_rewrite import rewrite_node
from batcher.plan.expr_ir import Expr, MathExpr
from batcher.plan.expr_ir.core import Math2Expr
from batcher.plan.logical import Filter, LogicalPlan, Project

__all__ = ["round_with_zero_digits"]


def _round_zero_digits(expr: Expr) -> Expr:
    if isinstance(expr, Math2Expr) and expr.fn == "round" and _int_lit(expr.right) == 0:
        return MathExpr("round", expr.left)
    return expr


@rule(
    name="round_with_zero_digits",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_round_zero_digits,
    expr_matches=(Math2Expr,),
)
def round_with_zero_digits(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`round(x, 0) -> round(x)`. Rounding to zero decimal places is the one-argument
    form, which skips the scaling multiply and divide the general path performs. It
    also feeds the existing nested-rounding collapse, which only recognizes the
    single-argument node."""
    return rewrite_node(node, _round_zero_digits)
