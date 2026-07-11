"""Expression simplification — drop the algebraic identities a rewrite leaves behind.

`x AND true → x`, `x OR false → x`, `x + 0 → x`, `x * 1 → x`, `NOT NOT x → x`, and the
redundant `Cast(Cast(x, t), t)`. Only **identity-element** rewrites are applied, never
annihilators (the engine's boolean ops are non-Kleene), and the numeric identities use
*integer* `0`/`1` only.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.plan.expr_ir import Binary, Cast, Expr, Lit, Not
from batcher.plan.expr_rewrite import map_node_expressions, transform_expr_up
from batcher.plan.logical import LogicalPlan
from batcher.plan.visitor import transform_up

__all__ = ["ExprSimplification", "simplify_expressions"]


# --- Expression simplification ----------------------------------------------


def simplify_expressions(plan: LogicalPlan) -> LogicalPlan:
    """Apply identity simplifications throughout the plan."""
    return transform_up(plan, lambda n: map_node_expressions(n, _simplify_expr))


def _simplify_expr(expr: Expr) -> Expr:
    return transform_expr_up(expr, _simplify)


def _simplify(expr: Expr) -> Expr:
    if isinstance(expr, Not) and isinstance(expr.input, Not):
        return expr.input.input  # NOT NOT x → x

    # Cast(Cast(x, t), t) → Cast(x, t): casting to a type then to that same type again
    # is redundant. Only when the dtype AND try_cast semantics match — a strict cast
    # wrapping a try-cast (or vice versa) is not equivalent (different null behavior).
    if (
        isinstance(expr, Cast)
        and isinstance(expr.input, Cast)
        and expr.input.dtype == expr.dtype
        and expr.input.try_cast == expr.try_cast
    ):
        return Cast(expr.input.input, expr.dtype, try_cast=expr.try_cast)

    if not isinstance(expr, Binary):
        return expr
    op, left, right = expr.op, expr.left, expr.right

    if op == "and":
        if _is_true(right):
            return left
        if _is_true(left):
            return right
    elif op == "or":
        if _is_false(right):
            return left
        if _is_false(left):
            return right
    elif op == "add":
        if _is_int_zero(right):
            return left
        if _is_int_zero(left):
            return right
    elif op == "sub":
        if _is_int_zero(right):
            return left
    elif op == "mul":
        if _is_int_one(right):
            return left
        if _is_int_one(left):
            return right
    elif op == "div":
        if _is_int_one(right):
            return left
    return expr


def _is_true(expr: Expr) -> bool:
    return isinstance(expr, Lit) and expr.value is True


def _is_false(expr: Expr) -> bool:
    return isinstance(expr, Lit) and expr.value is False


def _is_int_zero(expr: Expr) -> bool:
    return isinstance(expr, Lit) and type(expr.value) is int and expr.value == 0


def _is_int_one(expr: Expr) -> bool:
    return isinstance(expr, Lit) and type(expr.value) is int and expr.value == 1


class ExprSimplification:
    """Pass: drop algebraic identity operations throughout the plan."""

    name = "expr_simplification"

    def apply(self, plan: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan:
        return simplify_expressions(plan)
