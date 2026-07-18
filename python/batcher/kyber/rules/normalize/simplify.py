"""Expression simplification — drop the algebraic identities a rewrite leaves behind.

`x AND true → x`, `x OR false → x`, `x + 0 → x`, `x * 1 → x`, `NOT NOT x → x`, and the
redundant `Cast(Cast(x, t), t)`. Only **identity-element** rewrites are applied here;
annihilators (`x AND false → false`, `x OR true → true`), absorption, and complementation
live in `kyber/rules/extra/boolean_algebra.py`. Those *are* sound under the engine's boolean
ops, which are **Kleene** (three-valued): `and_kleene`/`or_kleene` in
`crates/bc-expr/src/eval/binary.rs` make `false` annihilate `AND` and `true` annihilate `OR`
even against a NULL operand. (An earlier version of this note claimed the ops were
*non-Kleene* — the opposite of the truth, and a trap for anyone tempted to "fix" the
correct Kleene annihilators in that sibling module.)

`+ 0` is the one that needs a type, and using an *integer* literal `0` is not enough to
make it safe — that guards the wrong operand. IEEE-754 says `-0.0 + 0.0 = +0.0`, so for a
**float** `x`, `x + 0` is not `x`: it erases the sign of negative zero. Batcher returned
`-0.0` where DuckDB returns `+0.0`, and no differential test could see it because the
comparison harness canonicalizes `±0.0`. The identity therefore fires only when the
*surviving* operand is provably integral; an unknown type is left alone. `- 0`, `* 1` and
`/ 1` need no such guard — each preserves `-0.0` exactly.
"""

from __future__ import annotations

import pyarrow as pa

from batcher.kyber.pass_base import OptimizerContext
from batcher.plan.expr_ir import Binary, Cast, Expr, Lit, Not
from batcher.plan.expr_rewrite import map_node_expressions, transform_expr_up
from batcher.plan.logical import LogicalPlan
from batcher.plan.schema import SchemaRef
from batcher.plan.types import infer_type
from batcher.plan.visitor import transform_up

__all__ = ["ExprSimplification", "simplify_expressions"]


# --- Expression simplification ----------------------------------------------


def simplify_expressions(plan: LogicalPlan) -> LogicalPlan:
    """Apply identity simplifications throughout the plan."""

    def visit(node: LogicalPlan) -> LogicalPlan:
        # A node's expressions are written over its *input's* columns, so that is the
        # schema `+ 0` needs to decide whether the surviving operand is a float.
        child = getattr(node, "input", None)
        schema = child.available_schema() if child is not None else None
        return map_node_expressions(node, lambda e: _simplify_expr(e, schema))

    return transform_up(plan, visit)


def _simplify_expr(expr: Expr, schema: SchemaRef | None = None) -> Expr:
    return transform_expr_up(expr, lambda e: _simplify(e, schema))


def _simplify(expr: Expr, schema: SchemaRef | None = None) -> Expr:
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
        # `x + 0 → x` only when `x` is provably integral — see the module docstring: for a
        # float, IEEE-754 makes `-0.0 + 0.0` equal `+0.0`, so the "identity" changes the
        # value. An unknown type is not a proof, so it is left alone.
        if _is_int_zero(right) and _is_integral(left, schema):
            return left
        if _is_int_zero(left) and _is_integral(right, schema):
            return right
    elif op == "sub":
        # `x - 0` preserves `-0.0` (IEEE: `-0.0 - 0.0 = -0.0`), so it needs no type guard.
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


def _is_integral(expr: Expr, schema: SchemaRef | None) -> bool:
    """Whether `expr` provably has an integer type — the guard `x + 0 → x` requires.

    Returns False when the type cannot be inferred: an unproven type is not a proof, and
    firing on a float would silently rewrite `-0.0` to `+0.0`.
    """
    if isinstance(expr, Lit) and isinstance(expr.value, int) and not isinstance(expr.value, bool):
        return True
    if schema is None:
        return False
    dtype = infer_type(expr, schema)
    return dtype is not None and pa.types.is_integer(dtype)


def _is_int_one(expr: Expr) -> bool:
    return isinstance(expr, Lit) and type(expr.value) is int and expr.value == 1


class ExprSimplification:
    """Pass: drop algebraic identity operations throughout the plan."""

    name = "expr_simplification"

    def apply(self, plan: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan:
        return simplify_expressions(plan)
