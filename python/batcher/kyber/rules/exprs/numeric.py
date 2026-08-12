"""Numeric algebra the earlier arithmetic families leave on the table.

`extra/arith_algebra` folds constant chains and `extra/arith_extra` collapses nested
math calls and bitwise identities. This module covers the two shapes neither can
reach: identities whose soundness depends on the operand's *type* (integer versus
float) and on its *nullability*, and the self-comparison collapses that DuckDB's
`comparison_simplification` and Spark's `SimplifyBinaryComparison` both perform.

Every type-dependent rule asks the schema first and declines when it cannot answer.
That guard is load-bearing rather than defensive. `x // 1 -> x` is an identity for an
integer and a silent type change for a float, because true division always produces
`Float64`. `x * 0 -> 0` is an identity for a non-nullable integer, wrong for a
nullable one (`NULL * 0` is `NULL`, not `0`), and wrong again for a float
(`inf * 0` is `NaN`). Stating the precondition and checking it exactly is what makes
these safe to run on any input.

The self-comparison collapses (`x = x`, `x < x`) live in the sibling `comparisons`
module. They carry the same preconditions and use the same `schema_rule` plumbing from
`guards`; splitting them keeps both modules inside the line limit.
"""

from __future__ import annotations

from math import gcd

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY, rule
from batcher.kyber.rule import Phase, node_rule
from batcher.kyber.rules.exprs.guards import is_float, is_integer, nullable, schema_rule
from batcher.kyber.rules.leaf_rewrite import node_expr_rule, rewrite_node, safe_expr
from batcher.plan.expr_ir import Expr, Lit
from batcher.plan.expr_ir.core import Binary, IsInf, IsNan, Math2Expr, MathExpr
from batcher.plan.expr_rewrite import expr_key
from batcher.plan.logical import Filter, LogicalPlan, Project
from batcher.plan.schema import SchemaRef

__all__ = [
    "INT_MATH2_FOLD_RULES",
    "SHIFT_FOLD_RULES",
    "drop_float_division_by_one",
    "drop_floor_division_by_one",
    "fold_mod_by_one_integer",
    "fold_mul_by_zero_integer",
    "fold_pow_exponent_zero",
    "gcd_of_self_to_abs",
    "gcd_with_zero_to_abs",
    "hypot_with_zero_to_abs",
    "inf_check_on_integer_to_false",
    "lcm_with_one_to_abs",
    "nan_check_on_integer_to_false",
    "xor_of_self_to_zero",
]


def _int_lit(expr: Expr) -> int | None:
    """The Python int a literal holds, or ``None`` if it is not an integer literal."""
    if isinstance(expr, Lit) and isinstance(expr.value, int) and not isinstance(expr.value, bool):
        return expr.value
    return None


def _is_zero(expr: Expr) -> bool:
    return isinstance(expr, Lit) and expr.value in (0, 0.0) and not isinstance(expr.value, bool)


def _is_one(expr: Expr) -> bool:
    return isinstance(expr, Lit) and expr.value in (1, 1.0) and not isinstance(expr.value, bool)


# --- type-guarded arithmetic identities --------------------------------------


def _float_div_one(expr: Expr, schema: SchemaRef | None) -> Expr:
    if (
        isinstance(expr, Binary)
        and expr.op == "truediv"
        and _is_one(expr.right)
        and is_float(expr.left, schema)
    ):
        return expr.left
    return expr


@rule(
    name="drop_float_division_by_one",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_matches=(Binary,),
    expr_ops=("truediv",),
    expr_schema=_float_div_one,
)
def drop_float_division_by_one(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`f / 1 -> f` for a float `f`. True division always yields `Float64`, so the
    rewrite only preserves the output type when the dividend is already floating
    point. On that operand it is exact for every value the column can hold, NaN,
    the infinities, and negative zero included."""
    return schema_rule(node, _float_div_one, carries=(Binary,))


def _int_floordiv_one(expr: Expr, schema: SchemaRef | None) -> Expr:
    if (
        isinstance(expr, Binary)
        and expr.op == "floor_div"
        and _is_one(expr.right)
        and is_integer(expr.left, schema)
    ):
        return expr.left
    return expr


@rule(
    name="drop_floor_division_by_one",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_matches=(Binary,),
    expr_ops=("floor_div",),
    expr_schema=_int_floordiv_one,
)
def drop_floor_division_by_one(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`i // 1 -> i` for an integer `i`. Floor division by one is the identity on the
    integers and keeps the integer output type, so the whole operation drops out."""
    return schema_rule(node, _int_floordiv_one, carries=(Binary,))


def _mul_zero(expr: Expr, schema: SchemaRef | None) -> Expr:
    if isinstance(expr, Binary) and expr.op == "mul":
        for value, other in ((expr.left, expr.right), (expr.right, expr.left)):
            if (
                _is_zero(other)
                and _int_lit(other) is not None
                and is_integer(value, schema)
                and not nullable(value, schema)
                and safe_expr(value)
            ):
                return Lit(0)
    return expr


@rule(
    name="fold_mul_by_zero_integer",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_matches=(Binary,),
    expr_ops=("mul",),
    expr_schema=_mul_zero,
)
def fold_mul_by_zero_integer(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`i * 0 -> 0` for a non-nullable integer `i`. Both guards are needed: a null
    operand would give `NULL`, not `0`, and a float operand would give `NaN` for an
    infinity. Restricted to a `safe_expr` operand so discarding it cannot drop an
    error the query would otherwise have raised."""
    return schema_rule(node, _mul_zero, carries=(Binary,))


def _mod_one(expr: Expr, schema: SchemaRef | None) -> Expr:
    if (
        isinstance(expr, Binary)
        and expr.op == "mod"
        and _int_lit(expr.right) == 1
        and is_integer(expr.left, schema)
        and not nullable(expr.left, schema)
        and safe_expr(expr.left)
    ):
        return Lit(0)
    return expr


@rule(
    name="fold_mod_by_one_integer",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_matches=(Binary,),
    expr_ops=("mod",),
    expr_schema=_mod_one,
)
def fold_mod_by_one_integer(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`i % 1 -> 0` for a non-nullable integer `i`. Every integer is divisible by one,
    so the remainder is zero regardless of sign. Nullability is checked because
    `NULL % 1` is `NULL`."""
    return schema_rule(node, _mod_one, carries=(Binary,))


def _xor_self(expr: Expr, schema: SchemaRef | None) -> Expr:
    if (
        isinstance(expr, Binary)
        and expr.op == "xor"
        and is_integer(expr.left, schema)
        and not nullable(expr.left, schema)
        and safe_expr(expr.left)
        and expr_key(expr.left) == expr_key(expr.right)
    ):
        return Lit(0)
    return expr


@rule(
    name="xor_of_self_to_zero",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_matches=(Binary,),
    expr_ops=("bit_xor",),
    expr_schema=_xor_self,
)
def xor_of_self_to_zero(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`i ^ i -> 0` for a non-nullable integer `i`. Bitwise exclusive-or of a value
    with itself clears every bit. The nullability guard keeps `NULL ^ NULL`, which is
    `NULL`, out of the rewrite."""
    return schema_rule(node, _xor_self, carries=(Binary,))


def _nan_on_int(expr: Expr, schema: SchemaRef | None) -> Expr:
    if (
        isinstance(expr, IsNan)
        and is_integer(expr.input, schema)
        and not nullable(expr.input, schema)
        and safe_expr(expr.input)
    ):
        return Lit(False)
    return expr


@rule(
    name="nan_check_on_integer_to_false",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_matches=(IsNan,),
    expr_schema=_nan_on_int,
)
def nan_check_on_integer_to_false(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`is_nan(i) -> FALSE` for a non-nullable integer `i`. No integer encodes a NaN,
    so on a non-null row the answer is constant.

    The nullability guard is the whole rule. `is_nan` is *not* a total predicate here:
    it returns `NULL` for a null input, matching DuckDB (`isnan(NULL)` is `NULL`, not
    `FALSE`). Folding a nullable column's test to `FALSE` would turn those nulls into
    false and change what a surrounding `NOT` or `OR` produces."""
    return schema_rule(node, _nan_on_int, carries=(IsNan,))


def _inf_on_int(expr: Expr, schema: SchemaRef | None) -> Expr:
    if (
        isinstance(expr, IsInf)
        and is_integer(expr.input, schema)
        and not nullable(expr.input, schema)
        and safe_expr(expr.input)
    ):
        return Lit(False)
    return expr


@rule(
    name="inf_check_on_integer_to_false",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_matches=(IsInf,),
    expr_schema=_inf_on_int,
)
def inf_check_on_integer_to_false(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`is_inf(i) -> FALSE` for a non-nullable integer `i`. The integers are finite by
    construction. As with the NaN test, `is_inf` propagates null rather than
    answering `FALSE` for one, so a nullable column is left alone."""
    return schema_rule(node, _inf_on_int, carries=(IsInf,))


# --- two-argument math --------------------------------------------------------


def _pow_exponent_zero(expr: Expr, schema: SchemaRef | None) -> Expr:
    if (
        isinstance(expr, Math2Expr)
        and expr.fn == "pow"
        and _is_zero(expr.right)
        and safe_expr(expr.left)
        and not nullable(expr.left, schema)
    ):
        return Lit(1.0)
    return expr


@rule(
    name="fold_pow_exponent_zero",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_matches=(Math2Expr,),
    expr_schema=_pow_exponent_zero,
)
def fold_pow_exponent_zero(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`pow(x, 0) -> 1.0` for a non-nullable `x`.

    The companion `pow(x, 1) -> x` is deliberately **not** implemented, and the
    difference between the two is worth stating because it is not obvious. Both look
    like identities, but `pow` is a computed function and libm's is not correctly
    rounded, so `pow(x, 1) == x` is not provable -- `tests/unit/test_arith_extra.py`
    pins that refusal, and this rule set respects it.

    The zero exponent is a different kind of claim. IEEE 754 specifies `pow(x, +-0)` as
    exactly one for *every* base, NaN and the infinities included; it is a special-case
    table entry rather than a rounded computation, so no approximation is involved.
    Verified against the engine across `+-0.0`, `+-inf`, and NaN.

    Only the null case needs excluding: `pow(NULL, 0)` is `NULL`, not one."""
    return schema_rule(node, _pow_exponent_zero, carries=(Math2Expr,))


def _hypot_zero(expr: Expr) -> Expr:
    if isinstance(expr, Math2Expr) and expr.fn == "hypot":
        if _is_zero(expr.right):
            return MathExpr("abs", expr.left)
        if _is_zero(expr.left):
            return MathExpr("abs", expr.right)
    return expr


@rule(
    name="hypot_with_zero_to_abs",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_hypot_zero,
    expr_matches=(Math2Expr,),
)
def hypot_with_zero_to_abs(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`hypot(x, 0) -> abs(x)`. The Euclidean norm of a single leg is its magnitude.
    The identity is exact rather than approximate, and it holds at the edges too:
    `hypot(-inf, 0)` and `abs(-inf)` are both infinity, and both forms map NaN to
    NaN. Trading a two-argument libm call for a sign mask is a real per-row win."""
    return rewrite_node(node, _hypot_zero)


def _gcd_zero(expr: Expr) -> Expr:
    if isinstance(expr, Math2Expr) and expr.fn == "gcd":
        if _int_lit(expr.right) == 0:
            return MathExpr("abs", expr.left)
        if _int_lit(expr.left) == 0:
            return MathExpr("abs", expr.right)
    return expr


@rule(
    name="gcd_with_zero_to_abs",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_gcd_zero,
    expr_matches=(Math2Expr,),
)
def gcd_with_zero_to_abs(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`gcd(x, 0) -> abs(x)`. Zero is the identity of the greatest-common-divisor
    monoid, and the result is by definition non-negative, which is what `abs`
    supplies. Null propagates identically through both forms."""
    return rewrite_node(node, _gcd_zero)


def _lcm_one(expr: Expr) -> Expr:
    if isinstance(expr, Math2Expr) and expr.fn == "lcm":
        if _int_lit(expr.right) == 1:
            return MathExpr("abs", expr.left)
        if _int_lit(expr.left) == 1:
            return MathExpr("abs", expr.right)
    return expr


@rule(
    name="lcm_with_one_to_abs",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_lcm_one,
    expr_matches=(Math2Expr,),
)
def lcm_with_one_to_abs(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`lcm(x, 1) -> abs(x)`. One is the identity of the least-common-multiple
    monoid, and the least common multiple is non-negative, so `abs` is the exact
    replacement rather than a bare column reference."""
    return rewrite_node(node, _lcm_one)


def _gcd_self(expr: Expr) -> Expr:
    if (
        isinstance(expr, Math2Expr)
        and expr.fn == "gcd"
        and safe_expr(expr.left)
        and expr_key(expr.left) == expr_key(expr.right)
    ):
        return MathExpr("abs", expr.left)
    return expr


@rule(
    name="gcd_of_self_to_abs",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_gcd_self,
    expr_matches=(Math2Expr,),
)
def gcd_of_self_to_abs(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`gcd(x, x) -> abs(x)`. A value's greatest common divisor with itself is its
    magnitude. No nullability guard is needed because `gcd(NULL, NULL)` and
    `abs(NULL)` are both `NULL`; the `safe_expr` guard covers collapsing two
    evaluations of `x` into one."""
    return rewrite_node(node, _gcd_self)


def _fold_int_math2(fn: str):
    """Build the leaf folding one exact integer two-argument function over literals."""

    def leaf(expr: Expr) -> Expr:
        if isinstance(expr, Math2Expr) and expr.fn == fn:
            left, right = _int_lit(expr.left), _int_lit(expr.right)
            if left is not None and right is not None:
                if fn == "gcd":
                    return Lit(gcd(left, right))
                if left == 0 or right == 0:
                    return Lit(0)
                return Lit(abs(left * right) // gcd(left, right))
        return expr

    return leaf


# One rule per exact integer two-argument function. Both have a single mathematical
# answer, so Python's arbitrary-precision result and the engine's agree by definition --
# unlike the transcendentals, which stay unfolded because implementations legitimately
# differ in the last bit.
INT_MATH2_FOLD_RULES = [
    DEFAULT_REGISTRY.add(
        node_rule(
            f"fold_{fn}_of_literals",
            Phase.NORMALIZE,
            node_expr_rule(_fold_int_math2(fn)),
            matches=(Filter, Project),
            expr_fn=_fold_int_math2(fn),
            expr_matches=(Math2Expr,),
            expr_ops=(fn,),
        )
    )
    for fn in ("gcd", "lcm")
]


def _fold_shift_op(op: str):
    """Build the leaf folding one bit shift over two integer literals."""

    def leaf(expr: Expr) -> Expr:
        return _fold_shift_impl(expr, op)

    return leaf


def _fold_shift_impl(expr: Expr, op: str) -> Expr:
    if isinstance(expr, Binary) and expr.op == op:
        left, right = _int_lit(expr.left), _int_lit(expr.right)
        if left is not None and right is not None and 0 <= right < 64:
            value = left << right if op == "shl" else left >> right
            if -(2**63) <= value < 2**63:
                return Lit(value)
    return expr


# One rule per shift direction. The distance must be in `[0, 64)` and the result must fit
# in `Int64`, so the fold never has to model the engine's behaviour for an over-wide shift
# or an overflow -- outside that window the data plane evaluates it as before.
SHIFT_FOLD_RULES = [
    DEFAULT_REGISTRY.add(
        node_rule(
            f"fold_{op}_of_literals",
            Phase.NORMALIZE,
            node_expr_rule(_fold_shift_op(op)),
            matches=(Filter, Project),
            expr_fn=_fold_shift_op(op),
            expr_matches=(Binary,),
            expr_ops=(op,),
        )
    )
    for op in ("shl", "shr")
]
