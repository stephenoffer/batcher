"""Unwrapping a widening cast out of a comparison against a literal.

This is Spark's `UnwrapCastInBinaryComparison` and DataFusion's `unwrap_cast`, and it
is the highest-leverage cast rule there is. `cast(i AS DOUBLE) > 3.5` converts every
row of an integer column to a float and then compares; `i > 3` compares the column as
it sits. The saving is one conversion per row, but the *point* is what it unblocks: a
comparison wearing a cast is opaque to zone-map pruning, to source-level predicate
pushdown, and to the sargable normalizers, so the predicate is stuck where it was
written. Unwrapped, it is a plain `col OP literal` that all three can act on, which
turns a full scan into a pruned one.

`extra/casts` already handles the narrow cases -- a self-cast, a cast to the inferred
type, a narrowing cast pushed down. The general form is what is here, and it splits
into two rules because the arithmetic genuinely differs.

**An integral literal** (`3.0`) maps straight through: every comparison against it
means the same thing over the integers, so all six operators unwrap to the same
operator against `3`.

**A fractional literal** (`3.5`) has no integer equal to it, so the ordered
comparisons shift to the floor and the equalities are excluded. `i > 3.5` is
`i >= 4`, which over the integers is `i > 3`; `i < 3.5` is `i <= 3`. Equality would
fold to a constant `FALSE`, and inequality to `TRUE` -- but the original yields `NULL`
on a null row where the constant does not, so those two are left alone rather than
rewritten into something that is only correct at the top of a filter.

Both rules keep null behaviour exactly: a null input is null through the cast and
null through the comparison on either side of the rewrite, which is what lets these
fire inside a `Project` and not just in a `Filter`.
"""

from __future__ import annotations

import math

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, node_rule
from batcher.kyber.rules.exprs.guards import is_integer, schema_rule
from batcher.plan.expr_ir import Cast, Expr, Lit
from batcher.plan.expr_ir.core import Binary
from batcher.plan.logical import Filter, LogicalPlan, Project
from batcher.plan.schema import SchemaRef

__all__ = [
    "UNWRAP_FRACTIONAL_CAST_RULES",
    "UNWRAP_INTEGRAL_CAST_RULES",
]

#: The comparison operators this module unwraps, and their mirror when the cast is on
#: the right-hand side (`3.5 < cast(i)` reads as `cast(i) > 3.5`).
_MIRROR = {"eq": "eq", "ne": "ne", "lt": "gt", "le": "ge", "gt": "lt", "ge": "le"}

#: Ordered comparisons -- the ones a fractional literal can still be shifted through.
_ORDERED = frozenset({"lt", "le", "gt", "ge"})

#: Beyond 2**53 a float64 cannot represent every integer, so a literal above this could
#: stand for a range of integers rather than one. Outside the window both rules decline.
_EXACT_INT_LIMIT = 2**53

#: Floating-point dtype names a widening integer cast may target.
_FLOAT_DTYPES = frozenset({"float64", "float32", "double", "float"})


def _float_cast_of_integer(expr: Expr, schema: SchemaRef | None) -> Expr | None:
    """The integer operand inside a widening float cast, or ``None``."""
    if (
        isinstance(expr, Cast)
        and expr.dtype in _FLOAT_DTYPES
        and not expr.try_cast
        and is_integer(expr.input, schema)
    ):
        return expr.input
    return None


def _float_literal(expr: Expr) -> float | None:
    if isinstance(expr, Lit) and isinstance(expr.value, float) and math.isfinite(expr.value):
        return expr.value
    return None


def _oriented(expr: Binary, schema: SchemaRef | None):
    """Resolve `expr` to `(integer_operand, op, literal)` with the cast on the left.

    Returns ``None`` unless one side is a widening float cast over an integer and the
    other is a finite float literal.
    """
    inner = _float_cast_of_integer(expr.left, schema)
    if inner is not None:
        value = _float_literal(expr.right)
        if value is not None:
            return inner, expr.op, value
    inner = _float_cast_of_integer(expr.right, schema)
    if inner is not None:
        value = _float_literal(expr.left)
        if value is not None:
            return inner, _MIRROR[expr.op], value
    return None


def _integral(op: str):
    """Build the schema-aware leaf unwrapping one operator against a whole-number literal."""

    def leaf(expr: Expr, schema: SchemaRef | None) -> Expr:
        if not (isinstance(expr, Binary) and expr.op == op):
            return expr
        resolved = _oriented(expr, schema)
        if resolved is None:
            return expr
        inner, resolved_op, value = resolved
        if not value.is_integer() or abs(value) > _EXACT_INT_LIMIT:
            return expr
        return Binary(resolved_op, inner, Lit(int(value)))

    return leaf


def _fractional(op: str):
    """Build the schema-aware leaf unwrapping one ordered operator against a fractional
    literal, shifted to the floor."""

    def leaf(expr: Expr, schema: SchemaRef | None) -> Expr:
        if not (isinstance(expr, Binary) and expr.op == op):
            return expr
        resolved = _oriented(expr, schema)
        if resolved is None:
            return expr
        inner, resolved_op, value = resolved
        if value.is_integer() or abs(value) > _EXACT_INT_LIMIT:
            return expr
        floor = math.floor(value)  # returns an int in Python 3
        # No integer lies strictly between `floor` and `value`, so a strict and a
        # non-strict bound on the same side collapse onto the same integer test.
        if resolved_op in ("gt", "ge"):
            return Binary("gt", inner, Lit(floor))
        return Binary("le", inner, Lit(floor))

    return leaf


def _make_unwrap_rule(leaf):
    def apply(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
        return schema_rule(node, leaf, carries=(Binary,))

    return apply


def _register_unwrap(name: str, leaf, op: str):
    """Register one unwrap rule, declaring its leaf so the driver runs it in the shared
    expression traversal rather than having it walk every expression itself.

    The leaf tests `expr.op == op` before anything else, so the operator declaration is
    exact: no mirroring happens at this level (`_oriented` reorients the *operands*, which
    leaves the node's own operator alone).
    """
    return DEFAULT_REGISTRY.add(
        node_rule(
            name,
            Phase.NORMALIZE,
            _make_unwrap_rule(leaf),
            matches=(Filter, Project),
            expr_schema_fn=leaf,
            expr_matches=(Binary,),
            expr_ops=(op,),
        )
    )


# One rule per operator, over two shared bodies -- the registration shape
# `extra/temporal_sargable` uses for its `(extraction, operator)` cross-product.
#
# **Whole-number literal**, all six operators: `cast(i AS DOUBLE) >= 3.0 -> i >= 3`.
# Comparing the widened integer against a whole number and comparing the integer against
# the truncated literal are the same predicate, every operator, every sign.
UNWRAP_INTEGRAL_CAST_RULES = [
    _register_unwrap(f"unwrap_float_cast_{op}_integral_literal", _integral(op), op)
    for op in sorted(_MIRROR)
]

# **Fractional literal**, ordered operators only: `cast(i AS DOUBLE) > 3.5 -> i > 3`, and
# `< 3.5 -> <= 3`. No integer lies strictly between the literal and its floor, so the
# strict and non-strict forms collapse onto the same integer bound.
#
# Equality is deliberately absent. `cast(i) = 3.5` is false for every integer, so it looks
# foldable to `FALSE` -- but on a null row the original yields `NULL`, and `FALSE` is a
# different value. That rewrite would be sound only at the top level of a filter, and
# these are written to be sound everywhere, so they decline instead. `!=` is excluded for
# the mirror-image reason.
UNWRAP_FRACTIONAL_CAST_RULES = [
    _register_unwrap(f"unwrap_float_cast_{op}_fractional_literal", _fractional(op), op)
    for op in sorted(_ORDERED)
]
