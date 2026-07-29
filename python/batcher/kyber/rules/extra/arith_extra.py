"""NORMALIZE-phase arithmetic the other families leave on the table — math-function
collapsing and the integer *bitwise* algebra.

`normalize.simplify` already drops the single-operator identities (`x + 0`, `x * 1`,
`x / 1`, `NOT NOT x`), `normalize.fold` folds a `Binary` over two literals (including
every comparison), and `arith_algebra` reassociates and factors integer `+`/`-`/`*`.
Nothing, however, looks at `MathExpr` (`abs`/`floor`/`sign`/…) or at the five bitwise
operators — which is where this module lives: nested math functions collapse, a math
call over a constant folds away, and the bitwise identity elements (`x | 0`, `x ^ 0`,
`x & -1`, `x << 0`, `x & x`) disappear.

Soundness is anchored to the engine's actual kernels, not to intuition:

* `bc_expr::eval_math` promotes an Int64 array to Float64 and applies the `f64`
  function — *except* `abs`, which stays Int64. So `floor`/`ceil`/`trunc`/`round` of an
  **integer** expression is exactly `cast(x, float64)` (the value is already integral),
  and every rounding function is the identity on an integral float — which is what makes
  the nesting collapses exact, NaN and ±inf included.
* `sign` is `v > 0 ? 1 : v < 0 ? -1 : 0` — its output is one of three float64 values, so
  it is idempotent; `abs` is `f64::abs`/`i64::abs`, also idempotent.
* The bitwise ops **cast both operands to Int64** before applying the arrow kernel. That
  makes the identity rules type-sensitive, not just value-sensitive: `bit_or(f, 0)` on a
  Float64 `f` *returns Int64*, so dropping the `| 0` would change the column's type. Every
  bitwise rule below therefore fires only when the surviving operand is **already Int64**
  per `infer_type`.

NULL is preserved throughout: every rewrite keeps the same operand under the same
null-propagating kernel, and no rule ever replaces a possibly-null value with a literal.

Deliberately **not** implemented (each is unsound, and a rule we cannot prove must not
ship):

* `x * 0 → 0`, `x & 0 → 0`, `x ^ x → 0`, `x - x → 0`, `x / x → 1` — all destroy NULL
  (`NULL * 0` is NULL, not 0), and the float forms additionally break on NaN/±inf
  (`NaN * 0` is NaN) and on a zero divisor.
* `-(-x) → x` for floats — `0 - (0 - (-0.0))` is `+0.0`, not `-0.0`. (The *integer* case
  is already handled by `arith_algebra.fold_neg_sub`.)
* `abs(sqrt(x)) → sqrt(x)` — `sqrt(-0.0)` is `-0.0`, and `abs` would flip its sign bit.
* `pow(x, 1) → x` and `pow(x, 0) → 1` — libm's `pow` is not a correctly-rounded
  operation, so `x.powf(1.0) == x` is a property we cannot prove for every input; and the
  `0` form destroys NULL besides.
* folding `ln`/`exp`/`log`/the trig family, `cbrt`, `atan2`/`hypot` over literals — none
  is correctly rounded by IEEE-754, so Python's libm and Rust's need not agree bit-for-bit.
* folding a **shift** of two literals — the arrow kernel's behavior for a shift count of
  64 or more is not one we can pin from here.
"""

from __future__ import annotations

import math

import pyarrow as pa

from batcher.kyber.pass_base import OptimizerContext

# `_is_int_lit` (a non-bool integer literal) and the boolean family's `_key` (structural
# identity), `_rewrite_node` (leaf Expr rule → rebuilt node, or None) and `_safe`
# (deterministic + non-erroring) are imported, never re-implemented — copy-paste is the one
# wrong way to share between rule families.
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.rules.exprs.guards import schema_rule
from batcher.kyber.rules.extra.arith_algebra import _is_int_lit
from batcher.kyber.rules.extra.boolean_algebra import _key, _rewrite_node, _safe
from batcher.plan.expr_ir import Binary, Cast, Expr, Lit
from batcher.plan.expr_ir.core import MathExpr
from batcher.plan.logical import Aggregate, Filter, LogicalPlan, Project, Sort, Window
from batcher.plan.schema import SchemaRef
from batcher.plan.types import infer_type

__all__ = [
    "abs_of_negation",
    "bit_and_minus_one",
    "bit_and_self",
    "bit_or_self",
    "bit_or_zero",
    "bit_xor_zero",
    "collapse_idempotent_math_fn",
    "collapse_nested_rounding",
    "fold_bitwise_literals",
    "fold_math_of_int_literal",
    "rounding_of_int_is_cast",
    "shift_by_zero",
]

# Every node whose expressions these rules rewrite (each has a single `.input`, whose
# schema types the expressions).
_NODES = (Filter, Project, Aggregate, Sort, Window)

_INT64_MIN = -(2**63)

# Unary math functions that are their own fixpoint: `f(f(x)) == f(x)` for every input,
# NaN and ±inf included. `abs`/`sign` by definition; the four rounding functions because
# their output is integral and each is the identity on an integral value.
# `rint` (round-half-to-even) belongs with the other rounding functions in all three of the
# roles these sets play: it is its own fixpoint on an integral value, it is the identity on
# the integral result of any of its siblings, and over an *integer* literal it is the
# engine's int→f64 promotion and nothing more. `even` is deliberately absent — it rounds
# away from zero to the next *even* integer (`even(5)` is 6), so it is idempotent but is not
# the identity on an integral value and does not fold to a plain promotion.
_IDEMPOTENT_MATH = frozenset({"abs", "sign", "floor", "ceil", "trunc", "round", "rint"})
# The rounding family — each maps any float to an integral float (or NaN/±inf).
_ROUNDING = frozenset({"floor", "ceil", "trunc", "round", "rint"})
# Unary math functions whose value over an *integer* literal is exactly computable here.
_FOLDABLE_MATH = _IDEMPOTENT_MATH | {"sqrt"}
# The bitwise logic ops (associative/commutative, and closed over i64). Each is also
# named alone, because `_drop_neutral` takes the *set* of ops the shift rule needs.
_BIT_AND = frozenset({"bit_and"})
_BIT_OR = frozenset({"bit_or"})
_BIT_XOR = frozenset({"bit_xor"})
_BITWISE_LOGIC = _BIT_AND | _BIT_OR | _BIT_XOR
_SHIFTS = frozenset({"shift_left", "shift_right"})
# The two numeric types the engine's math kernels accept (post-FFI widening, a column is
# never narrower than these). A rule that reasons about a math result's type is confined
# to them: `0 - float32_expr` promotes to Float64, so a narrower type would move.
_NUMERIC = (pa.int64(), pa.float64())


#: What `schema_rule` pre-checks a node for before threading its schema through.
_CARRIES = (Binary, MathExpr)


def _is_int64(expr: Expr, schema: SchemaRef | None) -> bool:
    """Whether `expr` is *provably* Int64 — the guard every bitwise rule needs, because
    the bitwise kernels cast their operands to Int64 and so would silently change a
    Float64 (or Int32) operand's type if the operation were dropped."""
    return schema is not None and infer_type(expr, schema) == pa.int64()


def _is_numeric(expr: Expr, schema: SchemaRef | None) -> bool:
    """Whether `expr` is provably Int64 or Float64 (the engine's two math types)."""
    return schema is not None and infer_type(expr, schema) in _NUMERIC


# --- math-function collapsing -----------------------------------------------


def _negated(expr: Expr) -> Expr | None:
    """The operand of a negation: `0 - x` (what `-x` desugars to) → `x`, else `None`."""
    if (
        isinstance(expr, Binary)
        and expr.op == "sub"
        and _is_int_lit(expr.left)
        and expr.left.value == 0
    ):
        return expr.right
    return None


def _abs_of_negation(expr: Expr, schema: SchemaRef | None) -> Expr:
    if isinstance(expr, MathExpr) and expr.fn == "abs":
        inner = _negated(expr.input)
        if inner is not None and _is_numeric(inner, schema):
            return MathExpr("abs", inner)
    return expr


@rule(
    name="abs_of_negation",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr_matches=(MathExpr,),
    expr_ops=("abs",),
    expr_schema=_abs_of_negation,
)
def abs_of_negation(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`abs(-x) → abs(x)` (the negation lowers to `0 - x`), for an Int64/Float64 `x`.

    Exact on both of the engine's numeric types. For floats, `abs(0.0 - v) == abs(v)` for
    every `v` — including `±0.0` (both magnitudes are `0.0`) and NaN (which stays NaN
    through both). For integers the subtraction *wraps*, so `0 - INT64_MIN` is `INT64_MIN`
    — but the right-hand side computes `abs(INT64_MIN)` too, so the two sides agree
    whatever `abs` does at that boundary. The type guard is load-bearing: a Float32 `x`
    would promote to Float64 through the subtraction, so dropping it would *narrow* the
    result. NULL propagates identically on both sides; one kernel pass instead of two.
    """
    return schema_rule(node, _abs_of_negation, carries=_CARRIES)


def _collapse_idempotent(expr: Expr) -> Expr:
    if (
        isinstance(expr, MathExpr)
        and expr.fn in _IDEMPOTENT_MATH
        and isinstance(expr.input, MathExpr)
        and expr.input.fn == expr.fn
    ):
        return expr.input
    return expr


@rule(
    name="collapse_idempotent_math_fn",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr=_collapse_idempotent,
    expr_matches=(MathExpr,),
)
def collapse_idempotent_math_fn(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`f(f(x)) → f(x)` for the idempotent unary math functions — `abs`, `sign`, `floor`,
    `ceil`, `trunc`, `round`.

    Each is its own fixpoint over the engine's kernels: `abs` of a non-negative value is
    itself (NaN → NaN, `-0.0` → `0.0` → `0.0`); `sign` yields one of `1.0/-1.0/0.0`, each
    of which is its own sign; and each rounding function returns an integral float (or
    NaN/±inf), on which it is the identity. The outer call is therefore pure overhead. The
    output type cannot move — the two calls are the *same* function, and `f(f(x))` and
    `f(x)` are typed identically (`abs` preserves its input type, the rest yield Float64).
    Nulls propagate through both. No type guard is needed, and none would help.
    """
    return _rewrite_node(node, _collapse_idempotent)


def _collapse_nested_rounding(expr: Expr) -> Expr:
    if (
        isinstance(expr, MathExpr)
        and expr.fn in _ROUNDING
        and isinstance(expr.input, MathExpr)
        and expr.input.fn in _ROUNDING
        and expr.input.fn != expr.fn
    ):
        return expr.input
    return expr


@rule(
    name="collapse_nested_rounding",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr=_collapse_nested_rounding,
    expr_matches=(MathExpr,),
)
def collapse_nested_rounding(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop the outer of two *different* nested rounding functions: `floor(round(x))` →
    `round(x)`, `ceil(trunc(x))` → `trunc(x)`, and every other pair drawn from
    `{floor, ceil, trunc, round}`.

    The inner call already produced an integral float, and each of the four is the identity
    on an integral value — as it is on NaN and on ±inf, the only non-integral outputs
    possible. (`-0.0` survives too: `floor(-0.0)` is `-0.0`.) Both calls yield Float64, so
    the output type is unchanged whichever survives, and nulls propagate through both. The
    same-function case is `collapse_idempotent_math_fn`; this is its mixed-pair sibling,
    and the two together collapse a whole chain bottom-up in one pass.
    """
    return _rewrite_node(node, _collapse_nested_rounding)


def _rounding_of_int(expr: Expr, schema: SchemaRef | None) -> Expr:
    if isinstance(expr, MathExpr) and expr.fn in _ROUNDING and _is_int64(expr.input, schema):
        return Cast(expr.input, "float64")
    return expr


@rule(
    name="rounding_of_int_is_cast",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr_matches=(MathExpr,),
    expr_ops=tuple(sorted(_ROUNDING)),
    expr_schema=_rounding_of_int,
)
def rounding_of_int_is_cast(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`floor(i) / ceil(i) / trunc(i) / round(i)` over an **Int64** expression → `cast(i,
    float64)` — the cast the engine was going to do anyway, with the now-pointless math
    kernel removed.

    `bc_expr::eval_math` promotes an Int64 array to Float64 and *then* applies the float
    function; that promoted value is always integral (an f64 built from an i64 is integral
    at every magnitude — beyond 2^53 the cast rounds, but it rounds to an integer), and all
    four functions are the identity on an integral float. So the whole call collapses to its
    own first step. The result stays Float64 (both the rounding functions and the cast
    produce it), and nulls propagate through the cast exactly as through the kernel. Fires
    only on a provably Int64 operand — over a float, rounding is real work.
    """
    return schema_rule(node, _rounding_of_int, carries=_CARRIES)


# --- literal folding (the engine's kernels, evaluated at plan time) ----------


def _math_of_int(fn: str, value: int) -> Lit | None:
    """`fn(<int literal>)` as a `Lit`, or `None` where the fold is not provably exact."""
    if fn == "abs":
        # `i64::abs(INT64_MIN)` has no i64 result (the engine wraps or traps); Python's
        # arbitrary-precision `abs` would silently produce 2**63, a different number.
        return None if value == _INT64_MIN else Lit(abs(value))
    if fn == "sign":
        return Lit(float((value > 0) - (value < 0)))
    if fn == "sqrt":
        # A negative operand yields NaN in the engine, and this IR has no NaN literal to
        # fold to. Otherwise: `sqrt` is one of the five operations IEEE-754 *requires* to
        # be correctly rounded, so Python's and Rust's agree bit-for-bit, over the same
        # int→f64 promotion (round-to-nearest-even in both).
        return None if value < 0 else Lit(math.sqrt(value))
    # floor/ceil/trunc/round: the engine promotes the integer to f64 and applies the
    # function, which is the identity on the (integral) result — so the fold is the
    # promotion alone. `float(value)` rounds exactly as the engine's cast does.
    return Lit(float(value))


def _fold_math_lit(expr: Expr) -> Expr:
    if isinstance(expr, MathExpr) and expr.fn in _FOLDABLE_MATH and _is_int_lit(expr.input):
        folded = _math_of_int(expr.fn, expr.input.value)
        if folded is not None:
            return folded
    return expr


@rule(
    name="fold_math_of_int_literal",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr=_fold_math_lit,
    expr_matches=(MathExpr,),
)
def fold_math_of_int_literal(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Evaluate a unary math function over an **integer literal** at plan time: `abs(-5)` →
    `5`, `sign(-5)` → `-1.0`, `floor(5)` → `5.0`, `sqrt(4)` → `2.0`.

    `normalize.fold` folds a `Binary` over two literals but never looks at `MathExpr`, so
    these survive to the data plane as a per-row kernel over a constant. Only the functions
    whose value is *exactly* computable here fold, and only over an integer literal: the
    rounding family is the identity on an integer (so the fold is just the engine's own
    Float64 promotion), `sign` yields one of three exact floats, `sqrt` is correctly rounded
    by IEEE-754 mandate (Python and Rust cannot disagree). `abs(INT64_MIN)` and `sqrt(<0)`
    are refused (no i64 result / no NaN literal), as are `ln`/`exp`/the trig family (libm is
    not correctly rounded) and every *float* literal (`-0.0` and NaN make the fold's identity
    observable). The output type is preserved exactly: `abs` folds to an int literal (Int64,
    as `abs` preserves its input type), everything else to a float literal (Float64).
    """
    return _rewrite_node(node, _fold_math_lit)


def _fold_bitwise(expr: Expr) -> Expr:
    if not (isinstance(expr, Binary) and expr.op in _BITWISE_LOGIC):
        return expr
    if not (_is_int_lit(expr.left) and _is_int_lit(expr.right)):
        return expr
    a, b = expr.left.value, expr.right.value
    if expr.op == "bit_and":
        return Lit(a & b)
    if expr.op == "bit_or":
        return Lit(a | b)
    return Lit(a ^ b)


@rule(
    name="fold_bitwise_literals",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr=_fold_bitwise,
    expr_matches=(Binary,),
    expr_ops=tuple(sorted(_BITWISE_LOGIC)),
)
def fold_bitwise_literals(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Fold `bit_and` / `bit_or` / `bit_xor` over two integer literals: `6 & 3` → `2`.

    `normalize.fold` folds arithmetic, boolean and comparison literals but stops short of
    the bitwise family. The fold is exact: the engine casts both operands to Int64 and
    applies the arrow kernel, and Python's `& | ^` agree bit-for-bit with two's complement
    on any pair of i64 values (negatives included — Python's infinite-precision sign
    extension is the same bit pattern), with the result closed under i64, so no overflow
    check is needed. The result is an integer literal (Int64), the type the operation
    already had. Shifts are *not* folded — the kernel's behavior at a shift count of 64 or
    more is not something we can pin from the control plane.
    """
    return _rewrite_node(node, _fold_bitwise)


# --- bitwise identity elements ----------------------------------------------


def _drop_neutral(
    expr: Expr, schema: SchemaRef | None, ops: frozenset[str], neutral: int, *, commutes: bool
) -> Expr:
    """`x OP k → x` when `k` is `OP`'s neutral literal and `x` is provably Int64 (also
    `k OP x` when `OP` commutes). The Int64 guard is what keeps the *type* still: the
    kernels cast to Int64, so a Float64 `x` would change type if the op were dropped."""
    if not (isinstance(expr, Binary) and expr.op in ops):
        return expr
    left, right = expr.left, expr.right
    if _is_int_lit(right) and right.value == neutral and _is_int64(left, schema):
        return left
    if commutes and _is_int_lit(left) and left.value == neutral and _is_int64(right, schema):
        return right
    return expr


# --- the schema leaves, named so the registrations below can declare them ----------------
# Declaring the leaf lets the driver run these inside the single expression traversal it
# already makes per node, instead of each rule walking every expression itself.


def _bit_or_zero_leaf(expr: Expr, schema: SchemaRef | None) -> Expr:
    return _drop_neutral(expr, schema, _BIT_OR, 0, commutes=True)


def _bit_xor_zero_leaf(expr: Expr, schema: SchemaRef | None) -> Expr:
    return _drop_neutral(expr, schema, _BIT_XOR, 0, commutes=True)


def _bit_and_minus_one_leaf(expr: Expr, schema: SchemaRef | None) -> Expr:
    return _drop_neutral(expr, schema, _BIT_AND, -1, commutes=True)


def _shift_by_zero_leaf(expr: Expr, schema: SchemaRef | None) -> Expr:
    return _drop_neutral(expr, schema, _SHIFTS, 0, commutes=False)


def _bit_and_self_leaf(expr: Expr, schema: SchemaRef | None) -> Expr:
    return _self_bitwise(expr, schema, "bit_and")


def _bit_or_self_leaf(expr: Expr, schema: SchemaRef | None) -> Expr:
    return _self_bitwise(expr, schema, "bit_or")


@rule(name="bit_or_zero", phase=Phase.NORMALIZE, matches=_NODES, expr_schema=_bit_or_zero_leaf)
def bit_or_zero(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`x | 0 → x` (and `0 | x → x`) for an Int64 `x`. Zero is OR's identity element in
    every bit position, and the kernel propagates nulls, so a null `x` is null on both
    sides — no `NULL`-destroying shortcut here, the operand simply survives. Gated on a
    provably Int64 `x` because the bitwise kernels *cast* their operands to Int64: over a
    Float64 column, `f | 0` is an Int64 expression, and dropping the `| 0` would hand back
    a Float64 one.
    """
    return schema_rule(node, _bit_or_zero_leaf, carries=_CARRIES)


@rule(name="bit_xor_zero", phase=Phase.NORMALIZE, matches=_NODES, expr_schema=_bit_xor_zero_leaf)
def bit_xor_zero(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`x ^ 0 → x` (and `0 ^ x → x`) for an Int64 `x`. XOR with an all-zero pattern flips
    no bit, and nulls propagate through the kernel, so the operand survives unchanged in
    every row. Same Int64 guard as `bit_or_zero` (the kernels cast to Int64). Note the
    sibling `x ^ x → 0` is *not* implemented: it is false for a null `x` (`NULL ^ NULL` is
    NULL, not 0).
    """
    return schema_rule(node, _bit_xor_zero_leaf, carries=_CARRIES)


@rule(
    name="bit_and_minus_one",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr_schema=_bit_and_minus_one_leaf,
)
def bit_and_minus_one(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`x & -1 → x` (and `-1 & x → x`) for an Int64 `x`. In two's complement `-1` is the
    all-ones mask — AND's identity element — so every bit of `x` survives, and a null `x`
    stays null through the kernel. Gated on a provably Int64 `x` (the kernels cast to
    Int64). The companion `x & 0 → 0` is refused: it would turn a NULL into a zero.
    """
    return schema_rule(node, _bit_and_minus_one_leaf, carries=_CARRIES)


@rule(name="shift_by_zero", phase=Phase.NORMALIZE, matches=_NODES, expr_schema=_shift_by_zero_leaf)
def shift_by_zero(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`x << 0 → x` and `x >> 0 → x` for an Int64 `x`. A zero-bit shift moves nothing, in
    either direction and at either sign, and the kernel propagates nulls. Not commutative
    (only a zero *shift count* qualifies — `0 << x` is a real computation), and gated on a
    provably Int64 `x`, since the shift kernels cast their operands to Int64.
    """
    return schema_rule(node, _shift_by_zero_leaf, carries=_CARRIES)


def _self_bitwise(expr: Expr, schema: SchemaRef | None, op: str) -> Expr:
    if (
        isinstance(expr, Binary)
        and expr.op == op
        and _safe(expr.left)
        and _key(expr.left) == _key(expr.right)
        and _is_int64(expr.left, schema)
    ):
        return expr.left
    return expr


@rule(name="bit_and_self", phase=Phase.NORMALIZE, matches=_NODES, expr_schema=_bit_and_self_leaf)
def bit_and_self(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`x & x → x` for an Int64 `x`. Bitwise AND is idempotent in every bit position, and
    the kernel's null propagation agrees (`NULL & NULL` is NULL, which is what a lone
    `NULL` is too) — so unlike `x ^ x`, this one survives three-valued logic. The dropped
    copy must be `_safe` (deterministic and non-erroring), so collapsing two evaluations
    into one changes neither the value nor whether the query errors; and `x` must be
    provably Int64, since the kernel casts its operands there.
    """
    return schema_rule(node, _bit_and_self_leaf, carries=_CARRIES)


@rule(name="bit_or_self", phase=Phase.NORMALIZE, matches=_NODES, expr_schema=_bit_or_self_leaf)
def bit_or_self(node: LogicalPlan, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`x | x → x` for an Int64 `x` — the dual of `bit_and_self`, with the same proof:
    bitwise OR is idempotent bit by bit, `NULL | NULL` is NULL, the dropped copy must be
    `_safe`, and the operand must be provably Int64 (the kernel casts to Int64).
    """
    return schema_rule(node, _bit_or_self_leaf, carries=_CARRIES)
