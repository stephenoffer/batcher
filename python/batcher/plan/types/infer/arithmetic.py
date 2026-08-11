"""Output types for the arithmetic families: binary operators and the math functions.

These are the rules that need their *operands'* types, not just an operator name, so each
entry point takes the dispatcher's `infer` callable and resolves operands lazily. Passing
the callable rather than importing it keeps the dependency one-way -- `dispatch` imports
this module, never the reverse -- while preserving the short-circuit that lets a comparison
answer `bool` without descending into either subtree.

Every rule here is derived from `bc_expr::eval`, not assumed from the operator's name. The
comments record which engine arm each one mirrors, because the tempting general rule is
wrong in a specific way for `div`, `abs`, `round`, `bit_count` and the decimals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from batcher.plan.types.lattice import promote, widen

if TYPE_CHECKING:
    from collections.abc import Callable

    from batcher.plan.expr_ir import Expr
    from batcher.plan.schema import SchemaRef

    #: Resolve a sub-expression's type: the dispatcher's `infer_type`, threaded in.
    InferFn = Callable[[Expr, SchemaRef], pa.DataType | None]

__all__ = ["binary_type", "math2func_type", "mathfunc_type"]

# Binary operator -> output-type category. Comparisons and logical ops yield bool;
# bit/shift ops yield int64; arithmetic promotes its operands (then widens to the
# engine's int64/float64 output). `div`, `floor_div` and `concat` are absent here
# because each carries a rule these flat sets cannot express, and `binary_type`
# handles them in their own arms -- notably `div`, which is integer division (Int64)
# on two integers rather than the unconditional double it was once assumed to be.
_BINARY_BOOL = frozenset({"gt", "ge", "lt", "le", "eq", "ne", "and", "or"})
_BINARY_INT = frozenset({"bit_and", "bit_or", "bit_xor", "shift_left", "shift_right"})
_BINARY_ARITH = frozenset({"add", "sub", "mul", "mod"})

#: Unary math functions whose result is defined by the two's-complement i64 bits rather
#: than an f64 approximation, so they yield Int64 for *every* input type. Mirrors the
#: `BitCount | Factorial` arm of `bc_expr::eval::math::eval_math`, which routes them to
#: `eval_int_math` precisely because an f64 round-trip both mistyped the schema and gave
#: wrong answers above 2^53.
_MATH_INT_RESULT = frozenset({"bit_count", "factorial"})

#: Unary math functions that keep an integer input integral. `bc_expr` special-cases both
#: (`(Abs, Int64)` and `(Round, Int64)`) because each is integer-valued on an integer and
#: DuckDB returns BIGINT for it; promoting to f64 first corrupted values above 2^53.
#: `floor`/`ceil`/`sqrt` genuinely do yield double there, so they are deliberately absent.
_MATH_TYPE_PRESERVING = frozenset({"abs", "round"})

#: Binary math functions returning Int64 whatever their operands' types -- the integer arms
#: of `bc_expr::eval::math::eval_math2`. A GCD or LCM is an integer quantity by definition.
_MATH2_INT_RESULT = frozenset({"gcd", "lcm"})

# Arrow / DataFusion decimal128 arithmetic precision+scale rules (verified against the
# engine). `div` is intentionally excluded -- its scale rule is not reproduced here, so it
# stays uncertain (-> ``None``) as the engine may return a decimal of an inferred scale.
_DECIMAL_MAX_PRECISION = 38


def _widened_numeric(t: pa.DataType) -> pa.DataType | None:
    """`t` widened to the engine's output width if numeric, else ``None``.

    A decimal or a non-numeric input is left uncertain on purpose: the engine's promotion
    for those is not reproduced here, and this module is sound rather than complete.
    """
    return widen(t) if pa.types.is_integer(t) or pa.types.is_floating(t) else None


def mathfunc_type(expr: object, schema: SchemaRef, infer: InferFn) -> pa.DataType | None:
    """The result type of a unary `MathExpr`.

    Derived from the engine rather than assumed: most unary math promotes to Float64, but
    three arms of `bc_expr::eval::math::eval_math` do not, and treating them as double made
    `Dataset.schema` advertise a type the engine never produces.
    """
    fn = expr.fn  # type: ignore[attr-defined]
    if fn in _MATH_INT_RESULT:
        return pa.int64()
    if fn in _MATH_TYPE_PRESERVING:
        operand = infer(expr.input, schema)  # type: ignore[attr-defined]
        return None if operand is None else _widened_numeric(operand)
    return pa.float64()


def math2func_type(expr: object, schema: SchemaRef, infer: InferFn) -> pa.DataType | None:
    """The result type of a binary `Math2Expr`.

    `pow`/`atan2`/`hypot`/`next_after` are Float64. `gcd`/`lcm` are Int64. `round` follows
    its *left* operand, because `bc_expr` routes `round(Int64, n)` to `round_int` and keeps
    it Int64 -- DuckDB returns BIGINT for `round(bigint, n)`, and the f64 round-trip
    corrupted values above 2^53.
    """
    fn = expr.fn  # type: ignore[attr-defined]
    if fn in _MATH2_INT_RESULT:
        return pa.int64()
    if fn == "round":
        left = infer(expr.left, schema)  # type: ignore[attr-defined]
        return None if left is None else _widened_numeric(left)
    return pa.float64()


def _arith_promotable(dtype: pa.DataType) -> bool:
    """Whether the promotion lattice's answer for `dtype` is also its arithmetic result.

    True for the plain numerics and for ``null`` (which adopts the other operand, and then
    the arithmetic is that operand's). False for a decimal -- whose arithmetic derives a
    *new* precision and scale from both operands -- and for every temporal and string type,
    whose arithmetic is not a widening at all.
    """
    return pa.types.is_null(dtype) or pa.types.is_integer(dtype) or pa.types.is_floating(dtype)


def _both_temporal(left: pa.DataType, right: pa.DataType) -> bool:
    """Whether both operands are Date or Timestamp."""
    temporal = (pa.types.is_date, pa.types.is_timestamp)
    return all(any(p(t) for p in temporal) for t in (left, right))


def _numeric_with_bool(left: pa.DataType, right: pa.DataType) -> pa.DataType | None:
    """The arithmetic result type when exactly one operand is Boolean.

    The engine coerces a Boolean operand to the other side's numeric type (`true` is 1),
    so `int + bool` is Int64 and `float * bool` is Double. Without this the pair fell to
    ``None``, because `promote` has no numeric answer for a Boolean.
    """
    pairs = ((left, right), (right, left))
    for other, flag in pairs:
        if pa.types.is_boolean(flag) and not pa.types.is_boolean(other):
            return _widened_numeric(other)
    return None


def _both_numeric(left: pa.DataType, right: pa.DataType) -> bool:
    """Whether both operands are plain (non-decimal) numerics."""
    numeric = (pa.types.is_integer, pa.types.is_floating)
    return all(any(p(t) for p in numeric) for t in (left, right))


def _int_preserving_div(left: pa.DataType, right: pa.DataType) -> pa.DataType | None:
    """Int64 for two integers, Float64 once either side floats, ``None`` otherwise.

    Shared by `div` and `floor_div`, which agree on this rule. A decimal operand is left
    uncertain: the engine evaluates it as Float64, but that is a fallback rather than a
    derived decimal rule.
    """
    if not _both_numeric(left, right):
        return None
    both_int = pa.types.is_integer(left) and pa.types.is_integer(right)
    return pa.int64() if both_int else pa.float64()


def _temporal_sub(left: pa.DataType, right: pa.DataType) -> pa.DataType | None:
    """The result of subtracting two temporals, or ``None`` if the pair is not temporal.

    DATE - DATE is the integer count of days between the two dates (matching the engine
    and DuckDB), not a date or an interval -- so the public schema must say Int64, not
    date32. Subtracting two *instants* is an elapsed time instead: the engine returns
    Duration(us) whenever either side is a Timestamp, so `date - date` is tested first.
    """
    if pa.types.is_date(left) and pa.types.is_date(right):
        return pa.int64()
    if _both_temporal(left, right) and (
        pa.types.is_timestamp(left) or pa.types.is_timestamp(right)
    ):
        return pa.duration("us")
    return None


def _date_shift(op: str, left: pa.DataType, right: pa.DataType) -> pa.DataType | None:
    """DATE +/- <integer days>, which shifts the date and keeps the date type.

    `int + date` is commutative. Matches the engine and DuckDB (`DATE - 5` -> a DATE).
    """
    if pa.types.is_date(left) and pa.types.is_integer(right):
        return left
    if op == "add" and pa.types.is_integer(left) and pa.types.is_date(right):
        return right
    return None


def _decimal_arith_type(op: str, left: pa.DataType, right: pa.DataType) -> pa.DataType | None:
    """The decimal128 result type of `add`/`sub`/`mul` over two decimal128 operands.

    Returns ``None`` (fall back) unless *both* operands are decimal128 and `op` is one of
    the three whose result precision/scale the engine derives deterministically. A decimal
    mixed with an int/float, ``mod``, or ``div`` is left uncertain on purpose.
    """
    if op not in ("add", "sub", "mul"):
        return None
    if not (pa.types.is_decimal128(left) and pa.types.is_decimal128(right)):
        return None
    p1, s1, p2, s2 = left.precision, left.scale, right.precision, right.scale
    if op == "mul":
        scale = s1 + s2
        precision = p1 + p2 + 1
    else:  # add / sub
        scale = max(s1, s2)
        precision = max(p1 - s1, p2 - s2) + scale + 1
    precision = min(precision, _DECIMAL_MAX_PRECISION)
    if scale > precision:  # cannot be represented -> stay uncertain, don't guess
        return None
    return pa.decimal128(precision, scale)


def _arith_type(op: str, left: pa.DataType, right: pa.DataType) -> pa.DataType | None:
    """The result of `add`/`sub`/`mul`/`mod` over two resolved operand types."""
    if op == "sub" and (temporal := _temporal_sub(left, right)) is not None:
        return temporal
    if (mixed := _numeric_with_bool(left, right)) is not None:
        return mixed
    if op in ("add", "sub") and (shifted := _date_shift(op, left, right)) is not None:
        return shifted
    if (dec := _decimal_arith_type(op, left, right)) is not None:
        return dec
    # `promote` answers "what type holds both of these" -- the right question for a
    # union, a coalesce, or a comparison, and the WRONG one for arithmetic. A
    # `decimal(10,2)` and an `int64` *union* to `decimal(21,2)` (wide enough for
    # either), but they *add* to `decimal(11,2)` (one carry digit past the widest
    # operand). Letting the lattice answer here made `Dataset.schema` advertise a type
    # the engine does not produce, which is the failure this whole module exists to
    # prevent. So the fallback is confined to the operands the lattice's answer and the
    # arithmetic result agree on -- the numeric ones -- and everything else stays silent.
    if not all(_arith_promotable(t) for t in (left, right)):
        return None
    common = promote(left, right)
    return widen(common) if common is not None else None


def binary_type(expr: object, schema: SchemaRef, infer: InferFn) -> pa.DataType | None:
    """The result type of a `Binary` node, or ``None`` if not certain.

    The name-only arms answer first so a comparison never descends into its operands.
    """
    op = expr.op  # type: ignore[attr-defined]
    if op in _BINARY_BOOL:
        return pa.bool_()
    if op in _BINARY_INT:
        return pa.int64()
    if op == "concat":
        # `||` renders both operands as text whatever their types (the engine stringifies
        # a number, a date and a list alike), so the result is String unconditionally.
        return pa.string()
    if op not in _BINARY_ARITH and op not in ("div", "floor_div"):
        return None

    left = infer(expr.left, schema)  # type: ignore[attr-defined]
    right = infer(expr.right, schema)  # type: ignore[attr-defined]
    if left is None or right is None:
        return None
    if op in _BINARY_ARITH:
        return _arith_type(op, left, right)
    # `div` coerces a Boolean operand the way the other arithmetic does; `floor_div` does
    # not, which is why only one of the two consults `_numeric_with_bool`.
    if op == "div" and (mixed := _numeric_with_bool(left, right)) is not None:
        return mixed
    # Neither is unconditionally double. `bc_expr` routes Int64 / Int64 through
    # `int_div_or_mod`, which divides in integers and stays Int64 (DuckDB's truncating
    # integer division, with a zero divisor nulled rather than trapping). Claiming
    # Float64 here was unsound: `Dataset.schema` said double where the engine returns
    # int64, and `kyber`'s `drop_cast_to_inferred_type` deletes a cast it believes is a
    # no-op on the strength of that answer. The public `/` operator casts its left side
    # to Float64 first, so it is unaffected either way -- but SQL and hand-built IR reach
    # this arm directly.
    return _int_preserving_div(left, right)
