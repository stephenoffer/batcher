"""Comparisons against a rounded, bucketed, or popcounted value, restated as a range.

`floor`, `ceil` and integer `//` are the three ways a query buckets a column, and a
predicate on the bucket is the shape a dashboard filter or a partition-style lookup takes:
`floor(price) = 42`, `ts // 3600 >= 12345`. Every one of them has an exact half-open
interval on the raw column, which is what the zonemap pruner and source pushdown need.

The arithmetic is over the integers even when the column is not, which is what makes
these exact where the corresponding transform through a multiplication or a true division
would not be. `floor(x) <= k` is `x < k + 1` for an integer `k` and *every* real `x`, and
`k + 1` is exact in both i64 and f64. The rules therefore require an integral literal and
check that the shifted bound is still representable, and decline otherwise.

Three engine-specific guards carry the rest of the soundness argument:

* The engine compares floats in Arrow's *total* order, where a NaN sorts above every
  finite value rather than answering `false` to everything. That does not disturb these
  rules: `floor(NaN)` and `ceil(NaN)` are themselves NaN, so the rounded value and the raw
  value sit at the same place in that order and both forms classify the row identically.
  The infinities behave the same way. No float edge case needs a separate rule.
* `bit_count` returns *null* for a NaN or an infinity rather than a count, so it is
  null-strict only on an integer. Its rules carry an `is_integer` guard.
* `//` is floored division, not truncated, so `x // k` is monotone in `x` for a positive
  `k` and the interval argument holds for negative `x` unchanged. A negative or zero
  divisor is not rewritten: zero aborts the query, and a negative one reverses the
  inequality while also making `INT64_MIN // -1` overflow.
"""

from __future__ import annotations

from collections.abc import Callable

from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, node_rule
from batcher.kyber.rules.exprs.guards import is_integer, schema_rule
from batcher.kyber.rules.leaf_rewrite import rewrite_node
from batcher.plan.expr_ir import Binary, Expr, Lit
from batcher.plan.expr_ir.core import MathExpr
from batcher.plan.logical import Aggregate, Filter, Project, Sort, Window
from batcher.plan.schema import SchemaRef

__all__ = [
    "BIT_COUNT_RULES",
    "CEIL_RANGE_RULES",
    "EVEN_RANGE_RULES",
    "FLOOR_DIV_RANGE_RULES",
    "FLOOR_RANGE_RULES",
    "RINT_RANGE_RULES",
    "ROUND_RANGE_RULES",
    "TRUNC_RANGE_RULES",
]

_NODES = (Filter, Project, Aggregate, Sort, Window)
_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1
_FLIP = {"eq": "eq", "ne": "ne", "lt": "gt", "gt": "lt", "le": "ge", "ge": "le"}
_COMPARISONS = ("lt", "le", "gt", "ge", "eq", "ne")


def _integral(expr: Expr) -> int | None:
    """The integer value of an integral numeric literal (`5` or `5.0`), else ``None``.

    A fractional literal answers ``None``: `floor(x) = 2.5` is unsatisfiable rather than an
    interval, and turning an unsatisfiable predicate into `false` is a different rule with
    a different null argument.
    """
    if not isinstance(expr, Lit) or isinstance(expr.value, bool):
        return None
    value = expr.value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer() and abs(value) < 2**53:
        return int(value)
    return None


def _in_int64(value: int) -> bool:
    return _INT64_MIN <= value <= _INT64_MAX


def _comparison_parts(expr: Expr) -> tuple[str, Expr, int] | None:
    """`(op, left_operand, integral_literal)` for a comparison against an integral literal,
    normalized so the computed side is on the left."""
    if not isinstance(expr, Binary) or expr.op not in _FLIP:
        return None
    for computed, other, op in (
        (expr.left, expr.right, expr.op),
        (expr.right, expr.left, _FLIP[expr.op]),
    ):
        value = _integral(other)
        if value is not None:
            return op, computed, value
    return None


def _register(name: str, leaf: Callable[[Expr], Expr], op: str):
    # The leaf fires on `op` *and* on its mirror, because `_comparison_parts` normalizes the
    # computed side to the left: `5 > floor(x)` reaches a `lt` leaf as a `gt` node. Declaring
    # both is what keeps the operator index a pure filter rather than a behavior change.
    return DEFAULT_REGISTRY.add(
        node_rule(
            name,
            Phase.NORMALIZE,
            lambda node, _ctx, _leaf=leaf: rewrite_node(node, _leaf),
            matches=_NODES,
            expr_fn=leaf,
            expr_matches=(Binary,),
            expr_ops=(op, _FLIP[op]),
        )
    )


#: The negation of each half-bound operator, used to complement an interval for `<>`.
_NEGATE = {"ge": "lt", "lt": "ge", "gt": "le", "le": "gt"}


def _band(arg: Expr, op: str, low_op: str, low: float, high_op: str, high: float) -> Expr | None:
    """Assemble the interval form for `=` or its complement for `<>`.

    `=` is the conjunction of the two half-bounds. `<>` is their De Morgan complement:
    *both* operators flip, which is what makes it `x < low OR x >= high` rather than the
    same two comparisons re-joined by an `OR`.
    """
    if op == "eq":
        return Binary("and", Binary(low_op, arg, Lit(low)), Binary(high_op, arg, Lit(high)))
    if op == "ne":
        return Binary(
            "or",
            Binary(_NEGATE[low_op], arg, Lit(low)),
            Binary(_NEGATE[high_op], arg, Lit(high)),
        )
    return None


# --- floor -------------------------------------------------------------------


def _floor_form(op: str, arg: Expr, k: int) -> Expr | None:
    """`floor(arg) OP k` over the bare `arg`.

    `floor(x) >= k <=> x >= k` and `floor(x) < k <=> x < k` need no shift; the other two
    half-bounds move by one, since `floor(x) > k` is `floor(x) >= k+1`.
    """
    if not _in_int64(k + 1):
        return None
    if op == "lt":
        return Binary("lt", arg, Lit(k))
    if op == "le":
        return Binary("lt", arg, Lit(k + 1))
    if op == "gt":
        return Binary("ge", arg, Lit(k + 1))
    if op == "ge":
        return Binary("ge", arg, Lit(k))
    return _band(arg, op, "ge", k, "lt", k + 1)


def _ceil_form(op: str, arg: Expr, k: int) -> Expr | None:
    """`ceil(arg) OP k` over the bare `arg` — the mirror of `_floor_form`, with the
    half-open interval closed at the top instead of the bottom."""
    if not _in_int64(k - 1):
        return None
    if op == "lt":
        return Binary("le", arg, Lit(k - 1))
    if op == "le":
        return Binary("le", arg, Lit(k))
    if op == "gt":
        return Binary("gt", arg, Lit(k))
    if op == "ge":
        return Binary("gt", arg, Lit(k - 1))
    return _band(arg, op, "gt", k - 1, "le", k)


def _trunc_lower(k: int) -> tuple[str, int]:
    """The half-bound meaning `trunc(x) >= k`.

    Truncation rounds *toward zero*, so its interval is not a fixed shift of `k` the way
    `floor`'s and `ceil`'s are. For a positive bucket it behaves like `floor`
    (`trunc(x) >= 1` is `x >= 1`); at and below zero it behaves like `ceil`
    (`trunc(x) >= 0` is `x > -1`, since `trunc(-0.5)` is `-0.0`).
    """
    return ("ge", k) if k >= 1 else ("gt", k - 1)


def _trunc_upper(k: int) -> tuple[str, int]:
    """The half-bound meaning `trunc(x) <= k` — the mirror of `_trunc_lower`.

    `trunc(x) <= 0` is `x < 1` and `trunc(x) <= -1` is `x <= -1`.
    """
    return ("lt", k + 1) if k >= 0 else ("le", k)


def _trunc_form(op: str, arg: Expr, k: int) -> Expr | None:
    """`trunc(arg) OP k` over the bare `arg`, split at zero."""
    if not (_in_int64(k + 1) and _in_int64(k - 1)):
        return None
    if op == "ge":
        low_op, low = _trunc_lower(k)
        return Binary(low_op, arg, Lit(low))
    if op == "gt":
        low_op, low = _trunc_lower(k + 1)
        return Binary(low_op, arg, Lit(low))
    if op == "le":
        high_op, high = _trunc_upper(k)
        return Binary(high_op, arg, Lit(high))
    if op == "lt":
        high_op, high = _trunc_upper(k - 1)
        return Binary(high_op, arg, Lit(high))
    low_op, low = _trunc_lower(k)
    high_op, high = _trunc_upper(k)
    return _band(arg, op, low_op, low, high_op, high)


def _round_lower(k: int) -> tuple[str, float]:
    """The half-bound meaning `round(x) >= k`, for half-**away-from-zero** rounding.

    The engine rounds `2.5` to `3` and `-2.5` to `-3` (measured), so the tie at `k - 0.5`
    belongs to `k` above zero and to `k - 1` at or below it.
    """
    return ("ge", k - 0.5) if k >= 1 else ("gt", k - 0.5)


def _round_upper(k: int) -> tuple[str, float]:
    """The half-bound meaning `round(x) <= k` — the mirror of `_round_lower`."""
    return ("lt", k + 0.5) if k >= 0 else ("le", k + 0.5)


def _rint_lower(k: int) -> tuple[str, float]:
    """The half-bound meaning `rint(x) >= k`, for half-**to-even** rounding.

    `rint` sends a tie to the nearer even integer (measured: `0.5 -> 0`, `1.5 -> 2`,
    `2.5 -> 2`), so whether the endpoint `k - 0.5` belongs to `k` depends on `k`'s parity —
    which is the whole difference between this family and `round`'s.
    """
    return ("ge", k - 0.5) if k % 2 == 0 else ("gt", k - 0.5)


def _rint_upper(k: int) -> tuple[str, float]:
    """The half-bound meaning `rint(x) <= k` — the mirror of `_rint_lower`."""
    return ("le", k + 0.5) if k % 2 == 0 else ("lt", k + 0.5)


def _half_form(op: str, arg: Expr, k: int, lower, upper) -> Expr | None:
    """`round(arg) OP k` / `rint(arg) OP k` over the bare `arg`.

    `lower` and `upper` supply the family's half-bounds; the operator dispatch is shared,
    since both families place their bucket at `k ± 0.5` and differ only in which endpoint
    the tie belongs to.
    """
    if not (_in_int64(k + 1) and _in_int64(k - 1)):
        return None
    if op == "ge":
        low_op, low = lower(k)
        return Binary(low_op, arg, Lit(low))
    if op == "gt":
        low_op, low = lower(k + 1)
        return Binary(low_op, arg, Lit(low))
    if op == "le":
        high_op, high = upper(k)
        return Binary(high_op, arg, Lit(high))
    if op == "lt":
        high_op, high = upper(k - 1)
        return Binary(high_op, arg, Lit(high))
    low_op, low = lower(k)
    high_op, high = upper(k)
    return _band(arg, op, low_op, low, high_op, high)


def _round_form(op: str, arg: Expr, k: int) -> Expr | None:
    return _half_form(op, arg, k, _round_lower, _round_upper)


def _rint_form(op: str, arg: Expr, k: int) -> Expr | None:
    return _half_form(op, arg, k, _rint_lower, _rint_upper)


def _even_lower(k: int) -> tuple[str, int]:
    """The half-bound meaning `even(x) >= k`, for an **even** `k`.

    `even` rounds away from zero to the next even integer, so its buckets are two units
    wide and the boundary belongs to whichever side is nearer zero: `even(x) >= 2` is
    `x > 0` (because `even(0)` is `0`), while `even(x) >= 0` is `x >= 0`.
    """
    return ("gt", k - 2) if k > 0 else ("ge", k)


def _even_upper(k: int) -> tuple[str, int]:
    """The half-bound meaning `even(x) <= k` — the mirror of `_even_lower` below zero."""
    return ("lt", k + 2) if k < 0 else ("le", k)


def _even_form(op: str, arg: Expr, k: int) -> Expr | None:
    """`even(arg) OP k` over the bare `arg`, for an even bucket.

    An odd `k` names an empty bucket rather than an interval — `even(x) = 3` holds for no
    `x` at all — and turning an unsatisfiable predicate into `false` would reclassify a
    null row, so an odd bucket declines instead.
    """
    if k % 2 != 0 or not (_in_int64(k + 2) and _in_int64(k - 2)):
        return None
    if op == "ge":
        low_op, low = _even_lower(k)
        return Binary(low_op, arg, Lit(low))
    if op == "gt":
        low_op, low = _even_lower(k + 2)
        return Binary(low_op, arg, Lit(low))
    if op == "le":
        high_op, high = _even_upper(k)
        return Binary(high_op, arg, Lit(high))
    if op == "lt":
        high_op, high = _even_upper(k - 2)
        return Binary(high_op, arg, Lit(high))
    low_op, low = _even_lower(k)
    high_op, high = _even_upper(k)
    return _band(arg, op, low_op, low, high_op, high)


def _rounding_leaf(fn: str, op_wanted: str, form) -> Callable[[Expr], Expr]:
    def leaf(expr: Expr) -> Expr:
        parts = _comparison_parts(expr)
        if parts is None:
            return expr
        op, computed, k = parts
        if op != op_wanted or not isinstance(computed, MathExpr) or computed.fn != fn:
            return expr
        rewritten = form(op, computed.input, k)
        return expr if rewritten is None else rewritten

    return leaf


#: `floor(x) OP k` as a half-open interval on `x`, one rule per operator:
#: `< k` -> `x < k`, `<= k` -> `x < k+1`, `> k` -> `x >= k+1`, `>= k` -> `x >= k`,
#: `= k` -> `k <= x < k+1`, `<> k` -> `x < k OR x >= k+1`.
FLOOR_RANGE_RULES = [
    _register(f"floor_{op}_to_range", _rounding_leaf("floor", op, _floor_form), op)
    for op in _COMPARISONS
]

#: `ceil(x) OP k` as a half-open interval on `x`, closed at the top:
#: `< k` -> `x <= k-1`, `<= k` -> `x <= k`, `> k` -> `x > k`, `>= k` -> `x > k-1`,
#: `= k` -> `k-1 < x <= k`, `<> k` -> `x <= k-1 OR x > k`.
CEIL_RANGE_RULES = [
    _register(f"ceil_{op}_to_range", _rounding_leaf("ceil", op, _ceil_form), op)
    for op in _COMPARISONS
]

#: `trunc(x) OP k` as an interval on `x`. Truncation rounds toward zero, so the interval
#: is `[k, k+1)` above zero, `(k-1, k]` below it, and `(-1, 1)` at zero — which is why this
#: family cannot reuse `floor`'s or `ceil`'s fixed shift and derives each half-bound from
#: the sign of the bucket instead.
TRUNC_RANGE_RULES = [
    _register(f"trunc_{op}_to_range", _rounding_leaf("trunc", op, _trunc_form), op)
    for op in _COMPARISONS
]

#: `round(x) OP k` as an interval on `x`, for the engine's half-**away-from-zero** rounding:
#: the bucket for `k` is `[k-0.5, k+0.5)` above zero, `(k-0.5, k+0.5]` below it, and
#: `(-0.5, 0.5)` at zero. Both endpoints are exact in f64 for every `k` the guard admits,
#: since `k ± 0.5` needs one more bit of mantissa than `k` itself.
ROUND_RANGE_RULES = [
    _register(f"round_{op}_to_range", _rounding_leaf("round", op, _round_form), op)
    for op in _COMPARISONS
]

#: `rint(x) OP k` as an interval on `x`, for half-**to-even** rounding. Identical to the
#: `round` family except at the endpoints, where the tie belongs to `k` when `k` is even and
#: to its neighbour when `k` is odd — so the bucket is closed on both sides for an even `k`
#: and open on both sides for an odd one.
RINT_RANGE_RULES = [
    _register(f"rint_{op}_to_range", _rounding_leaf("rint", op, _rint_form), op)
    for op in _COMPARISONS
]

#: `even(x) OP k` as an interval on `x`. `even` rounds away from zero to the next even
#: integer, so its buckets are two units wide, its boundary belongs to whichever side is
#: nearer zero, and an odd `k` names no interval at all — the rule declines rather than
#: folding an unsatisfiable predicate to `false`, which would reclassify a null row.
EVEN_RANGE_RULES = [
    _register(f"even_{op}_to_range", _rounding_leaf("even", op, _even_form), op)
    for op in _COMPARISONS
]


# --- bit_count ---------------------------------------------------------------

#: `(operator, literal) -> operator against zero`. A popcount is zero exactly when the
#: value is zero and positive otherwise, so these four comparisons carry no information
#: the raw column does not.
_BIT_COUNT_FORMS: dict[tuple[str, int], str] = {
    ("eq", 0): "eq",
    ("ne", 0): "ne",
    ("gt", 0): "ne",
    ("ge", 1): "ne",
}

_BIT_COUNT_NAMES = {
    ("eq", 0): "bit_count_eq_zero_to_zero",
    ("ne", 0): "bit_count_ne_zero_to_nonzero",
    ("gt", 0): "bit_count_gt_zero_to_nonzero",
    ("ge", 1): "bit_count_ge_one_to_nonzero",
}


def _bit_count_leaf(key: tuple[str, int]) -> Callable[[Expr, SchemaRef | None], Expr]:
    def leaf(expr: Expr, schema: SchemaRef | None) -> Expr:
        parts = _comparison_parts(expr)
        if parts is None:
            return expr
        op, computed, k = parts
        if (op, k) != key or not isinstance(computed, MathExpr) or computed.fn != "bit_count":
            return expr
        if not is_integer(computed.input, schema):
            return expr
        return Binary(_BIT_COUNT_FORMS[key], computed.input, Lit(0))

    return leaf


def _register_schema(name: str, leaf, op: str) -> object:
    # `op` and its mirror, for the same reason as `_register`: the comparison is normalized
    # with the computed side on the left before the leaf inspects the operator.
    return DEFAULT_REGISTRY.add(
        node_rule(
            name,
            Phase.NORMALIZE,
            lambda node, _ctx, _leaf=leaf: schema_rule(node, _leaf, carries=(Binary,)),
            matches=(Filter, Project),
            expr_ops=(op, _FLIP[op]),
            expr_schema_fn=leaf,
            expr_matches=(Binary,),
        )
    )


#: `bit_count(i) = 0` -> `i = 0`, and the three spellings of "the popcount is positive"
#: -> `i <> 0`. Restricted to an integer argument: on a float the engine answers *null*
#: for a NaN or an infinity, so the popcount comparison and the value comparison disagree
#: on exactly the rows a null test would find.
BIT_COUNT_RULES = [
    _register_schema(_BIT_COUNT_NAMES[key], _bit_count_leaf(key), key[0])
    for key in _BIT_COUNT_FORMS
]


# --- integer floored division ------------------------------------------------


def _floor_div_form(op: str, arg: Expr, k: int, divisor: int) -> Expr | None:
    """`arg // divisor OP k` over the bare `arg`, for a positive `divisor`.

    `x // d >= k` is `x >= k*d` and `x // d < k` is `x < k*d`; the strict/loose duals shift
    the bucket index by one before multiplying. Both products are overflow-checked, so a
    bucket at the edge of the i64 range declines instead of folding to a wrapped bound.
    """
    low, high = k * divisor, (k + 1) * divisor
    if not (_in_int64(low) and _in_int64(high)):
        return None
    if op == "lt":
        return Binary("lt", arg, Lit(low))
    if op == "le":
        return Binary("lt", arg, Lit(high))
    if op == "gt":
        return Binary("ge", arg, Lit(high))
    if op == "ge":
        return Binary("ge", arg, Lit(low))
    return _band(arg, op, "ge", low, "lt", high)


def _floor_div_leaf(op_wanted: str) -> Callable[[Expr, SchemaRef | None], Expr]:
    def leaf(expr: Expr, schema: SchemaRef | None) -> Expr:
        parts = _comparison_parts(expr)
        if parts is None:
            return expr
        op, computed, k = parts
        if op != op_wanted or not isinstance(computed, Binary) or computed.op != "floor_div":
            return expr
        divisor = _integral(computed.right)
        if divisor is None or divisor <= 0:
            return expr
        if not (is_integer(computed.left, schema) and is_integer(computed.right, schema)):
            return expr
        rewritten = _floor_div_form(op, computed.left, k, divisor)
        return expr if rewritten is None else rewritten

    return leaf


#: `i // d OP k` as an interval on `i` for a positive integer divisor `d`:
#: `>= k` -> `i >= k*d`, `> k` -> `i >= (k+1)*d`, `< k` -> `i < k*d`, `<= k` -> `i < (k+1)*d`,
#: `= k` -> `k*d <= i < (k+1)*d`, `<> k` -> the complement.
#:
#: This is the bucket-lookup shape — an hour bucket `ts // 3600`, a page `id // 1000` —
#: and turning it into a contiguous range on the raw column is what lets a sorted or
#: partitioned source skip everything outside the bucket. Both operands must be integers:
#: a float `//` divides before flooring, so its bucket boundaries are subject to the
#: division's rounding and the interval would not be exact.
FLOOR_DIV_RANGE_RULES = [
    _register_schema(f"floor_div_{op}_to_range", _floor_div_leaf(op), op) for op in _COMPARISONS
]
