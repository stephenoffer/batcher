"""`abs` and `sign` inside a comparison, restated over the bare column.

Both functions destroy sargability while carrying almost no information: `abs(x) < 5` is
the interval `-5 < x < 5`, and `sign(x) = 1` is `x > 0`. Written the second way the
zonemap pruner can refute a whole row group from its min/max pair and the source can push
the predicate into the scan; written the first way neither can do anything.

Three engine facts bound what is exact here, and all three were measured rather than
assumed:

* **Integer `abs` saturates.** `abs(INT64_MIN)` answers `INT64_MAX`, not `INT64_MIN` and
  not an error. That single value is where the interval restatement can disagree with the
  original, and only when the compared literal is itself `INT64_MAX` — so every `abs` rule
  requires a literal strictly inside `(0, INT64_MAX)`. Inside that band the two forms
  agree on every i64, `INT64_MIN` included.
* **Float comparison uses Arrow's total order, not IEEE.** A NaN sorts *above* every
  finite value and equals itself, so `NaN > 0` is `true`, `NaN <= 0` is `false`, and
  `NaN = NaN` is `true`. The `abs` interval rules survive that intact — `abs(NaN)` is NaN,
  and the interval and the magnitude comparison classify it identically on both sides —
  which is why they need no type guard.
* **`sign` answers `0.0` for a NaN.** It does not propagate the NaN, and it does not
  report it as the largest value either. Combined with the point above, that makes *every*
  `sign` comparison disagree with its zero-test restatement on a NaN row: `sign(NaN) = 1`
  is `false` while `NaN > 0` is `true`. So the entire `sign` family is restricted to an
  integer argument, where no NaN exists — not merely the comparisons that straddle zero.

Null behavior is identical on both sides of every rule in this module: both functions are
null-strict, and every rewritten form is built from comparisons and Kleene connectives
over the same operand, so a null input yields a null answer either way.
"""

from __future__ import annotations

from collections.abc import Callable

from batcher.kyber.rules.exprs.guards import is_integer, register_schema_leaf_rule
from batcher.kyber.rules.leaf_rewrite import register_leaf_rule
from batcher.plan.expr_ir import Binary, Expr, Lit
from batcher.plan.expr_ir.core import MathExpr
from batcher.plan.ir_tags import COMPARISON_FLIP
from batcher.plan.schema import SchemaRef

__all__ = ["ABS_RANGE_RULES", "SIGN_INTEGER_RULES"]

_INT64_MAX = 2**63 - 1

#: Comparisons whose operands may be swapped by flipping the operator, so each rule below
#: matches `abs(x) OP c` and `c OP' abs(x)` alike without a second table.


def _number(expr: Expr) -> float | int | None:
    """The numeric value of a non-boolean numeric literal, else ``None``."""
    if isinstance(expr, Lit) and isinstance(expr.value, (int, float)):
        if isinstance(expr.value, bool):
            return None
        return expr.value
    return None


def _call_against_literal(expr: Expr, fn: str) -> tuple[str, Expr, float | int] | None:
    """Decompose `expr` into `(op, argument, literal)` for a `fn(argument) OP literal`.

    Accepts the literal on either side, normalizing to the column-on-the-left spelling by
    flipping the operator. Returns ``None`` when `expr` is not a comparison between a call
    to `fn` and a numeric literal.
    """
    if not isinstance(expr, Binary) or expr.op not in COMPARISON_FLIP:
        return None
    for call, other, op in (
        (expr.left, expr.right, expr.op),
        (expr.right, expr.left, COMPARISON_FLIP[expr.op]),
    ):
        if isinstance(call, MathExpr) and call.fn == fn:
            value = _number(other)
            if value is not None:
                return op, call.input, value
    return None


# --- abs: a comparison against a magnitude is an interval ----------------------


def _abs_interval(op: str, arg: Expr, bound: float | int) -> Expr | None:
    """The interval form of `abs(arg) OP bound`, or ``None`` when `op` has none."""
    lower, upper = Lit(-bound), Lit(bound)
    if op == "lt":
        return Binary("and", Binary("gt", arg, lower), Binary("lt", arg, upper))
    if op == "le":
        return Binary("and", Binary("ge", arg, lower), Binary("le", arg, upper))
    if op == "gt":
        return Binary("or", Binary("lt", arg, lower), Binary("gt", arg, upper))
    if op == "ge":
        return Binary("or", Binary("le", arg, lower), Binary("ge", arg, upper))
    if op == "eq":
        return Binary("or", Binary("eq", arg, upper), Binary("eq", arg, lower))
    if op == "ne":
        return Binary("and", Binary("ne", arg, upper), Binary("ne", arg, lower))
    return None


def _abs_leaf(op_wanted: str) -> Callable[[Expr], Expr]:
    def leaf(expr: Expr) -> Expr:
        found = _call_against_literal(expr, "abs")
        if found is None:
            return expr
        op, arg, bound = found
        if op != op_wanted or not 0 < bound < _INT64_MAX:
            return expr
        rewritten = _abs_interval(op, arg, bound)
        return expr if rewritten is None else rewritten

    return leaf


#: The mirror of each comparison, for the operator index. A leaf built for `op` is also
#: reached by a node carrying `COMPARISON_FLIP[op]`, because the comparison is normalized with the
#: computed side on the left: `5 > abs(x)` arrives as a `gt` node and matches the `lt` leaf.


def _register(name: str, leaf: Callable[[Expr], Expr], op: str):
    # `op` and its mirror: the leaf normalizes the computed side to the left, so a `lt` leaf
    # is reached by a `gt` node with the literal on the left.
    return register_leaf_rule(
        name, leaf, expr_matches=(Binary,), expr_ops=(op, COMPARISON_FLIP[op])
    )


#: `abs(x) OP c` restated as an interval (or its complement) on `x`, one rule per
#: comparison operator:
#:
#:   * `abs(x) <  c` -> `x > -c AND x <  c`      * `abs(x) >  c` -> `x < -c OR x >  c`
#:   * `abs(x) <= c` -> `x >= -c AND x <= c`     * `abs(x) >= c` -> `x <= -c OR x >= c`
#:   * `abs(x) =  c` -> `x = c OR x = -c`        * `abs(x) <> c` -> `x <> c AND x <> -c`
#:
#: Each holds for every value of `x` — integer, float, or decimal — including NaN and the
#: infinities (both sides answer `false` for a NaN, since every comparison against one
#: does) and `-0.0` (whose magnitude is `0.0`, and which equals `0.0`). The literal band
#: `0 < c < INT64_MAX` is what excludes the one integer where saturating `abs` disagrees.
ABS_RANGE_RULES = [
    _register(f"abs_{op}_to_range", _abs_leaf(op), op)
    for op in ("lt", "le", "gt", "ge", "eq", "ne")
]


# --- sign: a comparison against -1/0/1 is a comparison against zero -----------

#: `(operator, literal) -> operator against zero` for every `sign` comparison that has a
#: zero-test restatement. **Integer arguments only**, and the reason is subtler than it
#: looks: the engine's float comparisons use Arrow's *total* order, in which a NaN sorts
#: above every finite value, so `NaN > 0` is `true` and `NaN <= 0` is `false`. But the
#: engine's `sign` answers `0.0` for a NaN. The two disagree on every NaN row for all
#: twelve comparisons, not only the ones that straddle zero -- `sign(NaN) = 1` is `false`
#: while `NaN > 0` is `true`. An integer column has no NaN, which is what makes the whole
#: table exact there and nowhere else.
_SIGN_INTEGER: dict[tuple[str, int], str] = {
    ("eq", 1): "gt",
    ("eq", -1): "lt",
    ("gt", 0): "gt",
    ("lt", 0): "lt",
    ("ge", 1): "gt",
    ("le", -1): "lt",
    ("eq", 0): "eq",
    ("ne", 0): "ne",
    ("ge", 0): "ge",
    ("le", 0): "le",
    ("gt", -1): "ge",
    ("lt", 1): "le",
}


def _sign_key(expr: Expr) -> tuple[tuple[str, int], Expr] | None:
    """`((op, literal), argument)` for a `sign(argument) OP literal` against -1/0/1."""
    found = _call_against_literal(expr, "sign")
    if found is None:
        return None
    op, arg, value = found
    if value not in (-1, 0, 1) or value != int(value):
        return None
    return (op, int(value)), arg


def _sign_integer_leaf(key: tuple[str, int]) -> Callable[[Expr, SchemaRef | None], Expr]:
    def leaf(expr: Expr, schema: SchemaRef | None) -> Expr:
        found = _sign_key(expr)
        if found is None or found[0] != key or not is_integer(found[1], schema):
            return expr
        return Binary(_SIGN_INTEGER[key], found[1], Lit(0))

    return leaf


_SIGN_NAMES = {
    ("eq", 1): "sign_eq_one_to_positive",
    ("eq", -1): "sign_eq_minus_one_to_negative",
    ("gt", 0): "sign_gt_zero_to_positive",
    ("lt", 0): "sign_lt_zero_to_negative",
    ("ge", 1): "sign_ge_one_to_positive",
    ("le", -1): "sign_le_minus_one_to_negative",
    ("eq", 0): "sign_eq_zero_to_zero_integer",
    ("ne", 0): "sign_ne_zero_to_nonzero_integer",
    ("ge", 0): "sign_ge_zero_to_nonnegative_integer",
    ("le", 0): "sign_le_zero_to_nonpositive_integer",
    ("gt", -1): "sign_gt_minus_one_to_nonnegative_integer",
    ("lt", 1): "sign_lt_one_to_nonpositive_integer",
}


def _register_sign_integer(key: tuple[str, int]):
    return register_schema_leaf_rule(
        _SIGN_NAMES[key],
        _sign_integer_leaf(key),
        expr_matches=(Binary,),
        expr_ops=(key[0], COMPARISON_FLIP[key[0]]),
    )


#: `sign(i) OP c` -> `i OP' 0` over an integer column, for all twelve comparisons that
#: have a zero-test restatement: `= 1`, `>= 1` and `> 0` become `i > 0`; `= -1`, `<= -1`
#: and `< 0` become `i < 0`; `= 0`, `<> 0`, `>= 0`, `<= 0`, `> -1` and `< 1` map to the
#: matching comparison against zero.
#:
#: On a float column every one of these would reclassify NaN, because the engine reports
#: `sign(NaN)` as `0.0` while comparing the NaN itself puts it above every finite value.
#: The `is_integer` guard is the whole soundness argument, so these rules decline when the
#: schema is unknown rather than assuming an integer.
SIGN_INTEGER_RULES = [_register_sign_integer(key) for key in _SIGN_INTEGER]
