"""Reading a predicate through the arithmetic wrapped around its column.

`WHERE x + 1 < 100`, `WHERE -x > 0`, `WHERE price * 1.2 >= 60`, `WHERE id % 10 = 0` and
`WHERE abs(delta) < 3` all describe a region of one column's distribution, and the column's
statistics describe it exactly. None of them reached those statistics: the comparison's
operand was not a bare `Col`, so every one fell to the Selinger constant — `1/3` for a range
and `0.1` for an equality — however sharp the measured histogram was.

**Why this is estimation-only, and why that is the whole point.** `kyber.rules.extra.sargable`
performs the same inversions as a *rewrite*, and deliberately refuses the ordered comparisons:
the engine's integer arithmetic wraps (`add_wrapping`, bit-for-bit with the JIT), so
`x + 5 > 10` and `x > 5` genuinely disagree at `INT64_MAX` and rewriting one into the other
would change results. That argument does not apply here, because nothing here rewrites
anything. Being wrong about one row in `2^64` costs a slightly different row *estimate*,
which is what a selectivity model is for; the sargable pass keeps its guards because being
wrong there costs an answer.

So this module covers exactly the gap the rewrite must leave open, plus the two shapes that
are not invertible at all and so can never be rewritten: modulo, whose result is uniform on a
bounded domain regardless of the column, and absolute value, which folds the distribution
onto a symmetric interval.
"""

from __future__ import annotations

from typing import Any

from batcher.plan.expr_ir import Binary, Col, Expr, Lit, MathExpr
from batcher.plan.ir_tags import ORDERING_FLIP

__all__ = [
    "interval_containment",
    "invert_arithmetic",
    "modulo_domain",
    "symmetric_interval",
]

# The comparisons whose direction reverses when both sides are multiplied by a negative
# number, or when the column moves to the other side of the operator.
_FLIP = dict(ORDERING_FLIP) | {"eq": "eq", "ne": "ne"}


def _number(value: Any) -> float | None:
    """`value` as a float when it is a plain number (bool excluded), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _literal(expr: Expr) -> float | None:
    """The numeric value of a literal operand, else None."""
    return _number(expr.value) if isinstance(expr, Lit) else None


def _column_and_constant(inner: Binary) -> tuple[Expr, float, bool] | None:
    """`(column_side, constant, column_on_left)` for a binary with one literal operand."""
    right = _literal(inner.right)
    if right is not None and not isinstance(inner.left, Lit):
        return inner.left, right, True
    left = _literal(inner.left)
    if left is not None and not isinstance(inner.right, Lit):
        return inner.right, left, False
    return None


def invert_arithmetic(expr: Binary) -> Binary | None:
    """`f(col) OP lit` rewritten as the equivalent `col OP' lit'`, or None.

    `f` must be a strictly monotone function of one operand built from a constant: an
    addition or subtraction (either way round), or a multiplication or division by a non-zero
    constant. A negative multiplier reverses the comparison, and so does putting the column on
    the right of a subtraction — both are handled by flipping the operator, which is what
    makes `-x > 0` read as `x < 0` rather than as an unknown.

    The result is only ever *estimated* from, never executed, so the engine's wrapping integer
    arithmetic is not a correctness concern here (see the module docstring).

    One level is peeled per call, and the peeled form is itself a comparison — so a caller
    that re-estimates the result unwraps a nested expression a layer at a time, and
    `(x + 1) * 2 < 10` reaches `x < 4`. The recursion terminates because every peel strictly
    shrinks the expression.

    Args:
        expr: A comparison whose one side is a constant arithmetic expression.

    Returns:
        The equivalent bare-column comparison, or None when the shape does not qualify.
    """
    outer = _column_and_constant(expr)
    if outer is None:
        return None
    inner_side, literal, col_on_left = outer
    if not isinstance(inner_side, Binary):
        return None
    op = expr.op if col_on_left else _FLIP.get(expr.op, expr.op)
    match = _column_and_constant(inner_side)
    if match is None:
        return None
    column, constant, const_on_right = match
    folded = _fold(inner_side.op, literal, constant, const_on_right)
    if folded is None:
        return None
    value, reverses = folded
    return Binary(_FLIP.get(op, op) if reverses else op, column, Lit(value))


def _fold(
    op: str, literal: float, constant: float, const_on_right: bool
) -> tuple[float, bool] | None:
    """The literal moved across `op`, plus whether the comparison's direction reverses."""
    if op == "add":  # col + k  OP  lit   ->   col OP lit - k
        return literal - constant, False
    if op == "sub":
        if const_on_right:  # col - k  OP  lit   ->   col OP lit + k
            return literal + constant, False
        return constant - literal, True  # k - col  OP  lit   ->   col FLIP(OP) k - lit
    if op == "mul":  # col * k  OP  lit   ->   col OP lit / k, reversed when k < 0
        if constant == 0.0:
            return None  # `col * 0` is a constant, not a monotone function of the column
        return literal / constant, constant < 0.0
    if op == "div" and const_on_right:  # col / k  OP  lit   ->   col OP lit * k
        if constant == 0.0:
            return None
        return literal * constant, constant < 0.0
    return None


def modulo_domain(expr: Binary) -> tuple[float, Any, bool] | None:
    """The size of the domain a `col % k` comparison ranges over, else None.

    `x % k` takes `k` values whatever `x`'s distribution is, so `id % 10 = 0` keeps a tenth
    of the rows and `id % 10 < 3` keeps three tenths. Both are ordinary shapes — bucketing,
    deterministic sampling, sharding by key — and both took the flat cold-start constant,
    which put `id % 2 = 0` at a tenth of the table against a true half.

    The column's own statistics are deliberately not consulted: the residue is uniform on
    `[0, k)` for any input distribution that is not itself aligned to `k`, and that
    assumption is far better founded than the constant it replaces.

    Args:
        expr: A comparison whose one side is `col % k`.

    Returns:
        `(k, literal, modulo_on_left)` when the shape matches and `k` is an integer above 1,
        else None.
    """
    side = _column_and_constant(expr)
    if side is None:
        return None
    inner, literal, on_left = side
    if not (isinstance(inner, Binary) and inner.op == "mod" and isinstance(inner.right, Lit)):
        return None
    if not isinstance(inner.left, (Col, Binary, MathExpr)):
        return None
    divisor = inner.right.value
    if isinstance(divisor, bool) or not isinstance(divisor, int) or divisor <= 1:
        return None
    return float(divisor), literal, on_left


def symmetric_interval(expr: Binary) -> tuple[Expr, float, str] | None:
    """`abs(col) OP lit` as the pair of bounds it really is, else None.

    `|x| < v` is `-v < x < v` and `|x| > v` is its complement, so the predicate is a statement
    about two points of the column's CDF rather than an opaque function call. On a
    zero-centred distribution the difference is large: `abs(delta) < 1` over a standard normal
    keeps 68% of the rows, and the range constant priced it at 33%.

    A negative bound makes the answer certain rather than unknown — `|x| < -1` matches nothing
    and `|x| >= -1` matches everything — and those are returned through the normal CDF path by
    the caller rather than special-cased here.

    Args:
        expr: A comparison whose one side is `abs(col)`.

    Returns:
        `(column, bound, effective_op)` with the operator normalised so `abs(col)` is on the
        left, or None when the shape does not match.
    """
    side = _column_and_constant(expr)
    if side is None:
        return None
    inner, literal, col_on_left = side
    if not (isinstance(inner, MathExpr) and inner.fn == "abs"):
        return None
    op = expr.op if col_on_left else _FLIP.get(expr.op, expr.op)
    if op not in ORDERING_FLIP:
        return None
    return inner.input, literal, op


def interval_containment(conjuncts: list) -> tuple[str, str, str, list] | None:
    """`(probe, lower, upper, consumed)` for a `probe BETWEEN lower AND upper` over *columns*.

    The temporal-validity lookup: `WHERE ts >= valid_from AND ts < valid_to`. It is the whole
    of an SCD-2 point-in-time join (`ds.scd`), an IP-range or version-range lookup, and the
    payload of every range join — and it is the one conjunction where independence is not
    merely imprecise but structurally wrong, because `lower` and `upper` are the two ends of
    *one* interval and move together.

    Estimated as two independent comparisons, `t >= lo AND t <= hi` over a 100-wide interval
    inside a 1,000-wide domain came out at 0.247 of the join against 0.101 actual: both halves
    look like coin flips because each bound alone really does cut about half the rows. What
    decides the answer is the interval's *width*, which neither conjunct mentions.

    Returns None unless exactly this shape is present: two column-to-column ordering
    comparisons sharing one probe column, bounding it below and above by two *different*
    columns. Anything else keeps the ordinary per-conjunct estimate.

    Args:
        conjuncts: The conjuncts of one `AND`.

    Returns:
        The probe, lower-bound and upper-bound column names plus the conjuncts consumed, or
        None when the shape does not match.
    """
    constraints = []
    for conjunct in conjuncts:
        if not (isinstance(conjunct, Binary) and conjunct.op in ORDERING_FLIP):
            continue
        left, right = conjunct.left, conjunct.right
        if isinstance(left, Col) and isinstance(right, Col) and left.name != right.name:
            constraints.append((left.name, conjunct.op, right.name, conjunct))
    if len(constraints) < 2:
        return None
    for probe in {c[0] for c in constraints} | {c[2] for c in constraints}:
        lower = upper = None
        used: list = []
        for a, op, b, node in constraints:
            if a == probe and b != probe:
                bound, other = ("lower", b) if op in ("ge", "gt") else ("upper", b)
            elif b == probe and a != probe:
                bound, other = ("lower", a) if op in ("le", "lt") else ("upper", a)
            else:
                continue
            if bound == "lower" and lower is None:
                lower, _ = other, used.append(node)
            elif bound == "upper" and upper is None:
                upper, _ = other, used.append(node)
        if lower is not None and upper is not None and lower != upper:
            return probe, lower, upper, used
    return None
