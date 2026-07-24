"""Scalar SQL-compat sugar — the DuckDB/Spark spellings that are free functions, not `Expr` methods.

Most scalar work in Batcher is a method on `Expr`, because most of it reads better that way:
`col("x").ln()`, `col("x").abs()`. A handful of functions do not, either because they take two
columns symmetrically (`hypot`, `arctan2`, `gcd`, `lcm`, the two-argument `log`) or because the
familiar spelling from SQL is a call rather than a method (`iff`, `nanvl`, `cut`, `width_bucket`).
Those live here.

Nothing in this module introduces IR. Each function composes existing `Expr` nodes — a `when`
chain, a change-of-base division, an arithmetic tree — so the engine sees the same plan it would
have seen had you written the composition by hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import atan2
from batcher.plan.expr_ir.constructors import col, lit, when
from batcher.plan.expr_ir.core import Binary, Expr, IntoExpr, Math2Expr, MathExpr, _wrap

__all__ = [
    "arctan2",
    "cut",
    "gcd",
    "hypot",
    "iff",
    "lcm",
    "log",
    "nanvl",
    "width_bucket",
]


if TYPE_CHECKING:
    from collections.abc import Sequence


def iff(condition: Expr, if_true: IntoExpr, if_false: IntoExpr) -> Expr:
    """``if_true`` where `condition` is true, else ``if_false`` (DuckDB ``IF``/``IFF``).

    The two-branch shorthand for ``when(condition).then(if_true).otherwise(if_false)``.

    Args:
        condition: The boolean predicate selecting the branch per row.
        if_true: The value where ``condition`` is true.
        if_false: The value where ``condition`` is false or null.

    Returns:
        An expression yielding ``if_true`` or ``if_false`` per row.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [-1, 2]})
            >>> ds.select(s=bt.iff(bt.col("x") > 0, bt.lit("pos"), bt.lit("neg"))).to_pydict()
            {'s': ['neg', 'pos']}
    """
    return when(condition).then(_wrap(if_true)).otherwise(_wrap(if_false))


def nanvl(value: IntoExpr, fallback: IntoExpr) -> Expr:
    """`value` unless it is NaN, in which case `fallback` (Spark ``nanvl``).

    Distinct from `coalesce` — this replaces IEEE NaN, not NULL. A NULL `value`
    passes through unchanged (NULL is not NaN).

    Args:
        value: The value to return unless it is NaN.
        fallback: The replacement used where ``value`` is NaN.

    Returns:
        An expression yielding ``value``, or ``fallback`` where ``value`` is NaN.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, float("nan")]})
            >>> ds.select(r=bt.nanvl(bt.col("x"), bt.lit(0.0))).to_pydict()
            {'r': [1.0, 0.0]}
    """
    v = _wrap(value)
    return when(v.is_nan()).then(_wrap(fallback)).otherwise(v)


def cut(
    value: IntoExpr,
    breaks: Sequence[float],
    *,
    labels: Sequence[object] | None = None,
    right: bool = True,
) -> Expr:
    """Bin a numeric column into buckets defined by explicit edges.

    With ``n`` sorted break points the column is split into ``n + 1`` buckets: everything at or
    below the first break, each interval between consecutive breaks, and everything above the
    last. By default a value equal to a break falls into the lower bucket (``right=True``,
    left-open intervals ``(a, b]``), matching ``pandas.cut`` and the usual "up to and including"
    reading of a threshold. Set ``right=False`` for right-open intervals ``[a, b)``.

    The result is the integer bin index by default, or the matching entry from `labels` when
    given. It lowers to a `when`/`then` chain, so it is a pure per-row expression with no `fit`
    and no aggregate. Reach for `cut` when the edges are known up front and for
    `KBinsDiscretizer` when they must be learned from the data.

    Args:
        value: The numeric column (or expression) to bin.
        breaks: The sorted, strictly increasing interior edge values. ``n`` breaks yield
            ``n + 1`` buckets.
        labels: One label per bucket (so ``len(breaks) + 1`` of them) to return instead of the
            integer index. Omit to return the 0-based bin index.
        right: If true (default), intervals are left-open ``(a, b]`` and a value equal to a
            break goes to the lower bucket. If false, intervals are right-open ``[a, b)``.

    Returns:
        An expression giving each row's bucket, as an integer index or a `labels` entry.

    Raises:
        PlanError: If `breaks` is empty, or `labels` is given with the wrong length.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"age": [5, 18, 40, 70]})
            >>> ds.with_columns(band=bt.cut("age", [12, 19, 65])).to_pydict()["band"]
            [0, 1, 2, 3]

            >>> labeled = bt.cut("age", [12, 19, 65], labels=["child", "teen", "adult", "senior"])
            >>> ds.with_columns(band=labeled).to_pydict()["band"]
            ['child', 'teen', 'adult', 'senior']
    """
    if len(breaks) == 0:
        raise PlanError("cut needs at least one break point.")
    if labels is not None and len(labels) != len(breaks) + 1:
        raise PlanError(
            f"cut got {len(breaks)} breaks (so {len(breaks) + 1} buckets) but "
            f"{len(labels)} labels; pass one label per bucket."
        )
    column = col(value) if isinstance(value, str) else _wrap(value)
    outputs: list[object] = list(labels) if labels is not None else list(range(len(breaks) + 1))
    # Build the when/then chain from the last edge inward so the earliest (lowest) edge a value
    # falls under wins. A left-open interval (a, b] means "value <= b"; a right-open one [a, b)
    # means "value < b". The final bucket is the else branch.
    chain: Expr = _wrap(outputs[-1])
    for index in range(len(breaks) - 1, -1, -1):
        condition = column <= lit(breaks[index]) if right else column < lit(breaks[index])
        chain = when(condition).then(_wrap(outputs[index])).otherwise(chain)
    return chain


def arctan2(y: IntoExpr, x: IntoExpr) -> Math2Expr:
    """Two-argument arctangent — the NumPy/Polars ``arctan2`` spelling of :func:`atan2`.

    The angle in radians of the point ``(x, y)`` from the positive x-axis, using both
    signs to place it in the correct quadrant (unlike a plain ``arctan(y / x)``).

    Args:
        y: The ordinate (numerator).
        x: The abscissa (denominator).

    Returns:
        A Float64 expression of the angle in radians, in ``(-pi, pi]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1.0], "x": [1.0]})
            >>> ds.select(a=bt.arctan2(bt.col("y"), bt.col("x")).round(4)).to_pydict()
            {'a': [0.7854]}
    """
    return atan2(y, x)


def log(base: IntoExpr, value: IntoExpr) -> Expr:
    """Logarithm of `value` in the given `base` (→ Float64).

    Computed as ``ln(value) / ln(base)`` (change of base). For the common fixed
    bases use the methods ``.ln()``, ``.log10()``, or ``.log2()`` instead.

    Args:
        base: The logarithm base (column or literal).
        value: The value to take the logarithm of (column or literal).

    Returns:
        The logarithm of ``value`` in the given base.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [8.0]})
            >>> ds.select(bt.log(2, bt.col("x")).alias("r")).to_pydict()
            {'r': [3.0]}
    """
    return Binary("div", MathExpr("ln", _wrap(value)), MathExpr("ln", _wrap(base)))


def gcd(a: IntoExpr, b: IntoExpr) -> Math2Expr:
    """Greatest common divisor of two integers (DuckDB ``gcd``).

    Operates element-wise on integer columns or literals; ``gcd(0, n)`` is ``n``.
    The result is computed as a Float64.

    Args:
        a: First integer operand (column or literal).
        b: Second integer operand (column or literal).

    Returns:
        The greatest common divisor of the two operands.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [12, 15], "b": [18, 20]})
            >>> ds.select(bt.gcd(bt.col("a"), bt.col("b")).alias("r")).to_pydict()
            {'r': [6.0, 5.0]}
    """
    return Math2Expr("gcd", _wrap(a), _wrap(b))


def lcm(a: IntoExpr, b: IntoExpr) -> Math2Expr:
    """Least common multiple of two integers (DuckDB ``lcm``).

    Operates element-wise; ``lcm`` involving 0 is 0. The result is computed as a Float64.

    Args:
        a: First integer operand (column or literal).
        b: Second integer operand (column or literal).

    Returns:
        The least common multiple of the two operands.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [4, 6], "b": [6, 8]})
            >>> ds.select(bt.lcm(bt.col("a"), bt.col("b")).alias("r")).to_pydict()
            {'r': [12.0, 24.0]}
    """
    return Math2Expr("lcm", _wrap(a), _wrap(b))


def hypot(a: IntoExpr, b: IntoExpr) -> Math2Expr:
    """Euclidean norm ``sqrt(a² + b²)`` of two numbers (→ Float64; DuckDB ``hypot``).

    Computes the length of the hypotenuse element-wise, avoiding intermediate overflow.

    Args:
        a: First leg (column or literal).
        b: Second leg (column or literal).

    Returns:
        The Euclidean norm ``sqrt(a² + b²)`` of the two legs.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [3.0, 5.0], "b": [4.0, 12.0]})
            >>> ds.select(bt.hypot(bt.col("a"), bt.col("b")).alias("r")).to_pydict()
            {'r': [5.0, 13.0]}
    """
    return Math2Expr("hypot", _wrap(a), _wrap(b))


def width_bucket(value: IntoExpr, low: IntoExpr, high: IntoExpr, count: int) -> Expr:
    """Histogram bucket index (1..``count``) for a value over an equal-width range.

    Backs SQL ``width_bucket`` over the range ``[low, high]``: values below `low` fall
    in bucket 0 and values at or above `high` in bucket ``count + 1`` (the SQL
    out-of-range convention). Desugars to arithmetic + `clip`, so it needs no engine
    support.

    Args:
        value: The value to bucket (column or literal).
        low: Inclusive lower bound of the bucketed range.
        high: Exclusive upper bound of the bucketed range.
        count: Number of equal-width buckets between ``low`` and ``high``.

    Returns:
        The 1-based bucket index of ``value`` (0 or ``count + 1`` when out of range).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"v": [0.5, 5.0, -1.0, 11.0]})
            >>> ds.select(bt.width_bucket(bt.col("v"), 0, 10, 5).alias("r")).to_pydict()
            {'r': [1.0, 3.0, 0.0, 6.0]}
    """
    v, lo, hi = _wrap(value), _wrap(low), _wrap(high)
    # floor((value - low) / (high - low) * count) + 1, clamped to [0, count+1].
    # Cast the numerator to float so the division is true (not integer) division.
    numer = ((v - lo) * count).cast("float64")
    raw = MathExpr("floor", Binary("div", numer, (hi - lo))) + 1
    return raw.clip(0, count + 1)
