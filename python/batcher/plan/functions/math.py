"""Math free functions that compose existing scalar ops.

`log(base, value)` is change-of-base over the engine's natural log — no new IR.
The fixed-base logarithms (`ln`/`log10`/`log2`) are `Expr` methods; this is the
general two-argument form (DuckDB/Spark ``log(base, x)``).
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Binary, Expr, IntoExpr, Math2Expr, MathExpr, _wrap


def log(base: IntoExpr, value: IntoExpr) -> Expr:
    """Logarithm of `value` in the given `base` (→ Float64).

    Computed as ``ln(value) / ln(base)`` (change of base). For the common fixed
    bases use the methods ``.ln()``, ``.log10()``, or ``.log2()`` instead.

    Args:
        base: The logarithm base (column or literal).
        value: The value to take the logarithm of (column or literal).

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
    """
    return Math2Expr("lcm", _wrap(a), _wrap(b))


def hypot(a: IntoExpr, b: IntoExpr) -> Math2Expr:
    """Euclidean norm ``sqrt(a² + b²)`` of two numbers (→ Float64; DuckDB ``hypot``).

    Computes the length of the hypotenuse element-wise, avoiding intermediate overflow.

    Args:
        a: First leg (column or literal).
        b: Second leg (column or literal).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [3.0, 5.0], "b": [4.0, 12.0]})
            >>> ds.select(bt.hypot(bt.col("a"), bt.col("b")).alias("r")).to_pydict()
            {'r': [5.0, 13.0]}
    """
    return Math2Expr("hypot", _wrap(a), _wrap(b))


def width_bucket(value: IntoExpr, low: IntoExpr, high: IntoExpr, count: int) -> Expr:
    """Histogram bucket index (1..`count`) for `value` over the equal-width range
    ``[low, high]`` (SQL ``width_bucket``).

    Values below `low` fall in bucket 0 and values at or above `high` in bucket
    ``count + 1`` (the SQL out-of-range convention). Desugars to arithmetic +
    `clip`, so it needs no engine support.

    Args:
        value: The value to bucket (column or literal).
        low: Inclusive lower bound of the bucketed range.
        high: Exclusive upper bound of the bucketed range.
        count: Number of equal-width buckets between ``low`` and ``high``.

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
