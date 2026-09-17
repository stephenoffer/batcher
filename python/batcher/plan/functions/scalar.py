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

import math
from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError, require_int
from batcher.plan.expr_ir.constructors import col, lit, nullif, when
from batcher.plan.expr_ir.core import (
    Binary,
    Expr,
    IntoExpr,
    Math2Expr,
    MathExpr,
    _col_or_expr,
    _wrap,
)

__all__ = [
    "arctan2",
    "bit_get",
    "cut",
    "e",
    "elt",
    "gcd",
    "great_circle_distance",
    "hypot",
    "iff",
    "lcm",
    "log",
    "nanvl",
    "next_after",
    "pi",
    "pmod",
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
    """Two-argument arctangent of ``y / x`` in radians, quadrant chosen by both signs.

    The angle in radians of the point ``(x, y)`` from the positive x-axis, using both
    signs to place it in the correct quadrant (unlike a plain ``arctan(y / x)``).

    A bare string names a **column**, as it does in Polars: ``arctan2("y", "x")``.

    Args:
        y: The ordinate (numerator), or its column name.
        x: The abscissa (denominator), or its column name.

    Returns:
        A Float64 expression of the angle in radians, in ``(-pi, pi]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1.0], "x": [1.0]})
            >>> ds.select(a=bt.arctan2(bt.col("y"), bt.col("x")).round(4)).to_pydict()
            {'a': [0.7854]}
    """
    return Math2Expr("atan2", _col_or_expr(y), _col_or_expr(x))


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
            {'r': [6, 5]}
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
            {'r': [12, 24]}
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


def next_after(value: IntoExpr, toward: IntoExpr) -> Math2Expr:
    """The next representable float after ``value`` in the direction of ``toward``.

    DuckDB ``nextafter``. One unit in the last place — the smallest step the type can
    take, which is what makes it the way to write a strict floating-point boundary
    (``x > next_after(limit, inf)``); ``limit + tiny`` cannot express it, because for a
    large ``limit`` there is no ``tiny`` that changes the value.

    Args:
        value: The starting value (column or literal).
        toward: The direction to step in (column or literal).

    Returns:
        The adjacent representable float, or ``toward`` when the two are already equal.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [1.0, 1.0]})
            >>> ds.select(bt.next_after(bt.col("a"), bt.lit(2.0)).alias("r")).to_pydict()
            {'r': [1.0000000000000002, 1.0000000000000002]}
    """
    return Math2Expr("next_after", _wrap(value), _wrap(toward))


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
    count = require_int(count, func="width_bucket", arg="count", minimum=1)
    v, lo, hi = _wrap(value), _wrap(low), _wrap(high)
    # floor((value - low) / (high - low) * count) + 1, clamped to [0, count+1].
    # Cast the numerator to float so the division is true (not integer) division.
    numer = ((v - lo) * count).cast("float64")
    raw = MathExpr("floor", Binary("div", numer, (hi - lo))) + 1
    return raw.clip(0, count + 1)


# Mean Earth radius (IUGG) in each supported unit. Great-circle distance is a radius
# times an angle, so the unit is a multiplier and nothing else changes.
_EARTH_RADIUS = {
    "km": 6371.0088,
    "m": 6_371_008.8,
    "mi": 3958.7613,
    "nm": 3440.0695,
}


def great_circle_distance(
    lat1: IntoExpr,
    lon1: IntoExpr,
    lat2: IntoExpr,
    lon2: IntoExpr,
    unit: str = "km",
) -> Expr:
    """Great-circle distance between two lat/lon points, in `unit` (kilometres by default).

    The inputs are degrees; the result is a length, not an angle. Daft's
    ``great_circle_distance`` answers in metres on a 6,371,000 m sphere, which is
    ``unit="m"`` scaled by ``6371000 / 6371008.8``.

    The haversine formula on a sphere of mean Earth radius. Haversine rather than the
    law of cosines because the latter loses precision for nearby points, where the
    cosine of a tiny angle is indistinguishable from 1 in double precision, and nearby
    points are the interesting case for a proximity filter.

    Composed from existing expression nodes, so the engine evaluates the same arithmetic
    tree you would have written by hand; there is no new IR and no per-row Python.

    Args:
        lat1: Latitude of the first point, in degrees.
        lon1: Longitude of the first point, in degrees.
        lat2: Latitude of the second point, in degrees.
        lon2: Longitude of the second point, in degrees.
        unit: Output unit: ``"km"``, ``"m"``, ``"mi"`` (statute miles), or ``"nm"``
            (nautical miles).

    Returns:
        The distance between the two points in `unit`, as a Float64 expression.

    Raises:
        PlanError: If `unit` is not one of the four recognized units.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {
            ...         "alat": [51.5074],
            ...         "alon": [-0.1278],
            ...         "blat": [48.8566],
            ...         "blon": [2.3522],
            ...     }
            ... )
            >>> out = ds.select(
            ...     km=bt.great_circle_distance(
            ...         bt.col("alat"), bt.col("alon"), bt.col("blat"), bt.col("blon")
            ...     )
            ... ).to_pydict()
            >>> round(out["km"][0])  # London to Paris
            344
    """
    if unit not in _EARTH_RADIUS:
        raise PlanError(
            f"great_circle_distance(): unit must be one of {sorted(_EARTH_RADIUS)}, got {unit!r}"
        )
    phi1 = _wrap(lat1).radians()
    phi2 = _wrap(lat2).radians()
    d_phi = (_wrap(lat2) - _wrap(lat1)).radians()
    d_lambda = (_wrap(lon2) - _wrap(lon1)).radians()
    # a = sin²(Δφ/2) + cos φ₁ · cos φ₂ · sin²(Δλ/2)
    sin_half_phi = (d_phi / lit(2.0)).sin()
    sin_half_lambda = (d_lambda / lit(2.0)).sin()
    a = sin_half_phi * sin_half_phi + phi1.cos() * phi2.cos() * sin_half_lambda * sin_half_lambda
    # c = 2·atan2(√a, √(1−a)) — the atan2 form rather than 2·asin(√a) because it stays
    # defined when rounding pushes `a` a hair above 1 for antipodal points, where `asin`
    # would produce NaN.
    central_angle = lit(2.0) * arctan2(a.sqrt(), (lit(1.0) - a).sqrt())
    return lit(_EARTH_RADIUS[unit]) * central_angle


def pi() -> Expr:
    """The constant π as a Float64 expression (Spark, Daft and DuckDB ``pi()``).

    Folds to a literal at plan-build time, so it costs nothing per row.

    Returns:
        A Float64 literal expression holding π.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"r": [1.0, 2.0]})
            >>> ds.select(area=bt.pi() * bt.col("r") * bt.col("r")).to_pydict()
            {'area': [3.141592653589793, 12.566370614359172]}
    """
    return lit(math.pi)


def e() -> Expr:
    """Euler's number *e* as a Float64 expression (Spark and Daft ``e()``).

    Folds to a literal at plan-build time, like :func:`pi`.

    Returns:
        A Float64 literal expression holding *e*.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.from_pydict({"x": [0]}).select(r=bt.e()).to_pydict()
            {'r': [2.718281828459045]}
    """
    return lit(math.e)


def pmod(dividend: IntoExpr, divisor: IntoExpr) -> Expr:
    """The positive modulus of `dividend` by `divisor` (Spark and Daft ``pmod``).

    Spark's definition: take the remainder ``r = dividend % divisor``, and where ``r`` is
    negative answer ``(r + divisor) % divisor`` instead. So ``pmod(-10, 3)`` is 2 where
    ``%`` gives -1. A negative *divisor* can still give a negative result, as in Spark:
    ``pmod(-5.0, -6.0)`` is -5.0 and ``pmod(7.0, -8.0)`` is 7.0. A zero divisor or a null
    operand gives null, and a NaN operand gives NaN.

    Args:
        dividend: The value to divide: a number, a column name, or an expression.
        divisor: The modulus: a number, a column name, or an expression.

    Returns:
        The positive remainder, in the operands' common numeric type.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [10, -10, 7], "b": [3, 3, 0]})
            >>> ds.select(r=bt.pmod(bt.col("a"), bt.col("b"))).to_pydict()
            {'r': [1, 2, None]}
    """
    right = _col_or_expr(divisor)
    remainder = _col_or_expr(dividend) % right
    return when(remainder < 0).then((remainder + right) % right).otherwise(remainder)


def bit_get(value: IntoExpr, position: IntoExpr) -> Expr:
    """The bit of an integer at `position`, counting from 0 at the least significant bit.

    Spark ``bit_get`` and ``getbit``. The answer is 0 or 1, and null when either operand is
    null. DuckDB's ``get_bit`` is a different function: it indexes a ``BIT`` string from
    the left, and Batcher has no ``BIT`` type.

    Args:
        value: The integer to read: a number, a column name, or an expression.
        position: The 0-based bit index: a number, a column name, or an expression.

    Returns:
        An Int64 expression holding 0 or 1.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"v": [1, 2, 3, None]})
            >>> ds.select(r=bt.bit_get("v", 1)).to_pydict()
            {'r': [0, 1, 1, None]}
    """
    shifted = _col_or_expr(value).bitwise_right_shift(_col_or_expr(position))
    return shifted.bitwise_and(lit(1))


def elt(index: IntoExpr, *values: IntoExpr) -> Expr:
    """The `index`-th of `values`, counting from 1, or null outside ``1..len(values)``.

    Spark ``elt``. A Python ``int`` index selects its candidate at plan-build time. A
    column index becomes a ``CASE`` over the positions, so the choice is made per row. A
    string index is a column name.

    Args:
        index: The 1-based position: an int, a column name, or an expression.
        *values: The candidates, all of one type (columns or literals).

    Returns:
        An expression of the candidates' type.

    Raises:
        PlanError: If no candidate value is given.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"n": [1, 2, 3], "a": ["scala"] * 3, "b": ["java"] * 3})
            >>> ds.select(r=bt.elt("n", bt.col("a"), bt.col("b"))).to_pydict()
            {'r': ['scala', 'java', None]}
    """
    if not values:
        raise PlanError("elt(): expected at least one value after the index")
    candidates = [_wrap(v) for v in values]
    # `nullif(x, x)` is a null of `x`'s type: a CASE branch has no untyped NULL spelling.
    missing = nullif(candidates[0], candidates[0])
    if isinstance(index, int) and not isinstance(index, bool):
        return candidates[index - 1] if 1 <= index <= len(candidates) else missing
    position = _col_or_expr(index)
    chain = when(position == 1).then(candidates[0])
    for number, value in enumerate(candidates[1:], start=2):
        chain = chain.when(position == number).then(value)
    return chain.otherwise(missing)
