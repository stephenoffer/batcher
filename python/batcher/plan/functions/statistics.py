"""Derived statistical aggregates built as expressions over mergeable primitives.

Each is a standard summary statistic Batcher's base aggregates don't name directly —
population variance (its `var`/`std` are the *sample* statistics), the geometric and
harmonic means, the root-mean-square, the coefficient of variation, the standard error
of the mean, the midrange. They are *expressions over aggregates*, so `group_by().agg()`
runs them as one mergeable aggregate pass plus a projection: identical single-node and
distributed, no new engine state.

Nulls are ignored per input aggregate, as in SQL. `geometric_mean` and `harmonic_mean`
are defined for positive inputs (a non-positive value makes ``ln`` / the reciprocal
undefined, as elsewhere).

The module also carries the `weighted_mean` and the positional selectors `first` / `last`
/ `arg_min` / `arg_max` — the top-level SQL-style spellings of the ordered/by-keyed
aggregates — since they share the same "reduce a group to one value" shape.
"""

from __future__ import annotations

from batcher.plan.expr_ir import count
from batcher.plan.expr_ir.core import AggExpr, Expr, IntoExpr, Lit
from batcher.plan.functions.aggregate import _as_column, covar_pop


def _n(x: Expr) -> Expr:
    """The non-null count of `x` as Float64 — the sample size the ratios divide by."""
    return x.count().cast("float64")


def weighted_mean(value: IntoExpr, weight: IntoExpr) -> Expr:
    """Weighted mean — ``sum(value * weight) / sum(weight)`` over rows where both are present.

    Each row contributes in proportion to its `weight`; with equal weights it reduces to
    the plain mean. Following SQL pairing, only rows where *both* `value` and `weight` are
    non-null count (a null in either drops the pair from both sums), and a group with no
    such rows is null.

    Args:
        value: The column (or expression) being averaged.
        weight: The per-row weight column (or expression).

    Returns:
        The weighted mean per group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"v": [1.0, 2.0, 3.0], "w": [1.0, 2.0, 3.0]})
            >>> ds.agg(m=bt.weighted_mean(bt.col("v"), bt.col("w")).round(4)).to_pydict()
            {'m': [2.3333]}
    """
    v, w = _as_column(value), _as_column(weight)
    # Null-propagating pair mask: ``a + b*0`` is ``a`` when both are present, else null.
    vp = v + w * Lit(0)
    wp = w + v * Lit(0)
    return (vp * wp).sum() / wp.sum()


def var_pop(column: str | Expr) -> Expr:
    """Population variance — the sum of squared deviations divided by ``n`` (SQL ``VAR_POP``).

    Batcher's :meth:`~batcher.Expr.var` is the *sample* variance (divides by ``n - 1``);
    this is the population form, which divides by ``n``.

    Built as ``covar_pop(x, x)`` — the co-moment of a column with itself *is* its
    population variance — rather than the algebraically equivalent
    ``var_samp * (n - 1) / n``. That rescaling is wrong at ``n == 1``: `var_samp` is
    NULL there (it divides by ``n - 1 == 0``), so the product is NULL, while the
    population variance of a single value is defined and equal to ``0``. Going through
    the `covar_pop` state, which divides the centered co-moment by ``n`` directly,
    returns ``0`` and matches DuckDB. It is the same primitive
    :func:`~batcher.plan.functions.regression.regr_slope` already reduces to.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        The population variance per group, or null for an empty group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 4.0, 8.0]})
            >>> ds.agg(v=bt.var_pop("x").round(4)).to_pydict()
            {'v': [7.1875]}

            >>> bt.from_pydict({"x": [5.0]}).agg(v=bt.var_pop("x")).to_pydict()
            {'v': [0.0]}
    """
    col = _as_column(column)
    return covar_pop(col, col)


def stddev_pop(column: str | Expr) -> Expr:
    """Population standard deviation — the square root of :func:`var_pop` (SQL ``STDDEV_POP``).

    Args:
        column: The column (or expression) to summarize.

    Returns:
        The population standard deviation per group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 4.0, 8.0]})
            >>> ds.agg(s=bt.stddev_pop("x").round(4)).to_pydict()
            {'s': [2.681]}
    """
    return var_pop(column).sqrt()


def geometric_mean(column: str | Expr) -> Expr:
    """Geometric mean — ``exp(mean(ln(x)))``, the ``n``-th root of the product of the values.

    The right average for rates and multiplicative growth. Defined for positive inputs.

    Args:
        column: The positive column (or expression) to summarize.

    Returns:
        The geometric mean per group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 4.0, 8.0]})
            >>> ds.agg(g=bt.geometric_mean("x").round(4)).to_pydict()
            {'g': [2.8284]}
    """
    return _as_column(column).ln().mean().exp()


def harmonic_mean(column: str | Expr) -> Expr:
    """Harmonic mean — ``n / sum(1/x)``, the reciprocal of the mean of the reciprocals.

    The right average for rates defined per unit (speeds, price-earnings ratios). Defined
    for positive inputs.

    Args:
        column: The positive column (or expression) to summarize.

    Returns:
        The harmonic mean per group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 4.0, 8.0]})
            >>> ds.agg(h=bt.harmonic_mean("x").round(4)).to_pydict()
            {'h': [2.1333]}
    """
    col = _as_column(column)
    return _n(col) / (Lit(1.0) / col).sum()


def rms(column: str | Expr) -> Expr:
    """Root mean square — ``sqrt(mean(x**2))``, the quadratic mean of the values.

    The magnitude average used for signals and errors (an RMS error weights large
    deviations more than a plain mean does).

    The column is widened to Float64 *before* squaring. Squaring in the input type
    overflows silently for Int64: ``x * x`` wraps, so a column of ``4e9`` — whose RMS
    is just ``4e9`` — produced a negative mean and a ``NaN`` root. The result is a
    real-valued statistic regardless of input type, so there is nothing to gain by
    squaring narrow, and a silently wrong answer to lose.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        The root mean square per group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 4.0, 8.0]})
            >>> ds.agg(r=bt.rms("x").round(4)).to_pydict()
            {'r': [4.6098]}

            >>> ds = bt.from_pydict({"x": [4_000_000_000, 4_000_000_000]})
            >>> ds.agg(r=bt.rms("x")).to_pydict()
            {'r': [4000000000.0]}
    """
    col = _as_column(column).cast("float64")
    return (col * col).mean().sqrt()


def cv(column: str | Expr) -> Expr:
    """Coefficient of variation — ``stddev / mean``, the relative (unitless) spread.

    Compares dispersion across series on different scales, where a raw standard deviation
    would not. Uses the sample standard deviation.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        The coefficient of variation per group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 4.0, 8.0]})
            >>> ds.agg(c=bt.cv("x").round(4)).to_pydict()
            {'c': [0.8255]}
    """
    col = _as_column(column)
    return col.std() / col.mean()


def sem(column: str | Expr) -> Expr:
    """Standard error of the mean — ``stddev / sqrt(n)``, the precision of the sample mean.

    How much the group's mean would wobble under resampling; the basis for a confidence
    interval on the mean. Uses the sample standard deviation.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        The standard error of the mean per group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 4.0, 8.0]})
            >>> ds.agg(e=bt.sem("x").round(4)).to_pydict()
            {'e': [1.5478]}
    """
    col = _as_column(column)
    return col.std() / _n(col).sqrt()


def midrange(column: str | Expr) -> Expr:
    """Midrange — ``(max + min) / 2``, the midpoint of the value range.

    A fast, outlier-sensitive center of location.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        The midrange per group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 4.0, 8.0]})
            >>> ds.agg(m=bt.midrange("x")).to_pydict()
            {'m': [4.5]}
    """
    col = _as_column(column)
    return (col.max() + col.min()) / Lit(2)


def first(column: str | Expr, order_by: IntoExpr) -> AggExpr:
    """The value of `column` at the first row in `order_by` order (SQL ``FIRST``).

    A partition-independent first: it picks the row that sorts first by `order_by`, so
    the result is identical single-node and distributed (an arrival-order first would
    not be). Ties break arbitrarily.

    Args:
        column: The column (or expression) whose value to return.
        order_by: The column (or expression) whose ascending order defines "first".

    Returns:
        An aggregate expression of the first value per group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a"], "x": [10, 20], "t": [3, 1]})
            >>> ds.group_by("g").agg(f=bt.first("x", order_by="t")).to_pydict()
            {'g': ['a'], 'f': [20]}
    """
    return _as_column(column).first(order_by)


def last(column: str | Expr, order_by: IntoExpr) -> AggExpr:
    """The value of `column` at the last row in `order_by` order (SQL ``LAST``).

    The `order_by` companion to :func:`first`; partition-independent for the same reason.

    Args:
        column: The column (or expression) whose value to return.
        order_by: The column (or expression) whose ascending order defines "last".

    Returns:
        An aggregate expression of the last value per group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a"], "x": [10, 20], "t": [3, 1]})
            >>> ds.group_by("g").agg(v=bt.last("x", order_by="t")).to_pydict()
            {'g': ['a'], 'v': [10]}
    """
    return _as_column(column).last(order_by)


def arg_min(value: str | Expr, by: IntoExpr) -> AggExpr:
    """The `value` at the row where `by` is smallest (SQL ``ARG_MIN`` / ``MIN_BY``).

    Args:
        value: The column (or expression) whose value to return.
        by: The column (or expression) minimized to select the row.

    Returns:
        An aggregate expression of the `value` at the minimizing row per group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a"], "x": [10, 20], "t": [3, 1]})
            >>> ds.group_by("g").agg(v=bt.arg_min("x", "t")).to_pydict()
            {'g': ['a'], 'v': [20]}
    """
    return _as_column(value).arg_min(by)


def arg_max(value: str | Expr, by: IntoExpr) -> AggExpr:
    """The `value` at the row where `by` is largest (SQL ``ARG_MAX`` / ``MAX_BY``).

    Args:
        value: The column (or expression) whose value to return.
        by: The column (or expression) maximized to select the row.

    Returns:
        An aggregate expression of the `value` at the maximizing row per group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"g": ["a", "a"], "x": [10, 20], "t": [3, 1]})
            >>> ds.group_by("g").agg(v=bt.arg_max("x", "t")).to_pydict()
            {'g': ['a'], 'v': [10]}
    """
    return _as_column(value).arg_max(by)


def q1(column: str | Expr) -> AggExpr:
    """First quartile — the 0.25 quantile of a column.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        An aggregate expression of the first quartile.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
            >>> ds.agg(v=bt.q1("x")).to_pydict()
            {'v': [2.75]}
    """
    return _as_column(column).quantile(0.25)


def q3(column: str | Expr) -> AggExpr:
    """Third quartile — the 0.75 quantile of a column.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        An aggregate expression of the third quartile.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
            >>> ds.agg(v=bt.q3("x")).to_pydict()
            {'v': [6.25]}
    """
    return _as_column(column).quantile(0.75)


def iqr(column: str | Expr) -> Expr:
    """Interquartile range — ``q3 - q1``, the robust spread used for outlier fences.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        The interquartile range per group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
            >>> ds.agg(v=bt.iqr("x")).to_pydict()
            {'v': [3.5]}
    """
    return q3(column) - q1(column)


def value_range(column: str | Expr) -> Expr:
    """Range — ``max - min``, the full spread of a column.

    Named `value_range` rather than ``range`` so it cannot shadow the `bt.range`
    dataset constructor.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        The range per group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
            >>> ds.agg(v=bt.value_range("x")).to_pydict()
            {'v': [7.0]}
    """
    col = _as_column(column)
    return col.max() - col.min()


def null_rate(column: str | Expr) -> Expr:
    """Fraction of rows where the column is null, in ``[0, 1]`` — the completeness check.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        The null fraction per group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, None, 3.0, None]})
            >>> ds.agg(v=bt.null_rate("x")).to_pydict()
            {'v': [0.5]}
    """
    return (count() - _as_column(column).count()) / count()


def non_null_rate(column: str | Expr) -> Expr:
    """Fraction of rows where the column is present, in ``[0, 1]`` — ``1 - null_rate``.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        The non-null fraction per group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, None, 3.0, None]})
            >>> ds.agg(v=bt.non_null_rate("x")).to_pydict()
            {'v': [0.5]}
    """
    return _as_column(column).count() / count()


def nunique_ratio(column: str | Expr) -> Expr:
    """Distinct values divided by row count, in ``[0, 1]`` — the cardinality ratio.

    Near 1 marks an identifier-like column; near 0 a low-cardinality categorical. The
    quickest way to tell which columns are keys and which are features.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        The distinct-to-total ratio per group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 1, 2, 2]})
            >>> ds.agg(v=bt.nunique_ratio("x")).to_pydict()
            {'v': [0.5]}
    """
    return _as_column(column).n_unique() / count()
