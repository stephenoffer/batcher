"""Distribution shape — skew, tail weight, and how far from normal a column is.

Whether a column is symmetric and how heavy its tails are decides which model, which
transform, and which summary statistic is appropriate, and it is the first thing worth
knowing about a feature. The moment-based answers (`skewness`, `kurtosis`) already exist
as aggregates; what is here is the rest of the shape toolkit:

- **Quantile-based shape** (`bowley_skew`, `moors_kurtosis`) — the robust versions, which
  do not need the fourth moment to exist. On a heavy-tailed column the moment estimates
  are dominated by the largest few rows and effectively measure the maximum; these do not.
- **Normality** (`jarque_bera`) — the standard test statistic, so "is this normal enough"
  becomes a number rather than a look at a histogram.
All are single-pass expressions over mergeable aggregates.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr
from batcher.plan.functions.aggregate import _as_column
from batcher.plan.functions.quantiles import q1, q3, quantile

__all__ = ["bowley_skew", "jarque_bera", "moors_kurtosis", "pearson_mode_skew"]


def bowley_skew(column: str | Expr) -> Expr:
    """Quartile skewness — ``(q3 + q1 - 2 * median) / (q3 - q1)``, bounded in ``[-1, 1]``.

    The robust alternative to `skewness`: it asks whether the upper quartile is further
    from the median than the lower one, which is what "skewed" means, without letting the
    single largest value decide the answer. 0 is symmetric, positive is a longer right
    tail.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        The Bowley skewness over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
            >>> ds.agg(m=bt.bowley_skew("x")).to_pydict()
            {'m': [0.0]}
    """
    value = _as_column(column)
    lower, upper = q1(value), q3(value)
    return (upper + lower - 2.0 * value.median()) / (upper - lower)


def moors_kurtosis(column: str | Expr) -> Expr:
    """Octile-based kurtosis — tail weight measured from order statistics.

    ``((o7 - o5) + (o3 - o1)) / (o6 - o2)`` over the octiles, which is roughly 1.23 for a
    normal distribution; higher means heavier tails. Unlike the fourth moment it stays
    finite and meaningful on a distribution whose fourth moment does not exist, which
    covers most real latency, revenue, and file-size columns.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        Moors' kurtosis over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [float(i) for i in range(1, 101)]})
            >>> round(ds.agg(m=bt.moors_kurtosis("x")).to_pydict()["m"][0], 4)
            1.0
    """
    value = _as_column(column)
    octile = [quantile(value, i / 8.0) for i in range(9)]
    return ((octile[7] - octile[5]) + (octile[3] - octile[1])) / (octile[6] - octile[2])


def pearson_mode_skew(column: str | Expr) -> Expr:
    """Pearson's first skewness coefficient — ``(mean - mode) / stddev``.

    Reads directly as "how many standard deviations the average sits above the most common
    value", which is the sentence a non-statistician understands. Meaningful only when the
    column has a genuine mode; on a continuous column with no repeated values the mode is
    an artifact and this number is noise.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        Pearson's mode skewness over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 1.0, 1.0, 2.0, 6.0]})
            >>> round(ds.agg(m=bt.pearson_mode_skew("x")).to_pydict()["m"][0], 4)
            0.5535
    """
    value = _as_column(column)
    return (value.mean() - value.mode()) / value.std()


def jarque_bera(column: str | Expr) -> Expr:
    """The Jarque-Bera normality statistic — ``n/6 * (S^2 + K^2/4)``.

    Combines skewness and excess kurtosis into one number that is chi-squared with two
    degrees of freedom under normality, so roughly: below 6 is consistent with a normal
    column, and far above it is not. Useful as a *screen* over hundreds of features rather
    than as a formal test, because at a large `n` it rejects normality for any real column.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        The Jarque-Bera statistic over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
            >>> round(ds.agg(m=bt.jarque_bera("x")).to_pydict()["m"][0], 4)
            9.334
    """
    value = _as_column(column)
    skew = value.skewness()
    excess = value.kurtosis()
    return value.count() / 6.0 * (skew * skew + excess * excess / 4.0)
