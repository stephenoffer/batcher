"""Quantile-based location and spread — the robust half of "describe".

The mean and the standard deviation are the wrong summary for most real columns, because
one bad row moves both without limit. Every measure here is built from quantiles instead,
so a single extreme value cannot shift it: an order statistic ignores *how* far the tail
goes, only how many rows are in it.

All of them are expressions over the existing mergeable `quantile` / `median` aggregates,
so they cost one aggregate pass and work per group and distributed. They are the numbers
worth putting next to a mean before deciding whether the mean means anything.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr
from batcher.plan.functions.aggregate import _as_column
from batcher.plan.functions.quantiles import q1, q3, quantile

__all__ = [
    "decile_ratio",
    "interdecile_range",
    "midhinge",
    "quartile_dispersion",
    "robust_cv",
    "trimean",
]


def midhinge(column: str | Expr) -> Expr:
    """The mean of the first and third quartiles — ``(q1 + q3) / 2``.

    A location estimate that ignores the outer half of the data entirely, so it is the
    center of the bulk rather than the center of mass.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        The midhinge over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
            >>> ds.agg(m=bt.midhinge("x")).to_pydict()
            {'m': [3.0]}
    """
    value = _as_column(column)
    return (q1(value) + q3(value)) / 2.0


def trimean(column: str | Expr) -> Expr:
    """Tukey's trimean — ``(q1 + 2 * median + q3) / 4``.

    Nearly as robust as the median and nearly as efficient as the mean, which is why it is
    the location estimate to reach for when you do not know the distribution's shape. It
    weights the median twice, so it tracks the peak without being pinned to a single row.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        The trimean over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
            >>> ds.agg(m=bt.trimean("x")).to_pydict()
            {'m': [3.0]}
    """
    value = _as_column(column)
    return (q1(value) + 2.0 * value.median() + q3(value)) / 4.0


def quartile_dispersion(column: str | Expr) -> Expr:
    """Quartile coefficient of dispersion — ``(q3 - q1) / (q3 + q1)``.

    A unitless spread measure in ``[0, 1]`` for a strictly positive column, so two columns
    on different scales compare directly. The robust counterpart of the coefficient of
    variation, and the one that stays meaningful on a heavy-tailed column where the
    standard deviation does not.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        The quartile coefficient of dispersion over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
            >>> ds.agg(m=bt.quartile_dispersion("x")).to_pydict()
            {'m': [0.3333333333333333]}
    """
    value = _as_column(column)
    lower, upper = q1(value), q3(value)
    return (upper - lower) / (upper + lower)


def robust_cv(column: str | Expr) -> Expr:
    """The interquartile range over the median — the outlier-proof coefficient of variation.

    Answers "how spread out is this, relative to its typical value" without letting one
    extreme row answer it. Use it instead of `cv` on anything with a tail: response times,
    order values, session lengths.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        The robust coefficient of variation over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
            >>> ds.agg(m=bt.robust_cv("x")).to_pydict()
            {'m': [0.6666666666666666]}
    """
    value = _as_column(column)
    return (q3(value) - q1(value)) / value.median()


def interdecile_range(column: str | Expr) -> Expr:
    """The span between the 10th and 90th percentiles — the middle 80% of the data.

    Wider than the interquartile range and still immune to the extremes, so it is the
    honest answer to "what range do most values fall in" for a service-level or capacity
    conversation.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        The interdecile range over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
            >>> round(ds.agg(m=bt.interdecile_range("x")).to_pydict()["m"][0], 4)
            3.2
    """
    value = _as_column(column)
    return quantile(value, 0.9) - quantile(value, 0.1)


def decile_ratio(column: str | Expr) -> Expr:
    """The 90th percentile divided by the 10th — the classic inequality ratio.

    How many times better off the top of the distribution is than the bottom. Reported for
    income and wealth as the P90/P10 ratio; equally the right summary for latency fairness
    or per-customer revenue concentration.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        The P90/P10 ratio over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
            >>> round(ds.agg(m=bt.decile_ratio("x")).to_pydict()["m"][0], 4)
            3.2857
    """
    value = _as_column(column)
    return quantile(value, 0.9) / quantile(value, 0.1)
