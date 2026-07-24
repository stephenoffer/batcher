"""Dispersion ratios — spread expressed relative to level, in one pass.

A raw standard deviation is hard to compare across columns on different scales. These express
spread *relative to* the column's own level — variance over mean, range over standard
deviation — so the number is unitless and comparable, and each says something a raw spread
does not. They are the summaries that answer "is this column noisy for its size", which is
the question that decides whether a feature is worth keeping.

Every one is a single mergeable aggregate over the existing `mean`/`var`/`std`/`min`/`max`
primitives, so it runs distributed and composes inside `group_by`.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr
from batcher.plan.functions.aggregate import _as_column


def index_of_dispersion(column: str | Expr) -> Expr:
    """The variance-to-mean ratio (Fano factor) — ``var / mean``.

    The number that classifies a count process: exactly 1 for a Poisson process (the
    variance equals the mean), above 1 for a clustered/over-dispersed one, below 1 for a
    regular/under-dispersed one. The first thing to check before fitting a Poisson model — an
    index far from 1 says the Poisson assumption is wrong and a negative-binomial or a
    quasi-Poisson is needed.

    Args:
        column: The non-negative count column (or expression) to summarize.

    Returns:
        The variance-to-mean ratio over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [2.0, 2.0, 2.0, 2.0]})
            >>> ds.agg(m=bt.index_of_dispersion("x")).to_pydict()
            {'m': [0.0]}
    """
    value = _as_column(column)
    return value.var() / value.mean()


def signal_to_noise(column: str | Expr) -> Expr:
    """The signal-to-noise ratio — ``mean / std``, the reciprocal of the coefficient of variation.

    How large the typical value is relative to its own scatter. High means a stable, reliable
    column; near zero means the noise swamps the level. The engineering framing of dispersion,
    and the right screen for a feature whose usefulness depends on being consistently nonzero.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        The signal-to-noise ratio over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [10.0, 11.0, 9.0, 10.0]})
            >>> round(ds.agg(m=bt.signal_to_noise("x")).to_pydict()["m"][0], 4)
            12.2474
    """
    value = _as_column(column)
    return value.mean() / value.std()


def studentized_range(column: str | Expr) -> Expr:
    """The range in standard deviations — ``(max - min) / std``.

    How many standard deviations separate the extremes. It is the statistic behind the
    normality range test and a quick outlier smell: for a normal sample it grows slowly with
    ``n`` (about 6 at n=1000), so a much larger value flags a heavy tail or a contaminant
    without needing a full outlier pass.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        The studentized range over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
            >>> round(ds.agg(m=bt.studentized_range("x")).to_pydict()["m"][0], 4)
            2.5298
    """
    value = _as_column(column)
    return (value.max() - value.min()) / value.std()


def relative_range(column: str | Expr) -> Expr:
    """The range relative to the mean — ``(max - min) / mean``.

    A unitless spread that, unlike the coefficient of variation, is driven entirely by the two
    extremes rather than the whole distribution. Useful as a cheap volatility measure and as a
    complement to `signal_to_noise`: a large relative range with a small coefficient of
    variation says the spread is a couple of outliers, not a broad distribution.

    Args:
        column: The column (or expression) to summarize.

    Returns:
        The relative range over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [2.0, 4.0, 6.0]})
            >>> ds.agg(m=bt.relative_range("x")).to_pydict()
            {'m': [1.0]}
    """
    value = _as_column(column)
    return (value.max() - value.min()) / value.mean()


def geometric_std(column: str | Expr) -> Expr:
    """The geometric standard deviation — ``exp(std(ln x))``, the spread of a log-normal column.

    The multiplicative counterpart of the standard deviation, for a strictly positive column whose
    values span orders of magnitude (a price, a population, a concentration). It is a *factor*, not
    an amount: a geometric std of 2 means a typical value is within a factor of 2 of the geometric
    mean, which is the honest way to describe scatter on a log scale where an ordinary standard
    deviation is dominated by the largest values. Matches ``scipy.stats.gstd``.

    Args:
        column: The strictly positive column (or expression) to summarize.

    Returns:
        The geometric standard deviation over the group, a factor at least 1.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 4.0, 8.0, 16.0]})
            >>> round(ds.agg(g=bt.geometric_std("x")).to_pydict()["g"][0], 4)
            2.9921
    """
    return _as_column(column).ln().std().exp()
