"""How strongly two columns move together — the feature-selection primitives.

Pearson `corr` already exists and answers one narrow question: is there a *linear*
relationship between two numeric columns. Most feature-selection questions are not that
one. A monotone but curved relationship, a numeric feature against a binary label, a
feature whose relationship reverses direction — each needs a different measure, and
reporting Pearson for all of them is how a genuinely predictive feature gets dropped.

Everything here is arithmetic over the existing mergeable `corr` / `var` aggregates, so a
correlation screen over 500 features against a label is one pass. Plain Pearson (`corr`)
and the linear fit statistics (`regr_r2` and the rest of the ``regr_*`` family) already
exist and are not restated here.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr, IntoExpr, Lit
from batcher.plan.functions.aggregate import _as_column, corr

__all__ = ["correlation_ratio", "point_biserial", "signal_ratio"]


def _paired(x: IntoExpr, y: IntoExpr) -> tuple[Expr, Expr]:
    """Both columns masked to null wherever either is null, so the pairing is exact."""
    left, right = _as_column(x), _as_column(y)
    return left + right * Lit(0), right + left * Lit(0)


def point_biserial(value: IntoExpr, outcome: Expr) -> Expr:
    """The correlation between a numeric column and a binary outcome.

    Mathematically Pearson's correlation with the outcome coded 0/1, which is exactly what
    makes it useful: it is on the same ``[-1, 1]`` scale as every other correlation, so a
    mixed screen of numeric and boolean features against a numeric target ranks on one
    axis. Positive means the feature is larger where the outcome is true.

    Args:
        value: The numeric column.
        outcome: A boolean expression (the binary side).

    Returns:
        The point-biserial correlation in ``[-1, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 8.0, 9.0], "hit": [False, False, True, True]})
            >>> ds.agg(r=bt.point_biserial("x", bt.col("hit"))).to_pydict()
            {'r': [0.9899494936611665]}
    """
    return corr(_as_column(value), outcome.cast("float64"))


def correlation_ratio(value: IntoExpr, group_mean_of_value: IntoExpr) -> Expr:
    """Eta squared — the share of variance explained by a grouping, given its group means.

    The categorical-feature counterpart of a squared correlation: it measures how much of a numeric
    column's variance is between groups rather than within them, on a ``[0, 1]`` scale, and
    it detects a relationship of *any* shape rather than only a linear one.

    Takes the per-row group mean rather than the grouping column, because that is what
    makes it a single aggregate: compute the group means with a window
    (``bt.mean(col("y")).over(partition_by=["g"])``) and pass the resulting column here.

    Args:
        value: The numeric column.
        group_mean_of_value: The per-row mean of `value` within its group.

    Returns:
        Eta squared in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1.0, 2.0, 10.0, 11.0], "g": ["a", "a", "b", "b"]})
            >>> means = ds.with_columns(m=bt.mean(bt.col("y")).over(partition_by=["g"]))
            >>> round(means.agg(e=bt.correlation_ratio("y", "m")).to_pydict()["e"][0], 4)
            0.9878
    """
    observed, fitted = _paired(value, group_mean_of_value)
    return fitted.var() / observed.var()


def signal_ratio(value: IntoExpr, outcome: Expr) -> Expr:
    """The absolute difference in means between the two outcome groups, in standard deviations.

    A feature screen that survives a non-linear relationship, unlike a correlation: it asks
    only whether the feature *separates* the two classes, not whether it does so
    monotonically. Larger is a stronger feature; below about 0.2 the feature is unlikely to
    contribute on its own.

    Args:
        value: The numeric column.
        outcome: A boolean expression splitting the rows.

    Returns:
        The separation in standard deviations, always non-negative.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 8.0, 9.0], "hit": [False, False, True, True]})
            >>> round(ds.agg(s=bt.signal_ratio("x", bt.col("hit"))).to_pydict()["s"][0], 4)
            1.7146
    """
    from batcher.plan.functions.analysis.inference import group_mean

    column = _as_column(value)
    difference = group_mean(value, outcome) - group_mean(value, ~outcome)
    return (difference / column.std()).abs()
