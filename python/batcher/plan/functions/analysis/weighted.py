"""Weighted statistics — a mean, variance, and correlation where rows carry different weights.

Not every row counts equally. A survey has sampling weights, a time series decays older
observations, an aggregate-of-aggregates carries each group's size, and a cost-sensitive model
weights the expensive class up. The unweighted mean and variance are wrong for all of these,
and reaching for them silently is one of the more common quiet errors in applied statistics.

Each function here is the weighted form built from weighted moments — ``sum(w*x) / sum(w)`` and
its second-moment cousin — so each is a single mergeable aggregate that runs distributed and
composes inside `group_by` exactly as the unweighted versions do. The weights are used as
*reliability* weights (frequency weights), which is the convention every downstream tool
expects and matches ``numpy.average`` and a frequency-weighted ``numpy.cov``.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column

__all__ = [
    "weighted_correlation",
    "weighted_covariance",
    "weighted_std",
    "weighted_var",
]


def weighted_var(value: IntoExpr, weight: IntoExpr) -> Expr:
    """The weighted (population) variance — ``E_w[x^2] - E_w[x]^2``.

    The spread of a weighted sample around its weighted mean, using the frequency-weight
    convention (no Bessel correction, matching ``numpy.average`` on the squared deviations).
    A survey weight, a decay weight, or a per-group size all belong here; the unweighted
    variance would treat a heavily-weighted outlier as one ordinary row.

    Args:
        value: The measured column.
        weight: The per-row weight column.

    Returns:
        The weighted variance over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 3.0], "w": [1.0, 1.0]})
            >>> ds.agg(v=bt.weighted_var("x", "w")).to_pydict()
            {'v': [1.0]}
    """
    x, w = _as_column(value), _as_column(weight)
    total = w.sum()
    mean = (w * x).sum() / total
    mean_square = (w * x * x).sum() / total
    return mean_square - mean * mean


def weighted_std(value: IntoExpr, weight: IntoExpr) -> Expr:
    """The weighted standard deviation — ``sqrt(weighted_var)``.

    Args:
        value: The measured column.
        weight: The per-row weight column.

    Returns:
        The weighted standard deviation over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, 3.0], "w": [1.0, 1.0]})
            >>> ds.agg(s=bt.weighted_std("x", "w")).to_pydict()
            {'s': [1.0]}
    """
    return weighted_var(value, weight).sqrt()


def weighted_covariance(x: IntoExpr, y: IntoExpr, weight: IntoExpr) -> Expr:
    """The weighted covariance of two columns — ``E_w[xy] - E_w[x]E_w[y]``.

    The building block of a weighted correlation and a weighted regression slope. Uses the
    frequency-weight convention, matching a ``numpy.cov`` with integer ``fweights``.

    Args:
        x: The first column.
        y: The second column.
        weight: The per-row weight column.

    Returns:
        The weighted covariance over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0], "w": [1.0, 1.0, 1.0]}
            ... )
            >>> round(ds.agg(c=bt.weighted_covariance("x", "y", "w")).to_pydict()["c"][0], 4)
            1.3333
    """
    xc, yc, w = _as_column(x), _as_column(y), _as_column(weight)
    total = w.sum()
    mean_x = (w * xc).sum() / total
    mean_y = (w * yc).sum() / total
    mean_xy = (w * xc * yc).sum() / total
    return mean_xy - mean_x * mean_y


def weighted_correlation(x: IntoExpr, y: IntoExpr, weight: IntoExpr) -> Expr:
    """The weighted Pearson correlation — weighted covariance over the weighted standard deviations.

    The correlation of two columns when rows carry different weights: a survey-weighted
    association, a recency-weighted one, or a size-weighted correlation across pre-aggregated
    groups. In ``[-1, 1]`` as an ordinary correlation is, and it reduces to the unweighted
    correlation when every weight is equal.

    Args:
        x: The first column.
        y: The second column.
        weight: The per-row weight column.

    Returns:
        The weighted correlation over the group, in ``[-1, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "w": [1.0, 2.0, 3.0]}
            ... )
            >>> round(ds.agg(r=bt.weighted_correlation("x", "y", "w")).to_pydict()["r"][0], 6)
            1.0
    """
    covariance = weighted_covariance(x, y, weight)
    return covariance / (weighted_std(x, weight) * weighted_std(y, weight))
