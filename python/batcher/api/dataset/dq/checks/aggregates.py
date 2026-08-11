"""Relation-level constraints: one number over the whole table, inside bounds.

These are the checks that no single row can violate — the table has too few rows, the
average order value halved overnight, a column that was 1% null is now 40% null. Deequ
calls them analyzers with an assertion; Great Expectations calls them table expectations.
They answer the failure mode a row-level contract is structurally blind to: data that is
individually well-formed and collectively wrong.

Because there is no violating row, `drop` and `quarantine` cannot act on one, and refuse
rather than silently ignoring it. `validate` and `fail` are where these belong.
"""

from __future__ import annotations

from batcher._internal.errors import PlanError
from batcher.api.dataset.dq.constraints import AggregateConstraint
from batcher.plan.expr_ir import Col, count
from batcher.plan.functions.aggregate import median, n_unique, std
from batcher.plan.functions.quantiles import quantile
from batcher.plan.functions.statistics import null_rate, nunique_ratio

__all__ = [
    "distinct_count_between",
    "mean_between",
    "median_between",
    "null_rate_below",
    "quantile_between",
    "row_count_between",
    "stddev_between",
    "sum_between",
    "unique_ratio_above",
]


def _bounds(check: str, low: float | None, high: float | None) -> None:
    """Reject a bound pair that can never hold, where the check's name is still known."""
    if low is None and high is None:
        raise PlanError(
            f"{check}: give at least one of low/high — with neither, the constraint holds always."
        )
    if low is not None and high is not None and low > high:
        raise PlanError(f"{check}: low ({low!r}) > high ({high!r}) — swap the arguments?")


def row_count_between(low: int | None = None, high: int | None = None) -> AggregateConstraint:
    """The relation must have between `low` and `high` rows, inclusive.

    Args:
        low: Inclusive minimum row count, or `None` for no minimum.
        high: Inclusive maximum row count, or `None` for no maximum.

    Returns:
        The relation-level constraint.
    """
    _bounds("row_count_between", low, high)
    return AggregateConstraint(f"row_count_between({low}, {high})", count(), low, high)


def mean_between(column: str, low: float | None = None, high: float | None = None):
    """`column`'s mean must lie in ``[low, high]``.

    Args:
        column: The numeric column to average.
        low: Inclusive minimum, or `None`.
        high: Inclusive maximum, or `None`.

    Returns:
        The relation-level constraint.
    """
    _bounds(f"mean_between({column!r})", low, high)
    return AggregateConstraint(
        f"mean_between({column}, {low}, {high})", Col(column).mean(), low, high
    )


def sum_between(column: str, low: float | None = None, high: float | None = None):
    """`column`'s sum must lie in ``[low, high]``.

    Args:
        column: The numeric column to total.
        low: Inclusive minimum, or `None`.
        high: Inclusive maximum, or `None`.

    Returns:
        The relation-level constraint.
    """
    _bounds(f"sum_between({column!r})", low, high)
    return AggregateConstraint(
        f"sum_between({column}, {low}, {high})", Col(column).sum(), low, high
    )


def median_between(column: str, low: float | None = None, high: float | None = None):
    """`column`'s median must lie in ``[low, high]``.

    Args:
        column: The numeric column to measure.
        low: Inclusive minimum, or `None`.
        high: Inclusive maximum, or `None`.

    Returns:
        The relation-level constraint.
    """
    _bounds(f"median_between({column!r})", low, high)
    return AggregateConstraint(
        f"median_between({column}, {low}, {high})", median(Col(column)), low, high
    )


def stddev_between(column: str, low: float | None = None, high: float | None = None):
    """`column`'s sample standard deviation must lie in ``[low, high]``.

    A collapsed standard deviation is how a column that was replaced by a constant default
    announces itself, and no row-level check can see it.

    Args:
        column: The numeric column to measure.
        low: Inclusive minimum, or `None`.
        high: Inclusive maximum, or `None`.

    Returns:
        The relation-level constraint.
    """
    _bounds(f"stddev_between({column!r})", low, high)
    return AggregateConstraint(
        f"stddev_between({column}, {low}, {high})", std(Col(column)), low, high
    )


def quantile_between(
    column: str, q: float, low: float | None = None, high: float | None = None
) -> AggregateConstraint:
    """`column`'s `q`-quantile must lie in ``[low, high]``.

    A bound on a quantile survives the outliers that move a mean, so it is the right shape
    for "the typical value has not shifted" on a skewed column.

    Args:
        column: The numeric column to measure.
        q: The quantile, between 0 and 1.
        low: Inclusive minimum, or `None`.
        high: Inclusive maximum, or `None`.

    Returns:
        The relation-level constraint.
    """
    if not 0.0 <= q <= 1.0:
        raise PlanError(f"quantile_between({column!r}): q must be in [0, 1], got {q!r}")
    _bounds(f"quantile_between({column!r})", low, high)
    return AggregateConstraint(
        f"quantile_between({column}, q={q}, {low}, {high})", quantile(Col(column), q), low, high
    )


def null_rate_below(column: str, max_rate: float) -> AggregateConstraint:
    """`column`'s share of nulls must not exceed `max_rate`.

    The tolerated form of `not_null`, for a column that is legitimately sparse but whose
    sparsity is itself the signal: 2% missing is the feed working, 60% is it broken.

    Args:
        column: The column to measure.
        max_rate: The largest acceptable null share, between 0 and 1.

    Returns:
        The relation-level constraint.
    """
    if not 0.0 <= max_rate <= 1.0:
        raise PlanError(
            f"null_rate_below({column!r}): max_rate must be a fraction in [0, 1], got {max_rate!r}"
        )
    return AggregateConstraint(
        f"null_rate_below({column}, {max_rate})", null_rate(Col(column)), None, max_rate
    )


def distinct_count_between(
    column: str, low: int | None = None, high: int | None = None
) -> AggregateConstraint:
    """`column`'s number of distinct values must lie in ``[low, high]``.

    Args:
        column: The column to measure.
        low: Inclusive minimum, or `None`.
        high: Inclusive maximum, or `None`.

    Returns:
        The relation-level constraint.
    """
    _bounds(f"distinct_count_between({column!r})", low, high)
    return AggregateConstraint(
        f"distinct_count_between({column}, {low}, {high})", n_unique(Col(column)), low, high
    )


def unique_ratio_above(column: str, min_ratio: float) -> AggregateConstraint:
    """`column`'s distinct-to-row ratio must be at least `min_ratio`.

    The scale-free version of `distinct_count_between`: a bound that stays true as the table
    grows, which is what a "this is nearly a key" contract actually means.

    Args:
        column: The column to measure.
        min_ratio: The smallest acceptable distinct/row ratio, between 0 and 1.

    Returns:
        The relation-level constraint.
    """
    if not 0.0 <= min_ratio <= 1.0:
        raise PlanError(
            f"unique_ratio_above({column!r}): min_ratio must be in [0, 1], got {min_ratio!r}"
        )
    return AggregateConstraint(
        f"unique_ratio_above({column}, {min_ratio})", nunique_ratio(Col(column)), min_ratio, None
    )
