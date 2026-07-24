"""Robust location and spread — trimming, winsorizing, and the median absolute deviation.

The estimators that survive a corrupted tail. Each needs two passes and says so: the cut
points come from one aggregate, and the summary from a second over the filtered or clamped
rows. A single pass cannot do it, because a filter cannot reference an aggregate of the
rows it is filtering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.stats._shared import require_columns, scalar
from batcher.plan.expr_ir.constructors import col, lit

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = [
    "mean_abs_deviation",
    "median_abs_deviation",
    "outlier_mask",
    "trimmed_mean",
    "winsorized_mean",
]


def trimmed_mean(ds: Dataset, column: str, *, proportion: float = 0.1) -> float:
    """The mean of the middle ``1 - 2 * proportion`` of the values, by quantile.

    The robust location estimate that keeps most of the mean's efficiency: it discards the
    extreme tails rather than down-weighting them, so a handful of corrupt rows cannot move
    it at all while the bulk of the data still contributes.

    Two passes — one to find the cut points, one to average between them — because a filter
    cannot reference an aggregate of the rows it is filtering.

    Args:
        ds: The dataset holding the column.
        column: The numeric column to summarize.
        proportion: The fraction to trim from *each* tail, in ``[0, 0.5)``.

    Returns:
        The trimmed mean.

    Raises:
        PlanError: If `proportion` is outside ``[0, 0.5)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import trimmed_mean
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 1000.0]})
            >>> trimmed_mean(ds, "x", proportion=0.2)
            3.0
    """
    low, high = _tail_cuts(ds, column, proportion)
    kept = ds.filter((col(column) >= lit(low)) & (col(column) <= lit(high)))
    return scalar(kept.agg(m=col(column).mean()), "m")


def winsorized_mean(ds: Dataset, column: str, *, proportion: float = 0.1) -> float:
    """The mean after clamping the tails to the `proportion` quantiles instead of dropping them.

    The difference from `trimmed_mean` is what happens to an extreme row: here it is pulled
    in to the cut point and still counted, so the sample size is unchanged. That matters
    when the count itself is meaningful — a per-customer average where dropping the biggest
    customers would change what the average is *of*.

    The cut points are *interpolated* quantiles, as in `trimmed_mean`, not order statistics.
    On a sample small enough that the upper quantile interpolates toward the outlier, the
    clamp lands past it and the outlier survives; the two agree from a few dozen rows up.

    Args:
        ds: The dataset holding the column.
        column: The numeric column to summarize.
        proportion: The fraction to clamp at *each* tail, in ``[0, 0.5)``.

    Returns:
        The winsorized mean.

    Raises:
        PlanError: If `proportion` is outside ``[0, 0.5)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import winsorized_mean
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000.0]})
            >>> round(winsorized_mean(ds, "x", proportion=0.2), 4)
            5.5
    """
    low, high = _tail_cuts(ds, column, proportion)
    clamped = ds.with_columns(**{column: col(column).clip(lit(low), lit(high))})
    return scalar(clamped.agg(m=col(column).mean()), "m")


def _tail_cuts(ds: Dataset, column: str, proportion: float) -> tuple[float, float]:
    """The lower and upper quantile cut points for a trim or a winsorization."""
    require_columns(ds, column)
    if not 0.0 <= proportion < 0.5:
        raise PlanError(f"proportion must be in [0, 0.5), got {proportion}")
    bounds = ds.agg(
        low=col(column).quantile(proportion),
        high=col(column).quantile(1.0 - proportion),
    ).collect()
    return (
        float(bounds.column("low")[0].as_py()),
        float(bounds.column("high")[0].as_py()),
    )


def median_abs_deviation(ds: Dataset, column: str, *, scale: float = 1.4826) -> float:
    """The median absolute deviation from the median, scaled to estimate a standard deviation.

    The most robust spread estimate in common use: it tolerates up to half the data being
    arbitrary. The default `scale` of 1.4826 makes it equal to the standard deviation for a
    normal column, so it drops into any formula expecting a sigma — including the
    ``|x - median| / MAD > 3`` outlier rule, which is what to use instead of a z-score on
    anything with a tail.

    Args:
        ds: The dataset holding the column.
        column: The numeric column to summarize.
        scale: The consistency constant; 1.4826 matches the normal standard deviation.

    Returns:
        The scaled median absolute deviation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import median_abs_deviation
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 1000.0]})
            >>> median_abs_deviation(ds, "x", scale=1.0)
            1.0
    """
    require_columns(ds, column)
    center = scalar(ds.agg(m=col(column).median()), "m")
    deviations = ds.with_columns(__bt_dev=(col(column) - lit(center)).abs())
    return scale * scalar(deviations.agg(m=col("__bt_dev").median()), "m")


def outlier_mask(column: str, *, center: float, spread: float, threshold: float = 3.0) -> Any:
    """A boolean `Expr` marking rows more than `threshold` robust deviations from `center`.

    Split from `median_abs_deviation` so the two passes are explicit: measure once with
    that function, then filter, flag, or clip with this expression as many times as you
    like without re-measuring.

    Args:
        column: The numeric column to test.
        center: The robust center, usually the median.
        spread: The robust spread, usually the scaled median absolute deviation.
        threshold: How many spreads away counts as an outlier.

    Returns:
        A boolean `Expr`, true on an outlier.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import median_abs_deviation, outlier_mask
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 1000.0]})
            >>> spread = median_abs_deviation(ds, "x")
            >>> ds.filter(outlier_mask("x", center=3.0, spread=spread)).to_pydict()
            {'x': [1000.0]}
    """
    if spread <= 0.0:
        # A zero spread makes every non-central row infinitely far away, which is the right
        # answer only if you meant "not equal"; say so rather than dividing by zero.
        return col(column) != lit(center)
    return ((col(column) - lit(center)).abs() / lit(spread)) > lit(threshold)


def mean_abs_deviation(ds: Dataset, column: str) -> float:
    """The mean absolute deviation from the mean — ``mean(|x - mean(x)|)``.

    A dispersion measure between the standard deviation and the `median_abs_deviation`: it uses the
    mean rather than the median as its center, so it is less robust than the MAD, but it weights
    every deviation linearly rather than quadratically, so a single outlier moves it far less than
    the standard deviation. Reach for it when you want the standard deviation's familiar center but
    the median absolute deviation's resistance to a heavy tail.

    Computed in two passes — the mean, then the mean absolute deviation from it — because a metric
    cannot reference an aggregate of the rows it is summarizing.

    Args:
        ds: The dataset holding the column.
        column: The numeric column to summarize.

    Returns:
        The mean absolute deviation from the mean.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import mean_abs_deviation
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0]})
            >>> mean_abs_deviation(ds, "x")
            1.0
    """
    require_columns(ds, column)
    center = scalar(ds.agg(m=col(column).mean()), "m")
    return scalar(ds.agg(m=(col(column) - lit(center)).abs().mean()), "m")
