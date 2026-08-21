"""Rank-based classifier metrics — ROC AUC, average precision, KS, Gini.

These are the metrics that cannot be aggregates, because each one depends on the *order*
of every score in the dataset rather than on a per-row quantity. The usual implementation
sorts the scores on one machine, which is exactly what a billion-row evaluation cannot do.

Each is expressed here as a window function plus an aggregate, so the sort is the engine's
distributed sort and the reduction is one mergeable pass. Every function also takes `by=`,
which turns the whole computation into a partitioned window — per-segment AUC over the
full dataset for the same single scan, which is the query that actually gets asked.

ROC AUC uses the rank identity rather than trapezoidal integration over a threshold sweep:

    AUC = (sum of the positives' ranks - n_pos(n_pos + 1)/2) / (n_pos * n_neg)

It is exact, ties included, provided the ranks are *average* ranks over a tie group. The
engine's ``rank`` is the competition rank (the tie group's first position) and ``cume_dist``
gives its last position over n, so their mean recovers the average rank without a second
sort. This agrees with scikit-learn's ``roc_auc_score`` to the last bit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher.ml.stats._shared import require_columns as _require_columns
from batcher.plan.expr_ir.constructors import col, lit, when
from batcher.plan.expr_ir.nodes import cume_dist, rank
from batcher.plan.functions.aggregate import count_if
from batcher.plan.functions.aggregate import sum as sum_
from batcher.plan.functions.metrics.model.classification import positive_mask

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["average_precision", "gini_coefficient", "ks_statistic", "roc_auc"]

# Internal column names for the window quantities. Prefixed so they cannot collide with a
# user column, and dropped before anything is returned.
_RANK = "__bt_rank"
_CUME = "__bt_cume"
_POSITION = "__bt_position"
_CUM_POS = "__bt_cum_pos"
_LABEL = "__bt_label"


def _group_keys(by: str | list[str] | None) -> list[str]:
    """Normalize a `by=` argument into a (possibly empty) list of grouping columns."""
    if by is None:
        return []
    return [by] if isinstance(by, str) else list(by)


def _one_or_frame(result: Dataset, groups: list[str], metric: str) -> Any:
    """The metric as a float when ungrouped, or as a `Dataset` of one row per group."""
    if groups:
        return result
    row = result.collect()
    if row.num_rows == 0:
        return float("nan")
    return row.column(metric)[0].as_py()


def roc_auc(
    ds: Dataset,
    y_true: str,
    y_score: str,
    *,
    positive: Any = 1,
    by: str | list[str] | None = None,
    metric: str = "roc_auc",
) -> float | Dataset:
    """Area under the ROC curve — the probability a random positive outscores a random negative.

    The threshold-free measure of *ranking* quality, and the reason it is reported before
    accuracy on imbalanced data: it is unchanged by the class balance and by any monotone
    rescaling of the scores. 0.5 is chance; below 0.5 means the scores are inverted.

    Args:
        ds: The scored dataset.
        y_true: The label column.
        y_score: The predicted-score column (any monotone score, not only a probability).
        positive: The label value that counts as the positive class.
        by: Column(s) to compute a separate AUC for; returns a `Dataset` when given.
        metric: The output column name when `by` is used.

    Returns:
        The AUC as a float, or a `Dataset` of one row per group when `by` is given.

    Raises:
        ColumnNotFoundError: If `y_true` or `y_score` is not a column of `ds`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import roc_auc
            >>> ds = bt.from_pydict({"y": [0, 0, 1, 1], "s": [0.1, 0.4, 0.35, 0.8]})
            >>> roc_auc(ds, "y", "s")
            0.75
    """
    _require_columns(ds, y_true, y_score)
    groups = _group_keys(by)
    ranked = ds.with_columns(
        **{
            _RANK: rank().over(partition_by=list(groups), order_by=[y_score]),
            _CUME: cume_dist().over(partition_by=list(groups), order_by=[y_score]),
        }
    )
    is_positive = positive_mask(col(y_true), positive)
    positive_rank = when(is_positive).then(col(_RANK).cast("float64")).otherwise(lit(0.0))
    positive_cume = when(is_positive).then(col(_CUME)).otherwise(lit(0.0))
    n_total = col(y_true).count()
    n_positive = count_if(is_positive)
    # The average rank of a tie group is the mean of its first position (`rank`) and its
    # last (`cume_dist * n`), so summing that over the positives is the rank identity's
    # numerator without a second sort.
    rank_sum = (sum_(positive_rank) + n_total * sum_(positive_cume)) / lit(2.0)
    auc = (rank_sum - n_positive * (n_positive + lit(1.0)) / lit(2.0)) / (
        n_positive * (n_total - n_positive)
    )
    aggregated = _aggregate(ranked, groups, {metric: auc})
    return _one_or_frame(aggregated, groups, metric)


def gini_coefficient(
    ds: Dataset,
    y_true: str,
    y_score: str,
    *,
    positive: Any = 1,
    by: str | list[str] | None = None,
    metric: str = "gini",
) -> float | Dataset:
    """The Gini coefficient — ``2 * AUC - 1``, the credit-risk industry's spelling of AUC.

    Same information, rescaled so that 0 is chance and 1 is perfect. Reported here because
    scorecard and underwriting teams read Gini, not AUC, and converting by hand is exactly
    the sort of step that gets done wrong once.

    Args:
        ds: The scored dataset.
        y_true: The label column.
        y_score: The predicted-score column.
        positive: The label value that counts as the positive class.
        by: Column(s) to compute a separate coefficient for.
        metric: The output column name when `by` is used.

    Returns:
        The Gini coefficient as a float, or a `Dataset` when `by` is given.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import gini_coefficient
            >>> ds = bt.from_pydict({"y": [0, 0, 1, 1], "s": [0.1, 0.4, 0.35, 0.8]})
            >>> gini_coefficient(ds, "y", "s")
            0.5
    """
    auc = roc_auc(ds, y_true, y_score, positive=positive, by=by, metric=metric)
    if isinstance(auc, float):
        return 2.0 * auc - 1.0
    return auc.with_columns(**{metric: col(metric) * lit(2.0) - lit(1.0)})


def average_precision(
    ds: Dataset,
    y_true: str,
    y_score: str,
    *,
    positive: Any = 1,
    by: str | list[str] | None = None,
    metric: str = "average_precision",
) -> float | Dataset:
    """Average precision — the area under the precision/recall curve (scikit-learn's ``AP``).

    The metric to report instead of ROC AUC when positives are rare and the negatives are
    uninteresting: AUC counts every negative equally, so at 0.1% prevalence it stays high
    while the top of the ranking is worthless. AP is the mean precision at each positive,
    so it is dominated by the head of the ranking, which is the part anyone acts on.

    Computed as ``mean over positives of (positives seen so far / rows seen so far)`` with
    the rows in descending score order — the streaming definition, which needs one ordered
    running sum rather than a threshold sweep.

    Both running counts share one window frame, so **tied scores** count as a single
    threshold: every row of a tie group sees that group's totals. That is what scikit-learn's
    ``average_precision_score`` does, and it is why this agrees with it under ties. Counting
    the rows with ``row_number`` instead made the two counts disagree — positives counted per
    tie *group* against rows counted per *row* — which is not a precision at all, and drove
    both it and the AP above 1 on a heavily tied column.

    Args:
        ds: The scored dataset.
        y_true: The label column.
        y_score: The predicted-score column.
        positive: The label value that counts as the positive class.
        by: Column(s) to compute a separate value for.
        metric: The output column name when `by` is used.

    Returns:
        The average precision as a float, or a `Dataset` when `by` is given.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import average_precision
            >>> ds = bt.from_pydict({"y": [0, 0, 1, 1], "s": [0.1, 0.4, 0.35, 0.8]})
            >>> round(average_precision(ds, "y", "s"), 6)
            0.833333
    """
    _require_columns(ds, y_true, y_score)
    groups = _group_keys(by)
    labelled = ds.with_columns(
        **{_LABEL: when(positive_mask(col(y_true), positive)).then(lit(1.0)).otherwise(lit(0.0))}
    )
    running = labelled.with_columns(
        **{
            _POSITION: sum_(lit(1.0)).over(partition_by=list(groups), order_by=[(y_score, True)]),
            _CUM_POS: sum_(col(_LABEL)).over(partition_by=list(groups), order_by=[(y_score, True)]),
        }
    )
    precision_at_row = col(_CUM_POS) / col(_POSITION).cast("float64")
    contribution = when(col(_LABEL) == lit(1.0)).then(precision_at_row).otherwise(lit(0.0))
    value = sum_(contribution) / sum_(col(_LABEL))
    aggregated = _aggregate(running, groups, {metric: value})
    return _one_or_frame(aggregated, groups, metric)


def ks_statistic(
    ds: Dataset,
    y_true: str,
    y_score: str,
    *,
    positive: Any = 1,
    by: str | list[str] | None = None,
    metric: str = "ks",
) -> float | Dataset:
    """The Kolmogorov-Smirnov statistic — the widest gap between the two class CDFs.

    The separation measure credit scorecards are graded on: sweep the score from low to
    high and take the largest difference between the fraction of positives seen and the
    fraction of negatives seen. 0 means the two distributions are identical, 1 means a
    threshold separates them perfectly. Unlike AUC it names a *single* operating point,
    which is why a scorecard cutoff is usually set at the KS score.

    Args:
        ds: The scored dataset.
        y_true: The label column.
        y_score: The predicted-score column.
        positive: The label value that counts as the positive class.
        by: Column(s) to compute a separate statistic for.
        metric: The output column name when `by` is used.

    Returns:
        The KS statistic as a float, or a `Dataset` when `by` is given.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import ks_statistic
            >>> ds = bt.from_pydict({"y": [0, 0, 1, 1], "s": [0.1, 0.2, 0.8, 0.9]})
            >>> ks_statistic(ds, "y", "s")
            1.0
    """
    _require_columns(ds, y_true, y_score)
    groups = _group_keys(by)
    is_positive = positive_mask(col(y_true), positive)
    labelled = ds.with_columns(
        **{
            _LABEL: when(is_positive).then(lit(1.0)).otherwise(lit(0.0)),
            "__bt_neg": when(is_positive).then(lit(0.0)).otherwise(lit(1.0)),
        }
    )
    # Both the running counts and the group totals are window aggregates over the same
    # partition, so one projection computes all four — an ordered window for the running
    # side, an unordered one (the whole partition) for the totals.
    running = labelled.with_columns(
        **{
            _CUM_POS: sum_(col(_LABEL)).over(partition_by=list(groups), order_by=[y_score]),
            "__bt_cum_neg": sum_(col("__bt_neg")).over(
                partition_by=list(groups), order_by=[y_score]
            ),
            "__bt_tot_pos": sum_(col(_LABEL)).over(partition_by=list(groups)),
            "__bt_tot_neg": sum_(col("__bt_neg")).over(partition_by=list(groups)),
        }
    )
    gap = (col(_CUM_POS) / col("__bt_tot_pos") - col("__bt_cum_neg") / col("__bt_tot_neg")).abs()
    aggregated = _aggregate(running, groups, {metric: gap.max()})
    return _one_or_frame(aggregated, groups, metric)


def _aggregate(ds: Dataset, groups: list[str], metrics: dict[str, Any]) -> Dataset:
    """Reduce `ds` to the named metrics, grouped by `groups` (globally when empty)."""
    if groups:
        return ds.group_by(*groups).agg(**metrics)
    return ds.agg(**metrics)
