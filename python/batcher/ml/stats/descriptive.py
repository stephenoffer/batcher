"""Statistics that need two passes or a grouping — ranks, entropy, and category association.

The single-pass statistical expressions live in `batcher.plan.functions.analysis` and are
reachable as ``bt.trimean`` / ``bt.welch_t_statistic``. What is here is everything that
*cannot* be one aggregate: a rank correlation needs an ordering, entropy needs the value
distribution, and a trimmed mean needs the quantiles before it can filter on them.

Each is still expressed entirely in relational operators — a window, a `group_by`, or a
second aggregate over the first — so none of them materializes the column on the driver.
That is the difference from the `scipy.stats` equivalents, which all begin by pulling the
whole column into memory.

The sibling modules split the same idea by question: `association` for how two columns
relate, `robust` for location and spread that a corrupted tail cannot move, and `drift` for
comparing two datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher.ml.stats._shared import require_columns, scalar
from batcher.plan.expr_ir.constructors import col, lit
from batcher.plan.expr_ir.nodes import rank
from batcher.plan.functions.aggregate import sum as sum_

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = [
    "entropy",
    "gini_impurity",
    "herfindahl_index",
    "mode_share",
    "normalized_entropy",
    "spearman_corr",
]

_RANK_X = "__bt_rank_x"
_RANK_Y = "__bt_rank_y"
_SHARE = "__bt_share"


def spearman_corr(ds: Dataset, x: str, y: str) -> float:
    """Spearman's rank correlation — Pearson's, computed on the ranks instead of the values.

    The measure to use when the relationship is monotone but not linear, which covers most
    real feature/target pairs: a feature that doubles the target's odds at every step has a
    Spearman of 1 and a Pearson well below it. It is also immune to outliers, because an
    extreme value contributes only its rank.

    Ties take the average rank, matching ``scipy.stats.spearmanr``.

    Args:
        ds: The dataset holding both columns.
        x: The first column.
        y: The second column.

    Returns:
        The rank correlation in ``[-1, 1]``.

    Raises:
        ColumnNotFoundError: If either column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import spearman_corr
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 4.0, 9.0, 16.0]})
            >>> round(spearman_corr(ds, "x", "y"), 6)
            1.0
    """
    require_columns(ds, x, y)
    import batcher as bt

    # Average ranks, not competition ranks: with ties the two differ and only the average
    # rank makes the rank correlation agree with scipy.
    ranked = ds.with_columns(
        **{
            _RANK_X: _average_rank(x),
            _RANK_Y: _average_rank(y),
        }
    )
    return scalar(ranked.agg(r=bt.corr(col(_RANK_X), col(_RANK_Y))), "r")


def _average_rank(column: str) -> Any:
    """The average rank of `column` over the whole dataset, ties shared.

    ``rank`` gives a tie group's first position and ``cume_dist * n`` its last, so their
    mean is the average rank — the same identity the ROC AUC computation uses.
    """
    from batcher.plan.expr_ir.nodes import cume_dist

    total = sum_(lit(1.0)).over()
    return (
        rank().over(order_by=[column]).cast("float64") + cume_dist().over(order_by=[column]) * total
    ) / lit(2.0)


def _value_shares(ds: Dataset, column: str) -> Dataset:
    """One row per distinct value of `column` with its share of the non-null rows."""
    counts = ds.filter(col(column).is_not_null()).group_by(column).agg(n=col(column).count())
    total = sum_(col("n")).over()
    return counts.with_columns(**{_SHARE: col("n").cast("float64") / total})


def entropy(ds: Dataset, column: str, *, base: float = 2.0) -> float:
    """Shannon entropy of a column's value distribution, in `base` units.

    How many bits it takes to describe one value — 0 for a constant column, ``log2(k)`` for
    `k` equally likely values. The single most useful number for triaging a categorical
    feature: near zero means it carries no information, and near ``log2(n)`` means it is an
    identifier rather than a category.

    Args:
        ds: The dataset holding the column.
        column: The column whose distribution to measure.
        base: The logarithm base; 2 gives bits, `math.e` gives nats.

    Returns:
        The entropy in `base` units.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import entropy
            >>> ds = bt.from_pydict({"c": ["a", "a", "b", "b"]})
            >>> entropy(ds, "c")
            1.0
    """
    import math

    require_columns(ds, column)
    shares = _value_shares(ds, column)
    contribution = col(_SHARE) * col(_SHARE).ln()
    return -scalar(shares.agg(h=sum_(contribution)), "h") / math.log(base)


def gini_impurity(ds: Dataset, column: str) -> float:
    """Gini impurity — ``1 - sum(p^2)``, the probability two random rows differ.

    The split criterion a decision tree optimizes, and a faster-to-read sibling of
    `entropy`: 0 for a pure column, approaching 1 as the values spread out. Bounded above
    by ``1 - 1/k`` for `k` distinct values.

    Args:
        ds: The dataset holding the column.
        column: The column whose distribution to measure.

    Returns:
        The Gini impurity in ``[0, 1)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import gini_impurity
            >>> ds = bt.from_pydict({"c": ["a", "a", "b", "b"]})
            >>> gini_impurity(ds, "c")
            0.5
    """
    require_columns(ds, column)
    shares = _value_shares(ds, column)
    return 1.0 - scalar(shares.agg(g=sum_(col(_SHARE) * col(_SHARE))), "g")


def herfindahl_index(ds: Dataset, column: str) -> float:
    """The Herfindahl-Hirschman concentration index — ``sum(p^2)``.

    The complement of `gini_impurity`, read the other way round: 1 means one value holds
    everything, ``1/k`` means `k` values share it evenly. The standard measure of market
    concentration, and equally the right one for "is this traffic dominated by one client".

    Args:
        ds: The dataset holding the column.
        column: The column whose distribution to measure.

    Returns:
        The concentration index in ``(0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import herfindahl_index
            >>> ds = bt.from_pydict({"c": ["a", "a", "b", "b"]})
            >>> herfindahl_index(ds, "c")
            0.5
    """
    return 1.0 - gini_impurity(ds, column)


def mode_share(ds: Dataset, column: str) -> float:
    """The fraction of non-null rows holding the single most common value.

    The number that reveals a column is not what its dtype claims: an ``int64`` where 94%
    of rows share one value is a flag, not a measurement. Also the fastest detector of a
    broken upstream default that has silently replaced a real signal.

    Args:
        ds: The dataset holding the column.
        column: The column whose distribution to measure.

    Returns:
        The modal share in ``(0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import mode_share
            >>> ds = bt.from_pydict({"c": ["a", "a", "a", "b"]})
            >>> mode_share(ds, "c")
            0.75
    """
    require_columns(ds, column)
    shares = _value_shares(ds, column)
    return scalar(shares.agg(m=col(_SHARE).max()), "m")


def normalized_entropy(ds: Dataset, column: str) -> float:
    """The column's entropy scaled to ``[0, 1]`` — Shannon entropy over its maximum.

    `entropy` grows with the number of distinct values, so a value from a 3-category column is not
    comparable with one from a 300-category column. Dividing by ``log(k)`` — the entropy of `k`
    equally likely values — fixes that: 0 is a constant column, 1 is a perfectly uniform one, and
    the number means the same thing across columns of any cardinality. That comparability is what
    makes it the right entropy to rank features by or to threshold in a data-quality check.

    Args:
        ds: The dataset holding the column.
        column: The column whose distribution to measure.

    Returns:
        The normalized entropy in ``[0, 1]``; NaN for a column with fewer than two distinct values.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import normalized_entropy
            >>> ds = bt.from_pydict({"c": ["a", "a", "b", "b"]})
            >>> normalized_entropy(ds, "c")
            1.0
    """
    import math

    require_columns(ds, column)
    distinct = ds.agg(k=col(column).n_unique()).collect().column("k")[0].as_py()
    if distinct is None or distinct < 2:
        return float("nan")
    return entropy(ds, column, base=math.e) / math.log(distinct)
