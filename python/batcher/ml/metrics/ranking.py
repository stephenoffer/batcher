"""Ranking metrics — how good is the *order* a recommender produced, per query.

A recommender is not scored the way a classifier is. What matters is whether the relevant
items reached the top of one user's list, and then how that averages over users — so every
metric here is computed **within a group** and then averaged, never pooled across groups.
Pooling is the classic mistake and it silently rewards a model that ranks one heavy user
well and everyone else badly.

The shape all of these expect is one row per ``(query, candidate)`` pair with a score and a
relevance label — the natural output of scoring a candidate set, and the shape
`ds.ml.predict` already produces. The ranking is a window over the query, so a metric over
a billion pairs is one partitioned sort and one aggregate.

`k` is the cutoff, and it should be the number of items you actually show. A precision@10
on a UI that shows three is measuring something nobody sees.

Three conventions hold for every function here:

- **Rows with a null label, or a null or NaN score, are dropped** before ranking. A null
  score used to sort to the top of its query and be ranked as the best candidate.
- **Tied scores do not favour the relevant item.** Breaking a tie by arrival order with
  ``row_number`` put a relevant item first whenever it happened to arrive first, so four
  tied candidates with one relevant scored NDCG@1 = 1.0 where scikit-learn's ``ndcg_score``
  gives 0.25. The count-based metrics (`precision_at_k`, `recall_at_k`, `ndcg_at_k`) now
  score a tie group by its *average* relevance at each position it covers, which is
  ``ndcg_score``'s ``ignore_ties=False`` rule and the expected value over every order the
  tie allows. The first-hit and average-precision metrics (`hit_rate_at_k`,
  `mean_reciprocal_rank`, `map_at_k`) place a whole tie group at its *last* position, the
  rule scikit-learn's ``average_precision_score`` and
  ``label_ranking_average_precision_score`` use, so a tie straddling the cutoff counts as
  below it.
- **A query with no relevant item scores 0 and stays in the mean**, as it does in
  ``ndcg_score``. Every metric here averages over the same set of queries, so the columns of
  one evaluation table are comparable.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.stats._shared import require_columns
from batcher.plan.expr_ir.constructors import col, lit, when
from batcher.plan.expr_ir.core import Expr
from batcher.plan.expr_ir.nodes import row_number
from batcher.plan.functions.aggregate import mean as mean_
from batcher.plan.functions.aggregate import sum as sum_
from batcher.plan.functions.metrics.model.classification import positive_mask

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = [
    "hit_rate_at_k",
    "map_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]

_POSITION = "__bt_position"
_LAST = "__bt_last"
_CUM_GAIN = "__bt_cum_gain"
_GAIN = "__bt_gain"
_TIE_GAIN = "__bt_tie_gain"


def _ranked(
    ds: Dataset, query: str, score: str, label: str, positive: Any, *, graded: bool = False
) -> Dataset:
    """One row per candidate with its positions within its query and its relevance gain.

    `_POSITION` is a 1-based row number in descending score order, which is arbitrary
    within a tie; it is only ever used together with `_TIE_GAIN`, the tie group's mean gain,
    so which tied row holds which position cannot change a result. `_LAST` is the tie
    group's last position and `_CUM_GAIN` the gain seen through the end of the group: both
    are ordered window aggregates, whose frame takes in every peer of the current row.
    """
    require_columns(ds, query, score, label)
    ranking = col(score)
    gain = (
        col(label).cast("float64")
        if graded
        else when(positive_mask(col(label), positive)).then(lit(1.0)).otherwise(lit(0.0))
    )
    kept = ds.filter(col(label).is_not_null() & ranking.is_not_null() & ~ranking.is_nan())
    order = [(score, True)]
    return kept.with_columns(**{_GAIN: gain}).with_columns(
        **{
            _POSITION: row_number().over(partition_by=[query], order_by=order),
            _LAST: sum_(lit(1.0)).over(partition_by=[query], order_by=order),
            _CUM_GAIN: sum_(col(_GAIN)).over(partition_by=[query], order_by=order),
            _TIE_GAIN: mean_(col(_GAIN)).over(partition_by=[query, score]),
        }
    )


def _expected_hits(k: int) -> Expr:
    """The expected relevant count in the top `k`, with ties at their average relevance."""
    return sum_(when(col(_POSITION) <= lit(k)).then(col(_TIE_GAIN)).otherwise(lit(0.0)))


def _check_k(k: int) -> None:
    """Reject a cutoff that cannot select anything."""
    if k < 1:
        raise PlanError(f"k must be at least 1, got {k}")


def _mean_over_queries(per_query: Dataset, column: str) -> float:
    """The mean of a per-query value — averaging over queries, never pooling their rows."""
    row = per_query.agg(__bt_mean=col(column).mean()).collect()
    if row.num_rows == 0:
        return float("nan")
    value = row.column("__bt_mean")[0].as_py()
    return float("nan") if value is None else float(value)


def precision_at_k(
    ds: Dataset,
    query: str,
    score: str,
    label: str,
    *,
    k: int = 10,
    positive: Any = 1,
) -> float:
    """Of the top `k` items shown per query, the average fraction that were relevant.

    The metric that matches what a user experiences: they see `k` slots, and this is how
    many of them were worth showing. It ignores relevant items below the cutoff entirely,
    which is correct — nobody scrolled that far.

    Args:
        ds: One row per ``(query, candidate)`` pair.
        query: The column identifying one query, user, or session.
        score: The predicted relevance score, ranked descending.
        label: The true relevance label.
        k: The cutoff — the number of items actually shown.
        positive: The label value that counts as relevant.

    Returns:
        The mean precision at `k` over queries.

    Raises:
        PlanError: If `k` is less than 1.
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import precision_at_k
            >>> ds = bt.from_pydict(
            ...     {"user": ["a", "a", "a", "a"], "s": [0.9, 0.8, 0.2, 0.1],
            ...      "rel": [1, 0, 1, 0]}
            ... )
            >>> precision_at_k(ds, "user", "s", "rel", k=2)
            0.5
    """
    _check_k(k)
    ranked = _ranked(ds, query, score, label, positive)
    per_query = ranked.group_by(query).agg(__bt_value=_expected_hits(k) / lit(float(k)))
    return _mean_over_queries(per_query, "__bt_value")


def recall_at_k(
    ds: Dataset,
    query: str,
    score: str,
    label: str,
    *,
    k: int = 10,
    positive: Any = 1,
) -> float:
    """Of the relevant items a query has, the average fraction that reached the top `k`.

    `precision_at_k`'s complement, and the one that matters when the candidate set is what
    you control: it says whether the retrieval stage is even putting the right items in
    front of the ranker. A query with no relevant items scores 0 and stays in the mean,
    the module-wide convention, so this averages over the same queries as `precision_at_k`.

    Args:
        ds: One row per ``(query, candidate)`` pair.
        query: The column identifying one query, user, or session.
        score: The predicted relevance score, ranked descending.
        label: The true relevance label.
        k: The cutoff.
        positive: The label value that counts as relevant.

    Returns:
        The mean recall at `k` over queries.

    Raises:
        PlanError: If `k` is less than 1.
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import recall_at_k
            >>> ds = bt.from_pydict(
            ...     {"user": ["a", "a", "a", "a"], "s": [0.9, 0.8, 0.2, 0.1],
            ...      "rel": [1, 0, 1, 0]}
            ... )
            >>> recall_at_k(ds, "user", "s", "rel", k=2)
            0.5
    """
    _check_k(k)
    ranked = _ranked(ds, query, score, label, positive)
    per_query = ranked.group_by(query).agg(__bt_hits=_expected_hits(k), __bt_total=sum_(col(_GAIN)))
    with_value = per_query.with_columns(
        __bt_value=when(col("__bt_total") > lit(0.0))
        .then(col("__bt_hits") / col("__bt_total"))
        .otherwise(lit(0.0))
    )
    return _mean_over_queries(with_value, "__bt_value")


def hit_rate_at_k(
    ds: Dataset,
    query: str,
    score: str,
    label: str,
    *,
    k: int = 10,
    positive: Any = 1,
) -> float:
    """The fraction of queries with at least one relevant item in the top `k`.

    The bluntest and often the most honest recommender metric: did the user get *anything*
    useful. It is the number to report when one good result is enough — a search box, a
    "did you mean", a support-article suggestion — because precision is beside the point
    there.

    Args:
        ds: One row per ``(query, candidate)`` pair.
        query: The column identifying one query, user, or session.
        score: The predicted relevance score, ranked descending.
        label: The true relevance label.
        k: The cutoff.
        positive: The label value that counts as relevant.

    Returns:
        The share of queries with a hit in the top `k`, in ``[0, 1]``.

    Raises:
        PlanError: If `k` is less than 1.
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import hit_rate_at_k
            >>> ds = bt.from_pydict(
            ...     {"user": ["a", "a", "b", "b"], "s": [0.9, 0.1, 0.9, 0.1],
            ...      "rel": [1, 0, 0, 1]}
            ... )
            >>> hit_rate_at_k(ds, "user", "s", "rel", k=1)
            0.5
    """
    _check_k(k)
    ranked = _ranked(ds, query, score, label, positive)
    shown_hit = when((col(_LAST) <= lit(float(k))) & (col(_GAIN) > lit(0.0))).then(lit(1.0))
    per_query = ranked.group_by(query).agg(
        __bt_value=when(shown_hit.max() > lit(0.0)).then(lit(1.0)).otherwise(lit(0.0))
    )
    return _mean_over_queries(per_query, "__bt_value")


def mean_reciprocal_rank(
    ds: Dataset,
    query: str,
    score: str,
    label: str,
    *,
    positive: Any = 1,
) -> float:
    """The average of ``1 / rank`` of the first relevant item per query.

    Rewards getting the right answer to position one and falls away sharply after that:
    rank 1 scores 1.0, rank 2 scores 0.5, rank 10 scores 0.1. The right metric when there is
    a single correct answer and the question is how fast the user reaches it. A query with
    no relevant item anywhere contributes 0.

    Args:
        ds: One row per ``(query, candidate)`` pair.
        query: The column identifying one query, user, or session.
        score: The predicted relevance score, ranked descending.
        label: The true relevance label.
        positive: The label value that counts as relevant.

    Returns:
        The mean reciprocal rank in ``[0, 1]``.

    Raises:
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import mean_reciprocal_rank
            >>> ds = bt.from_pydict(
            ...     {"user": ["a", "a", "b", "b"], "s": [0.9, 0.1, 0.9, 0.1],
            ...      "rel": [1, 0, 0, 1]}
            ... )
            >>> mean_reciprocal_rank(ds, "user", "s", "rel")
            0.75
    """
    ranked = _ranked(ds, query, score, label, positive)
    # Irrelevant rows contribute 0, so the per-query max is the first hit's reciprocal rank,
    # or 0 for a query with no relevant item at all.
    reciprocal = when(col(_GAIN) > lit(0.0)).then(lit(1.0) / col(_LAST)).otherwise(lit(0.0))
    per_query = ranked.group_by(query).agg(__bt_value=reciprocal.max())
    return _mean_over_queries(per_query, "__bt_value")


def ndcg_at_k(
    ds: Dataset,
    query: str,
    score: str,
    label: str,
    *,
    k: int = 10,
    positive: Any = 1,
    graded: bool = False,
) -> float:
    """Normalized discounted cumulative gain at `k` — position-weighted relevance.

    The metric that knows position 1 is worth more than position 5. The item at rank ``r``
    contributes its gain times ``1 / log2(r + 1)``, and the total is divided by the best
    score the query's own items could have achieved, so a query with two relevant items is
    not penalised against one with ten. This is scikit-learn's ``ndcg_score`` (linear gain,
    ties averaged), and agrees with it on binary and graded labels alike.

    By default relevance is binary: the gain is 1 where `label` equals `positive` and 0
    elsewhere. With ``graded=True`` the label itself is the gain, so a label of 3 counts
    three times a label of 1; `positive` is then ignored.

    Args:
        ds: One row per ``(query, candidate)`` pair.
        query: The column identifying one query, user, or session.
        score: The predicted relevance score, ranked descending.
        label: The true relevance label, or the non-negative gain when `graded`.
        k: The cutoff.
        positive: The label value that counts as relevant, for binary relevance.
        graded: Whether `label` is a graded gain rather than a relevant/not-relevant value.

    Returns:
        The mean NDCG at `k` over queries, in ``[0, 1]``. A query with no relevant item
        has an ideal gain of zero and scores 0, as in ``ndcg_score``.

    Raises:
        PlanError: If `k` is less than 1.
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import ndcg_at_k
            >>> ds = bt.from_pydict(
            ...     {"user": ["a", "a"], "s": [0.9, 0.1], "rel": [1, 0]}
            ... )
            >>> ndcg_at_k(ds, "user", "s", "rel", k=2)
            1.0
            >>> graded = bt.from_pydict(
            ...     {"user": ["a"] * 4, "s": [0.9, 0.8, 0.7, 0.6], "rel": [1, 3, 0, 2]}
            ... )
            >>> round(ndcg_at_k(graded, "user", "s", "rel", k=4, graded=True), 4)
            0.7884
    """
    _check_k(k)
    ranked = _ranked(ds, query, score, label, positive, graded=graded)
    # The ideal ranking sorts the query's own gains descending; a tie there cannot change
    # the ideal gain, so an arbitrary row number is exact.
    ideal = ranked.with_columns(
        __bt_ideal=row_number().over(partition_by=[query], order_by=[(_GAIN, True)])
    )
    per_query = ideal.group_by(query).agg(
        __bt_dcg=sum_(
            when(col(_POSITION) <= lit(k))
            .then(col(_TIE_GAIN) * _discount(col(_POSITION)))
            .otherwise(lit(0.0))
        ),
        __bt_idcg=sum_(
            when(col("__bt_ideal") <= lit(k))
            .then(col(_GAIN) * _discount(col("__bt_ideal")))
            .otherwise(lit(0.0))
        ),
    )
    scored = per_query.with_columns(
        __bt_value=when(col("__bt_idcg") > lit(0.0))
        .then(col("__bt_dcg") / col("__bt_idcg"))
        .otherwise(lit(0.0))
    )
    return _mean_over_queries(scored, "__bt_value")


def _discount(position: Expr) -> Expr:
    """The DCG position discount ``1 / log2(position + 1)``."""
    return lit(math.log(2.0)) / (position.cast("float64") + lit(1.0)).ln()


def map_at_k(
    ds: Dataset,
    query: str,
    score: str,
    label: str,
    *,
    k: int = 10,
    positive: Any = 1,
) -> float:
    """Mean average precision at `k` — the rank-aware quality of a recommendation list.

    Where `precision_at_k` counts how many of the top `k` were relevant, MAP@k also rewards
    putting them *high*: for each query it averages the precision measured at every relevant
    position in the top `k`, then averages that over queries. A relevant item at rank 1
    contributes far more than the same item at rank `k`, which is what makes MAP the standard
    single number for a ranked-retrieval or recommender system.

    The average precision for a query divides by ``min(k, R)`` where ``R`` is the query's total
    relevant count, so a query with fewer than `k` relevant items is not penalized for the slots
    it could never fill. A tied group is scored at its last position, so it counts as one
    threshold the way `average_precision` counts it.

    Args:
        ds: One row per ``(query, candidate)`` pair.
        query: The column identifying one query, user, or session.
        score: The predicted relevance score, ranked descending.
        label: The true relevance label.
        k: The cutoff — the number of items actually shown.
        positive: The label value that counts as relevant.

    Returns:
        The mean average precision at `k` over queries. A query with no relevant item scores
        0 and stays in the mean, the same query set every metric here averages over.

    Raises:
        PlanError: If `k` is less than 1.
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import map_at_k
            >>> ds = bt.from_pydict(
            ...     {"user": ["a", "a", "a", "a"], "s": [0.9, 0.8, 0.2, 0.1],
            ...      "rel": [1, 0, 1, 0]}
            ... )
            >>> round(map_at_k(ds, "user", "s", "rel", k=4), 4)
            0.8333
    """
    _check_k(k)
    ranked = _ranked(ds, query, score, label, positive)
    shown = (col(_GAIN) > lit(0.0)) & (col(_LAST) <= lit(float(k)))
    contribution = when(shown).then(col(_CUM_GAIN) / col(_LAST)).otherwise(lit(0.0))
    per_query = ranked.group_by(query).agg(__bt_sum=sum_(contribution), __bt_rel=sum_(col(_GAIN)))
    cap = when(col("__bt_rel") < lit(float(k))).then(col("__bt_rel")).otherwise(lit(float(k)))
    per_query = per_query.with_columns(
        __bt_ap=when(cap > lit(0.0)).then(col("__bt_sum") / cap).otherwise(lit(0.0))
    )
    return _mean_over_queries(per_query, "__bt_ap")
