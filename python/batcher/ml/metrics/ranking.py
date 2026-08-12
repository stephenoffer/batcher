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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.stats._shared import require_columns
from batcher.plan.expr_ir.constructors import col, lit, when
from batcher.plan.expr_ir.nodes import row_number
from batcher.plan.functions.aggregate import count_if
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

_RANK = "__bt_rank"
_RELEVANT = "__bt_relevant"


def _ranked(ds: Dataset, query: str, score: str, label: str, positive: Any) -> Dataset:
    """One row per candidate, carrying its 1-based rank within its query and a 0/1 relevance."""
    require_columns(ds, query, score, label)
    return ds.with_columns(
        **{
            _RANK: row_number().over(partition_by=[query], order_by=[(score, True)]),
            _RELEVANT: when(positive_mask(col(label), positive)).then(lit(1.0)).otherwise(lit(0.0)),
        }
    )


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
    per_query = ranked.group_by(query).agg(
        __bt_value=sum_(when(col(_RANK) <= lit(k)).then(col(_RELEVANT)).otherwise(lit(0.0)))
        / lit(float(k))
    )
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
    front of the ranker. A query with no relevant items is excluded rather than scored 0,
    since there was nothing to find.

    Args:
        ds: One row per ``(query, candidate)`` pair.
        query: The column identifying one query, user, or session.
        score: The predicted relevance score, ranked descending.
        label: The true relevance label.
        k: The cutoff.
        positive: The label value that counts as relevant.

    Returns:
        The mean recall at `k` over queries that have at least one relevant item.

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
    per_query = ranked.group_by(query).agg(
        __bt_hits=sum_(when(col(_RANK) <= lit(k)).then(col(_RELEVANT)).otherwise(lit(0.0))),
        __bt_total=sum_(col(_RELEVANT)),
    )
    with_value = per_query.filter(col("__bt_total") > lit(0.0)).with_columns(
        __bt_value=col("__bt_hits") / col("__bt_total")
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
    per_query = ranked.group_by(query).agg(
        __bt_value=when(count_if((col(_RANK) <= lit(k)) & (col(_RELEVANT) == lit(1.0))) > lit(0))
        .then(lit(1.0))
        .otherwise(lit(0.0))
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
    relevant_only = ranked.filter(col(_RELEVANT) == lit(1.0))
    best = relevant_only.group_by(query).agg(__bt_first=col(_RANK).min())
    # A query with no relevant item is absent from `best` and must still count as 0, so the
    # denominator is the number of *queries*, not the number of queries that had a hit.
    queries = ds.select(query).distinct().count()
    if queries == 0:
        return float("nan")
    row = best.agg(__bt_total=sum_(lit(1.0) / col("__bt_first").cast("float64"))).collect()
    total = row.column("__bt_total")[0].as_py() if row.num_rows else None
    return 0.0 if total is None else float(total) / queries


def ndcg_at_k(
    ds: Dataset,
    query: str,
    score: str,
    label: str,
    *,
    k: int = 10,
    positive: Any = 1,
) -> float:
    """Normalized discounted cumulative gain at `k` — position-weighted relevance.

    The metric that knows position 1 is worth more than position 5. Each relevant item
    contributes ``1 / log2(rank + 1)``, and the total is divided by the best score the
    query's own relevant items could have achieved — so a query with two relevant items is
    not penalised against one with ten.

    Binary relevance: an item is relevant or it is not. Graded relevance would need the
    label as a gain value, which is a different (and much less common) input shape.

    Args:
        ds: One row per ``(query, candidate)`` pair.
        query: The column identifying one query, user, or session.
        score: The predicted relevance score, ranked descending.
        label: The true relevance label.
        k: The cutoff.
        positive: The label value that counts as relevant.

    Returns:
        The mean NDCG at `k` over queries with at least one relevant item, in ``[0, 1]``.

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
    """
    _check_k(k)
    import math

    ranked = _ranked(ds, query, score, label, positive)
    # The ideal ranking puts every relevant item at the top, so the ideal gain depends only
    # on *how many* relevant items the query has — a second rank over the label column.
    ideal = ranked.with_columns(
        __bt_ideal_rank=row_number().over(partition_by=[query], order_by=[(_RELEVANT, True)])
    )
    gain = lit(1.0) / (col(_RANK).cast("float64") + lit(1.0)).ln() * lit(math.log(2.0))
    ideal_gain = (
        lit(1.0) / (col("__bt_ideal_rank").cast("float64") + lit(1.0)).ln() * lit(math.log(2.0))
    )
    per_query = ideal.group_by(query).agg(
        __bt_dcg=sum_(
            when((col(_RANK) <= lit(k)) & (col(_RELEVANT) == lit(1.0)))
            .then(gain)
            .otherwise(lit(0.0))
        ),
        __bt_idcg=sum_(
            when((col("__bt_ideal_rank") <= lit(k)) & (col(_RELEVANT) == lit(1.0)))
            .then(ideal_gain)
            .otherwise(lit(0.0))
        ),
    )
    scored = per_query.filter(col("__bt_idcg") > lit(0.0)).with_columns(
        __bt_value=col("__bt_dcg") / col("__bt_idcg")
    )
    return _mean_over_queries(scored, "__bt_value")


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
    it could never fill.

    Args:
        ds: One row per ``(query, candidate)`` pair.
        query: The column identifying one query, user, or session.
        score: The predicted relevance score, ranked descending.
        label: The true relevance label.
        k: The cutoff — the number of items actually shown.
        positive: The label value that counts as relevant.

    Returns:
        The mean average precision at `k` over queries.

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
    ranked = ranked.with_columns(__bt_total_rel=sum_(col(_RELEVANT)).over(partition_by=[query]))
    top = ranked.filter(col(_RANK) <= lit(k))
    top = top.with_columns(
        __bt_cum=sum_(col(_RELEVANT)).over(partition_by=[query], order_by=[_RANK])
    )
    contribution = (col("__bt_cum") / col(_RANK)) * col(_RELEVANT)
    per_query = top.group_by(query).agg(
        __bt_sum=sum_(contribution),
        __bt_rel=col("__bt_total_rel").max(),
    )
    cap = when(col("__bt_rel") < lit(float(k))).then(col("__bt_rel")).otherwise(lit(float(k)))
    per_query = per_query.with_columns(
        __bt_ap=when(cap > lit(0.0)).then(col("__bt_sum") / cap).otherwise(lit(0.0))
    )
    return _mean_over_queries(per_query, "__bt_ap")
