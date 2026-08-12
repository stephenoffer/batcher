"""The nearest-neighbour primitive the k-NN estimators and imputer share.

A k-NN model has no parameters: it *is* its training data. Fitting therefore means keeping a
reference set, and predicting means measuring every new row against it. That shape is hostile
to a distributed engine — a join of every row against every reference row is quadratic — so
the design here bounds the one side that can be bounded.

The reference set is capped, held on the driver, and folded into the prediction as
**literals**, exactly the way a fitted linear model folds in its coefficients. What reaches
the engine is then an ordinary arithmetic tree over the feature columns: no join, no shuffle,
no per-row Python, and the same expression whether it runs on one core or a hundred.

Selecting the k nearest without an argsort is the one trick worth explaining. Sorting the
distances and reading the k-th gives a *threshold*, and every reference row at or under that
threshold is a neighbour. That is two passes over the same bounded set of distance
expressions, and it stays inside the expression language, where an argsort would not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.stats._shared import require_columns
from batcher.plan.expr_ir import Expr, array, col, lit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = [
    "MAX_REFERENCE_ROWS",
    "balanced_sum",
    "drop_staging",
    "neighbour_weights",
    "read_reference",
    "squared_distance",
    "stage_distances",
]

#: The ceiling on how many reference rows a k-NN model may keep. Exact k-NN costs one
#: distance per (scored row x reference row), and no amount of expression tidying changes
#: that: measured here, scoring the reference set against itself takes ~0.4s at 200 rows,
#: ~4s at 1,000 and ~16s at 2,000. 1,000 is the point where the default is still an
#: interactive query. Raise it deliberately, or reach for an approximate index
#: (`batcher.ml.build_vector_index`) for a corpus that is genuinely large.
MAX_REFERENCE_ROWS = 1000

#: The staged per-row distance list, and the k-th smallest of it. Both are attached as
#: columns so the prediction stays linear in the reference size; see `stage_distances`.
DISTANCE_COLUMN = "__bt_knn_distances"
THRESHOLD_COLUMN = "__bt_knn_threshold"


def read_reference(
    ds: Dataset, features: Sequence[str], extra: Sequence[str], *, what: str, limit: int
) -> tuple[list[list[float]], dict[str, list[Any]]]:
    """Read the bounded reference set to the driver, dropping rows with a null feature.

    Args:
        ds: The training data.
        features: The columns distance is measured over.
        extra: Further columns to carry along, such as the target.
        what: The caller's class name, for error messages.
        limit: The ceiling on how many rows to keep.

    Returns:
        ``(points, carried)`` — one feature vector per reference row, and the `extra`
        columns as lists aligned with it.

    Raises:
        PlanError: If the reference set is empty or larger than `limit`.
        ColumnNotFoundError: If a named column is missing.
    """
    names = [*features, *extra]
    require_columns(ds, *names, hint=f"{what} needs this column.")
    kept = ds.select(*names)
    for name in features:
        kept = kept.filter(col(name).is_not_null())
    table = kept.limit(limit + 1).collect()
    if table.num_rows == 0:
        raise PlanError(
            f"{what}: the training data has no rows with every feature present, so there are "
            "no neighbours to measure against. Impute the features first."
        )
    if table.num_rows > limit:
        raise PlanError(
            f"{what}: the training data has more than {limit} usable rows. A k-NN model keeps "
            "its training set and folds it into the prediction expression, so the plan grows "
            "with it. Sample the reference set down, raise max_reference, or use an "
            "approximate index (batcher.ml.build_vector_index) for a genuinely large corpus."
        )
    columns = {name: table.column(name).to_pylist() for name in names}
    points = [[float(columns[name][i]) for name in features] for i in range(table.num_rows)]
    return points, {name: columns[name] for name in extra}


def squared_distance(features: Sequence[str], point: Sequence[float]) -> Expr:
    """The squared Euclidean distance from each row to one reference `point`.

    Squared rather than Euclidean on purpose: the square root is monotone, so it cannot
    change which neighbours are nearest, and leaving it out keeps the expression cheaper and
    exact on integers.

    Args:
        features: The feature columns, in the order the point's values are given.
        point: One reference row's feature values.

    Returns:
        An expression evaluating to the squared distance for each row.
    """
    gaps = []
    for name, value in zip(features, point, strict=True):
        gap = col(name).cast("float64") - lit(float(value))
        gaps.append(gap * gap)
    return balanced_sum(gaps)


def stage_distances(
    ds: Dataset,
    features: Sequence[str],
    points: Sequence[Sequence[float]],
    k: int,
    *,
    tie_break: bool = False,
) -> Dataset:
    """Attach each row's distances to every reference point, and its k-th smallest.

    Both are *materialized as columns* rather than left as subexpressions, and that is the
    difference between this being usable and not. The k-th distance is a function of all `n`
    distances, so an inline threshold makes every one of the `n` weight expressions carry a
    copy of the whole distance set — an O(n^2) tree, which measured at 7 seconds for a
    40-row reference set and would be four million nodes at the default cap.

    Staged, the distances are computed once into a list column and the threshold reads that
    column, so the whole prediction is O(n).

    Args:
        ds: The dataset being scored.
        features: The feature columns.
        points: The reference rows' feature values.
        k: How many neighbours to use.
        tie_break: Make equal distances strictly ordered by reference index, for a caller
            that must select exactly one neighbour rather than weight all the tied ones.

    Returns:
        `ds` with the distance-list and threshold helper columns attached.
    """
    built = [squared_distance(features, point) for point in points]
    if tie_break:
        # Nudge each distance by a relative amount proportional to its reference index, so
        # equal distances become a strict order broken by index. A caller that needs to pick
        # *one* nearest neighbour cannot use ties: averaging two symmetric neighbours returns
        # the point between them, which is the base row itself.
        built = [d * lit(1.0 + index * 1e-12) for index, d in enumerate(built)]
    distances = array(*built)
    staged = ds.with_columns(**{DISTANCE_COLUMN: distances})
    # `k` may exceed the reference set, in which case every row is a neighbour and the
    # threshold is simply the largest distance.
    position = min(k, len(points)) - 1
    return staged.with_columns(
        **{THRESHOLD_COLUMN: col(DISTANCE_COLUMN).list.sort().list.get(position)}
    )


def neighbour_weights(count: int, *, distance_weighted: bool) -> tuple[list[Expr], Expr]:
    """One weight expression per reference row, non-zero only for the k nearest.

    Reads the columns `stage_distances` attached, so each weight is O(1) in the size of the
    reference set rather than carrying its own copy of every distance.

    Ties at the boundary all count, so a row can have more than `k` neighbours when several
    sit at exactly the same distance — the honest answer, where the alternative would be to
    break the tie by reference-set order and make the prediction depend on the order rows
    happened to arrive in.

    Args:
        count: How many reference rows there are.
        distance_weighted: Weight each neighbour by ``1 / distance`` rather than equally, so
            a closer row counts for more.

    Returns:
        ``(weights, total)`` — one weight expression per reference row, and their sum.
    """
    weights: list[Expr] = []
    for index in range(count):
        distance = col(DISTANCE_COLUMN).list.get(index)
        inside = (distance <= col(THRESHOLD_COLUMN)).cast("float64")
        if distance_weighted:
            # A reference row sitting exactly on the scored row would divide by zero, so the
            # denominator is floored. The floor is far below any distance that matters, so a
            # coincident row dominates the average, which is what "distance weighted" means.
            inside = inside / (distance + lit(1e-12))
        weights.append(inside)
    return weights, balanced_sum(weights)


def balanced_sum(terms: Sequence[Expr]) -> Expr:
    """Add `terms` as a balanced tree rather than a left-nested chain.

    ``a + b + c + ...`` builds a tree as deep as the term count, and every walker over the
    plan recurses once per level — so a thousand reference rows overflowed Python's stack
    inside `contains_aggregate` before the engine saw the query at all. Summing pairwise
    makes the depth logarithmic, which is the difference between a 2,000-row reference set
    being expressible and not.

    Args:
        terms: The expressions to add.

    Returns:
        Their sum, as an expression of logarithmic depth.
    """
    if not terms:
        return lit(0.0)
    level = list(terms)
    while len(level) > 1:
        level = [
            level[i] + level[i + 1] if i + 1 < len(level) else level[i]
            for i in range(0, len(level), 2)
        ]
    return level[0]


def drop_staging(ds: Dataset) -> Dataset:
    """Remove the helper columns `stage_distances` attached.

    Args:
        ds: The dataset carrying them.

    Returns:
        `ds` without the helper columns.
    """
    return ds.drop(DISTANCE_COLUMN, THRESHOLD_COLUMN)
