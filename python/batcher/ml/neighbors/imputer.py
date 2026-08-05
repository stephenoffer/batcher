"""`KNNImputer` — fill a gap with what similar rows had there.

`SimpleImputer` fills with one number for the whole column; `IterativeImputer` fits a model
per column and so assumes the relationship is one a linear model can state. This assumes
neither. It finds the training rows most like the one with the gap, using the columns that
*are* present, and averages what they had in the missing one.

That makes it the right choice when the columns relate locally rather than globally — a
missing price in a row whose neighbourhood, size and age are known is much better described
by similar properties than by the national mean or by a single regression through everything.

Like the k-NN estimators, the reference set is bounded and folded into the transform as
literals, so the fill is one expression over the feature columns rather than a join.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.neighbors.reference import (
    MAX_REFERENCE_ROWS,
    balanced_sum,
    drop_staging,
    neighbour_weights,
    read_reference,
    stage_distances,
)
from batcher.ml.preprocessors.base import Preprocessor, columns_arg
from batcher.plan.expr_ir import col, lit, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["KNNImputer"]


class KNNImputer(Preprocessor):
    """Fill each missing value with the weighted mean of its nearest complete rows.

    scikit-learn's ``KNNImputer``. `fit` keeps the rows that are complete across `columns` as
    the reference set; `transform` measures each row against them and fills any gap with what
    the neighbours had.

    The distance is measured over the columns that are present in the row being filled — a
    row missing its price is matched on size and age, and never on the price it does not
    have. That is why this is worth more than a column mean, and it is also why the reference
    set has to be complete: a neighbour with its own gaps could not supply the answer.

    Scale the columns first. Distance treats them alike, so a column in millions decides
    every neighbour and one in fractions is ignored.

    One deliberate difference from scikit-learn: a donor must be complete across `columns`.
    scikit-learn lets a row with its own gaps donate, measuring distance over the coordinates
    the two rows share and rescaling for the ones they do not. Requiring a complete donor is
    simpler to reason about and keeps the transform a single expression; the two agree
    wherever the neighbourhood is unambiguous and can pick different donors when neighbours
    are effectively tied.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import KNNImputer
            >>> ds = bt.from_pydict(
            ...     {"size": [10.0, 11.0, 50.0, 51.0, 10.5],
            ...      "price": [1.0, 1.2, 9.0, 9.4, None]}
            ... )
            >>> out = KNNImputer(["size", "price"], k=2).fit_transform(ds)
            >>> round(out.to_pydict()["price"][4], 3)
            1.1

    Args:
        columns: The numeric columns to impute, and to measure distance over.
        k: How many neighbours to average.
        weights: ``"uniform"``, or ``"distance"`` to weight a closer row more heavily.
        max_reference: The ceiling on the reference set.
    """

    __slots__ = ("columns", "k", "max_reference", "points_", "weights")

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        k: int = 5,
        weights: str = "uniform",
        max_reference: int = MAX_REFERENCE_ROWS,
    ) -> None:
        self.columns = columns_arg(columns, what="KNNImputer")
        if len(self.columns) < 2:
            raise PlanError(
                "KNNImputer needs at least two columns: it matches a row on the columns that "
                "are present to fill the one that is not, so a single column has nothing to "
                "match on. Use SimpleImputer."
            )
        if k < 1:
            raise PlanError(f"KNNImputer: k must be at least 1, got {k}")
        self.k = k
        if weights not in ("uniform", "distance"):
            raise PlanError(f"KNNImputer: weights must be 'uniform' or 'distance', got {weights!r}")
        self.weights = weights
        self.max_reference = max_reference
        self.points_: list[list[float]] = []

    def fit(self, ds: Dataset) -> KNNImputer:
        """Keep the rows that are complete across `columns` as the reference set.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import KNNImputer
                >>> ds = bt.from_pydict({"a": [1.0, 2.0, None], "b": [1.0, 2.0, 3.0]})
                >>> len(KNNImputer(["a", "b"], k=1).fit(ds).points_)
                2

        Args:
            ds: The training data.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If no row is complete, or the reference set exceeds `max_reference`.
        """
        points, _ = read_reference(
            ds, self.columns, [], what="KNNImputer", limit=self.max_reference
        )
        self.points_ = points
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Fill each column's nulls from the neighbours, leaving present values alone.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import KNNImputer
                >>> train = bt.from_pydict({"a": [1.0, 9.0], "b": [1.0, 9.0]})
                >>> pre = KNNImputer(["a", "b"], k=1).fit(train)
                >>> pre.transform(bt.from_pydict({"a": [8.5], "b": [None]})).to_pydict()["b"]
                [9.0]

        Args:
            ds: The dataset to fill.

        Returns:
            A new lazy `Dataset` with the gaps filled.
        """
        self._require_fitted()
        out = ds
        for index, name in enumerate(self.columns):
            # Distance is measured over the *other* columns: the one being filled is the
            # unknown, so including it would compare a null against the reference values.
            others = [c for c in self.columns if c != name]
            positions = [i for i, c in enumerate(self.columns) if c != name]
            points = [[row[i] for i in positions] for row in self.points_]
            # One staging pass per column: each measures distance over a different feature
            # set, so the distances cannot be shared between them.
            staged = stage_distances(out, others, points, self.k)
            weights, total = neighbour_weights(
                len(points), distance_weighted=self.weights == "distance"
            )
            numerator = balanced_sum(
                [w * lit(float(row[index])) for w, row in zip(weights, self.points_, strict=True)]
            )
            estimate = numerator / total
            filled = when(col(name).is_null()).then(estimate).otherwise(col(name).cast("float64"))
            out = drop_staging(staged.with_columns(**{name: filled}))
        return out
