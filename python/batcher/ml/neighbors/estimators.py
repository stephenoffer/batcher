"""`KNeighborsClassifier` and `KNeighborsRegressor` — prediction by local averaging.

The model that assumes nothing about the shape of the relationship: to predict a row, find
the training rows most like it and average what happened to them. That makes it the natural
first check on whether a problem has any local structure at all, and the right tool when the
boundary is genuinely irregular.

Both fold the reference set into the prediction as literals, so `predict` is one arithmetic
expression over the feature columns — no join, no shuffle, and identical on one core or a
hundred. The cost is that the reference set is bounded; see `neighbors.reference` for why,
and `batcher.ml.build_vector_index` for the approximate route when it is genuinely large.

Scale the features first. Distance treats every column alike, so a column measured in
millions decides every neighbour and one measured in fractions is ignored — which is not a
modelling choice anybody made.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml._estimator import argmax_prediction, require_fitted
from batcher.ml.neighbors.reference import (
    MAX_REFERENCE_ROWS,
    balanced_sum,
    drop_staging,
    neighbour_weights,
    read_reference,
    stage_distances,
)
from batcher.plan.expr_ir import Expr, lit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["KNeighborsClassifier", "KNeighborsRegressor"]

_WEIGHTS = ("uniform", "distance")


class _KNeighbors:
    """What the two k-NN estimators share: the reference set, and the weight expressions."""

    __slots__ = (
        "features",
        "k",
        "max_reference",
        "output_column",
        "points_",
        "target",
        "targets_",
        "weights",
    )

    def __init__(
        self,
        features: Sequence[str],
        target: str,
        *,
        k: int = 5,
        weights: str = "uniform",
        output_column: str = "prediction",
        max_reference: int = MAX_REFERENCE_ROWS,
    ) -> None:
        what = type(self).__name__
        self.features = list(features)
        if not self.features:
            raise PlanError(f"{what} needs at least one feature column.")
        self.target = target
        if k < 1:
            raise PlanError(f"{what}: k must be at least 1, got {k}")
        self.k = k
        if weights not in _WEIGHTS:
            raise PlanError(f"{what}: weights must be 'uniform' or 'distance', got {weights!r}")
        self.weights = weights
        self.output_column = output_column
        self.max_reference = max_reference
        self.points_: list[list[float]] = []
        self.targets_: list[Any] = []

    def fit(self, ds: Dataset) -> Any:
        """Keep the reference set. A k-NN model has no parameters to estimate.

        Rows with a null feature are dropped, because a distance to them is undefined; a row
        with a null *target* is dropped too, since it could never contribute an answer.

        Args:
            ds: The training data.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If nothing usable remains, or the reference set exceeds
                `max_reference`.
        """
        from batcher.plan.expr_ir import col

        usable = ds.filter(col(self.target).is_not_null())
        points, carried = read_reference(
            usable,
            self.features,
            [self.target],
            what=type(self).__name__,
            limit=self.max_reference,
        )
        self.points_ = points
        self.targets_ = carried[self.target]
        return self

    def _staged(self, ds: Dataset) -> tuple[Dataset, list[Expr], Expr]:
        """`ds` with the distance columns attached, plus the weight expressions over them."""
        require_fitted(self, self.points_)
        staged = stage_distances(ds, self.features, self.points_, self.k)
        weights, total = neighbour_weights(
            len(self.points_), distance_weighted=self.weights == "distance"
        )
        return staged, weights, total


class KNeighborsRegressor(_KNeighbors):
    """Predict a number by averaging the `k` nearest training rows' targets.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import KNeighborsRegressor
            >>> train = bt.from_pydict({"x": [0.0, 1.0, 10.0, 11.0], "y": [1.0, 1.0, 9.0, 9.0]})
            >>> model = KNeighborsRegressor(["x"], "y", k=2).fit(train)
            >>> model.predict(bt.from_pydict({"x": [0.5]})).to_pydict()["prediction"]
            [1.0]

    Args:
        features: The numeric columns distance is measured over.
        target: The numeric column to average.
        k: How many neighbours to average.
        weights: ``"uniform"``, or ``"distance"`` to weight a closer row more heavily.
        output_column: The name of the appended prediction column.
        max_reference: The ceiling on the reference set.
    """

    __slots__ = ()

    def predict(self, ds: Dataset) -> Dataset:
        """Append the neighbour-weighted mean target for each row, lazily.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import KNeighborsRegressor
                >>> train = bt.from_pydict({"x": [0.0, 1.0, 10.0], "y": [2.0, 4.0, 20.0]})
                >>> model = KNeighborsRegressor(["x"], "y", k=1).fit(train)
                >>> model.predict(bt.from_pydict({"x": [9.5]})).to_pydict()["prediction"]
                [20.0]

        Args:
            ds: The dataset to score.

        Returns:
            A new lazy `Dataset` with the prediction column appended.
        """
        staged, weights, total = self._staged(ds)
        numerator = balanced_sum(
            [
                weight * lit(float(value))
                for weight, value in zip(weights, self.targets_, strict=True)
            ]
        )
        return drop_staging(staged.with_columns(**{self.output_column: numerator / total}))


class KNeighborsClassifier(_KNeighbors):
    """Predict a label by voting among the `k` nearest training rows.

    Ties between classes resolve to whichever label sorts first, so the answer is
    reproducible rather than dependent on the order the reference rows arrived in.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import KNeighborsClassifier
            >>> train = bt.from_pydict(
            ...     {"x": [0.0, 1.0, 10.0, 11.0], "label": ["low", "low", "high", "high"]}
            ... )
            >>> model = KNeighborsClassifier(["x"], "label", k=2).fit(train)
            >>> model.predict(bt.from_pydict({"x": [0.5, 10.5]})).to_pydict()["prediction"]
            ['low', 'high']

    Args:
        features: The numeric columns distance is measured over.
        target: The label column to vote on.
        k: How many neighbours vote.
        weights: ``"uniform"``, or ``"distance"`` to weight a closer row more heavily.
        output_column: The name of the appended prediction column.
        max_reference: The ceiling on the reference set.
    """

    __slots__ = ()

    @property
    def classes_(self) -> list[Any]:
        """The labels seen in the training data, sorted.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import KNeighborsClassifier
                >>> train = bt.from_pydict({"x": [0.0, 1.0], "label": ["b", "a"]})
                >>> KNeighborsClassifier(["x"], "label", k=1).fit(train).classes_
                ['a', 'b']

        Returns:
            The distinct training labels, in a stable order.
        """
        return sorted({t for t in self.targets_ if t is not None}, key=repr)

    def predict(self, ds: Dataset) -> Dataset:
        """Append the winning label for each row, lazily.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import KNeighborsClassifier
                >>> train = bt.from_pydict({"x": [0.0, 9.0], "label": ["a", "b"]})
                >>> model = KNeighborsClassifier(["x"], "label", k=1).fit(train)
                >>> model.predict(bt.from_pydict({"x": [8.0]})).to_pydict()["prediction"]
                ['b']

        Args:
            ds: The dataset to score.

        Returns:
            A new lazy `Dataset` with the prediction column appended.
        """
        staged, weights, _ = self._staged(ds)
        labels = self.classes_

        def votes_for(label: Any) -> Expr:
            return balanced_sum(
                [w for w, value in zip(weights, self.targets_, strict=True) if value == label]
            )

        scored = staged.with_columns(**{self.output_column: argmax_prediction(labels, votes_for)})
        return drop_staging(scored)
