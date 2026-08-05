"""Baseline predictors — the "does my model beat doing nothing" reference.

Every model score needs a floor to be read against: an R² of 0.6 is only impressive if predicting
the mean scores 0, and an accuracy of 0.9 is worthless if 90% of rows are one class. These
predictors *are* that floor — they ignore the features and always predict the target's central
tendency or most common class — so scoring them on the same split gives the honest baseline every
real model must clear. They match scikit-learn's ``DummyRegressor`` and ``DummyClassifier``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.constructors import col, lit

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["DummyClassifier", "DummyRegressor"]

_REGRESSOR_STRATEGIES = ("mean", "median")
_CLASSIFIER_STRATEGIES = ("most_frequent",)


class DummyRegressor:
    """Predict a constant — the target's mean or median — ignoring the features.

    The regression baseline: `strategy="mean"` minimizes squared error, `strategy="median"`
    minimizes absolute error, and either is the number a real regressor must beat. Reproduces
    scikit-learn's ``DummyRegressor``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.dummy import DummyRegressor
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
            >>> DummyRegressor("y").fit(ds).constant_
            20.0

    Args:
        target: The column to predict.
        strategy: ``"mean"`` (the default) or ``"median"``.
        output_column: The name of the prediction column `predict` appends.
    """

    __slots__ = ("constant_", "output_column", "strategy", "target")

    def __init__(
        self, target: str, *, strategy: str = "mean", output_column: str = "prediction"
    ) -> None:
        if not isinstance(target, str):
            # A baseline takes the target column alone - it has no features by definition -
            # so a list here is the caller writing it like every other estimator. Left
            # alone it reached a dict lookup and raised ``TypeError: unhashable type:
            # 'list'`` from inside fit, which names neither the argument nor the shape.
            raise PlanError(
                f"DummyRegressor takes the name of one target column, not "
                f"{target!r}. A baseline has no features, so there is no "
                "feature list to pass."
            )
        if strategy not in _REGRESSOR_STRATEGIES:
            raise PlanError(
                f"strategy must be one of {sorted(_REGRESSOR_STRATEGIES)}, got {strategy!r}."
            )
        self.target = target
        self.strategy = strategy
        self.output_column = output_column
        self.constant_: float = 0.0

    def fit(self, ds: Dataset) -> DummyRegressor:
        """Learn the constant to predict — the target's mean or median.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.dummy import DummyRegressor
                >>> DummyRegressor("y", strategy="median").fit(
                ...     bt.from_pydict({"y": [1.0, 2.0, 100.0]})
                ... ).constant_
                2.0

        Args:
            ds: The training dataset.

        Returns:
            ``self``, fitted.
        """
        aggregate = (
            col(self.target).median() if self.strategy == "median" else col(self.target).mean()
        )
        self.constant_ = float(ds.agg(c=aggregate).collect().column("c")[0].as_py())
        return self

    def predict(self, ds: Dataset) -> Dataset:
        """Append the learned constant as the prediction for every row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.dummy import DummyRegressor
                >>> ds = bt.from_pydict({"y": [2.0, 4.0, 6.0]})
                >>> DummyRegressor("y").fit(ds).predict(ds).to_pydict()["prediction"]
                [4.0, 4.0, 4.0]

        Args:
            ds: The dataset to score.

        Returns:
            A new lazy `Dataset` with the constant prediction appended.
        """
        return ds.with_columns(**{self.output_column: lit(self.constant_)})


class DummyClassifier:
    """Predict the most frequent class, ignoring the features.

    The classification baseline: it always predicts the majority class, which is exactly the trap a
    high accuracy hides on an imbalanced target — if 95% of rows are negative, this scores 0.95
    while learning nothing. A real classifier must beat it on a *balanced* metric. Reproduces
    scikit-learn's ``DummyClassifier`` with ``strategy="most_frequent"``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.dummy import DummyClassifier
            >>> ds = bt.from_pydict({"y": ["a", "a", "b"]})
            >>> DummyClassifier("y").fit(ds).constant_
            'a'

    Args:
        target: The class label column.
        strategy: ``"most_frequent"`` (the only supported strategy).
        output_column: The name of the prediction column `predict` appends.
    """

    __slots__ = ("constant_", "output_column", "strategy", "target")

    def __init__(
        self, target: str, *, strategy: str = "most_frequent", output_column: str = "prediction"
    ) -> None:
        if not isinstance(target, str):
            # A baseline takes the target column alone - it has no features by definition -
            # so a list here is the caller writing it like every other estimator. Left
            # alone it reached a dict lookup and raised ``TypeError: unhashable type:
            # 'list'`` from inside fit, which names neither the argument nor the shape.
            raise PlanError(
                f"DummyClassifier takes the name of one target column, not "
                f"{target!r}. A baseline has no features, so there is no "
                "feature list to pass."
            )
        if strategy not in _CLASSIFIER_STRATEGIES:
            raise PlanError(
                f"strategy must be one of {sorted(_CLASSIFIER_STRATEGIES)}, got {strategy!r}."
            )
        self.target = target
        self.strategy = strategy
        self.output_column = output_column
        self.constant_: object = None

    def fit(self, ds: Dataset) -> DummyClassifier:
        """Learn the most frequent class (ties broken toward the smaller label).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.dummy import DummyClassifier
                >>> DummyClassifier("y").fit(bt.from_pydict({"y": [0, 1, 1, 1]})).constant_
                1

        Args:
            ds: The training dataset.

        Returns:
            ``self``, fitted.
        """
        counts = (
            ds.filter(col(self.target).is_not_null())
            .group_by(self.target)
            .agg(__bt_n=col(self.target).count())
            .collect()
        )
        best_label: object = None
        best_count = -1
        for i in range(counts.num_rows):
            label = counts.column(self.target)[i].as_py()
            count = int(counts.column("__bt_n")[i].as_py())
            if count > best_count or (count == best_count and _less(label, best_label)):
                best_label, best_count = label, count
        self.constant_ = best_label
        return self

    def predict(self, ds: Dataset) -> Dataset:
        """Append the majority class as the prediction for every row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.dummy import DummyClassifier
                >>> ds = bt.from_pydict({"y": ["x", "x", "z"]})
                >>> DummyClassifier("y").fit(ds).predict(ds).to_pydict()["prediction"]
                ['x', 'x', 'x']

        Args:
            ds: The dataset to score.

        Returns:
            A new lazy `Dataset` with the majority-class prediction appended.
        """
        if self.constant_ is None:
            raise PlanError("DummyClassifier must be fitted before predict.")
        return ds.with_columns(**{self.output_column: lit(self.constant_)})


def _less(candidate: object, current: object) -> bool:
    """Whether `candidate` sorts before `current`, used only to break frequency ties."""
    if current is None:
        return True
    try:
        return candidate < current  # type: ignore[operator]
    except TypeError:
        return str(candidate) < str(current)
