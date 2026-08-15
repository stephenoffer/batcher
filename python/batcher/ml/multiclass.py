"""`OneVsRestClassifier` — multiclass classification built from a binary estimator.

`LogisticRegression` fits one weight vector, so it answers exactly one question: is this row
the positive class or not. Handed a target with three labels it does not refuse — it fits
against the raw label values as if they were probabilities and returns a model that predicts
nearly one class for everything. That is the worst kind of failure: a fitted object, a
plausible-looking prediction column, and no error anywhere.

Two things close that hole. `LogisticRegression.fit` now rejects a non-binary target and says
what to use instead, and this module provides the thing to use: one binary model per class,
each asking "is it *this* label?", with the highest score winning.

The decomposition is what makes it scale. Each sub-model is an ordinary Batcher estimator
fitted by the same mergeable aggregates, so a class is learned from partial sums combined
across partitions, single node or a hundred. Prediction is one projection: every sub-model's
score is staged as a column and folded into a single `argmax` expression tree, so classifying
a hundred classes is still one pass over the data rather than a hundred.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml._estimator import argmax_prediction, require_fitted
from batcher.ml.stats._shared import require_columns
from batcher.plan.expr_ir import col, lit, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["MAX_CLASSES", "OneVsRestClassifier"]

#: The ceiling on how many classes `OneVsRestClassifier` will fit.
#:
#: One binary model is fitted per class, so a target mistakenly pointed at a high-cardinality
#: column - an identifier, a timestamp - would fit one model per distinct value and appear to
#: hang. The bound turns that into an immediate error naming the column.
MAX_CLASSES = 100

_INDICATOR = "__bt_ovr_indicator"
_SCORE = "__bt_ovr_score_"


class OneVsRestClassifier:
    """Fit one binary model per class and predict whichever scores highest.

    The estimator is passed as a *class*, not an instance, because a Batcher estimator binds
    its feature and target columns at construction and each sub-model needs a different
    target. `params` carries the hyperparameters to give every sub-model.

    The base estimator must expose `predict_proba`. A hard 0/1 prediction cannot be compared
    across classes, so a model that only predicts labels gives nothing to take an `argmax`
    over and is rejected at `fit` rather than silently ranked.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import LogisticRegression, OneVsRestClassifier
            >>> ds = bt.from_pydict(
            ...     {"x": [0.0, 1.0, 2.0, 8.0, 9.0, 10.0, 20.0, 21.0, 22.0],
            ...      "y": [0, 0, 0, 1, 1, 1, 2, 2, 2]}
            ... )
            >>> model = OneVsRestClassifier(LogisticRegression, ["x"], "y").fit(ds)
            >>> model.classes_
            [0, 1, 2]
            >>> model.predict(ds).to_pydict()["prediction"]
            [0, 0, 0, 1, 1, 1, 2, 2, 2]

    Args:
        estimator: The binary estimator class to fit once per class.
        features: The feature columns.
        target: The class column. Its values may be of any type.
        params: Hyperparameters passed to every sub-model. The sub-model's `output_column` is
            managed here and must not be set.
        output_column: The name of the predicted-class column `predict` appends.
        max_classes: The ceiling on distinct labels, guarding a mis-aimed target column.

    Raises:
        PlanError: If `features` is empty, or the base estimator has no `predict_proba`.
    """

    __slots__ = (
        "classes_",
        "estimator",
        "estimators_",
        "features",
        "max_classes",
        "output_column",
        "params",
        "target",
    )

    def __init__(
        self,
        estimator: type,
        features: Sequence[str],
        target: str,
        *,
        params: dict[str, Any] | None = None,
        output_column: str = "prediction",
        max_classes: int = MAX_CLASSES,
    ) -> None:
        self.features = list(features)
        if not self.features:
            raise PlanError("OneVsRestClassifier needs at least one feature column.")
        if not callable(getattr(estimator, "predict_proba", None)):
            raise PlanError(
                f"OneVsRestClassifier needs a base estimator with predict_proba, and "
                f"{getattr(estimator, '__name__', estimator)!r} has none. Comparing classes "
                "means ranking their scores, which a hard 0/1 prediction cannot do."
            )
        self.estimator = estimator
        self.target = target
        self.params = dict(params or {})
        self.output_column = output_column
        self.max_classes = max_classes
        self.classes_: list[Any] = []
        self.estimators_: list[Any] = []

    def fit(self, ds: Dataset) -> OneVsRestClassifier:
        """Fit one binary sub-model per distinct label.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import LogisticRegression, OneVsRestClassifier
                >>> ds = bt.from_pydict(
                ...     {"x": [0.0, 1.0, 9.0, 10.0, 20.0, 21.0],
                ...      "y": ["low", "low", "mid", "mid", "high", "high"]}
                ... )
                >>> model = OneVsRestClassifier(LogisticRegression, ["x"], "y").fit(ds)
                >>> len(model.estimators_) == len(model.classes_) == 3
                True

        Args:
            ds: The training dataset.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If the target has fewer than two classes, or more than `max_classes`.
            ColumnNotFoundError: If a named column is missing.
        """
        require_columns(ds, *self.features, self.target)
        labels = [
            v.as_py()
            for v in ds.select(self.target)
            .distinct()
            .limit(self.max_classes + 1)
            .collect()
            .column(self.target)
        ]
        if len(labels) > self.max_classes:
            raise PlanError(
                f"OneVsRestClassifier: {self.target!r} has more than {self.max_classes} "
                "distinct values, which would fit that many models. If this column really is "
                "the class, raise max_classes; if it is an identifier, it is not the target."
            )
        if len(labels) < 2:
            raise PlanError(
                f"OneVsRestClassifier: {self.target!r} has {len(labels)} distinct value(s), "
                "so there is nothing to tell apart. Fit a binary model directly."
            )
        # Sorted so `classes_`, the sub-model order and the staged score columns agree
        # regardless of the order the scan happened to return the labels in - otherwise a
        # model fitted on two partitionings of the same data would number its columns
        # differently and could not be compared, or loaded, against the other.
        self.classes_ = sorted(labels, key=repr)
        self.estimators_ = []
        for index, label in enumerate(self.classes_):
            indicator = when(col(self.target) == lit(label)).then(lit(1.0)).otherwise(lit(0.0))
            binary = ds.with_columns(**{_INDICATOR: indicator})
            model = self.estimator(
                self.features,
                _INDICATOR,
                output_column=f"{_SCORE}{index}",
                **self.params,
            )
            self.estimators_.append(model.fit(binary))
        return self

    def predict(self, ds: Dataset) -> Dataset:
        """Append the highest-scoring class label for each row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import LogisticRegression, OneVsRestClassifier
                >>> ds = bt.from_pydict(
                ...     {"x": [0.0, 1.0, 9.0, 10.0, 20.0, 21.0],
                ...      "y": ["low", "low", "mid", "mid", "high", "high"]}
                ... )
                >>> model = OneVsRestClassifier(LogisticRegression, ["x"], "y").fit(ds)
                >>> model.predict(ds).to_pydict()["prediction"]
                ['low', 'low', 'mid', 'mid', 'high', 'high']

        Args:
            ds: The dataset to classify.

        Returns:
            A new lazy `Dataset` with the predicted-class column appended.
        """
        require_fitted(self, self.classes_ or None)
        staged = ds
        for model in self.estimators_:
            staged = model.predict_proba(staged)
        positions = {repr(label): index for index, label in enumerate(self.classes_)}
        prediction = argmax_prediction(
            self.classes_, lambda label: col(f"{_SCORE}{positions[repr(label)]}")
        )
        scored = staged.with_columns(**{self.output_column: prediction})
        return scored.drop(*[f"{_SCORE}{index}" for index in range(len(self.classes_))])
