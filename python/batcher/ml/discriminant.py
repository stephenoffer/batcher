"""Discriminant analysis — Gaussian classifiers that model each class's full covariance.

Where Gaussian naive Bayes assumes the features are independent within a class (a diagonal
covariance), discriminant analysis learns each class's *full* covariance, so it can model
correlated features and tilted, stretched class shapes. The fit is still aggregates: a mean
vector and a covariance matrix per class, both single-scan. Only the per-class matrix inverse and
determinant, over a tiny ``d x d`` matrix, run on the driver, and prediction is a quadratic-form
expression.

Quadratic discriminant analysis gives each class its own covariance, which is the general case;
it reduces to linear discriminant analysis when the covariances are assumed equal. This module
provides both: `LinearDiscriminantAnalysis` (one shared covariance, linear boundaries) and
`QuadraticDiscriminantAnalysis` (one covariance per class, quadratic boundaries).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml._estimator import argmax_prediction, linear_score, require_fitted
from batcher.plan.expr_ir.constructors import col, lit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["LinearDiscriminantAnalysis", "QuadraticDiscriminantAnalysis"]


class QuadraticDiscriminantAnalysis:
    """A Gaussian classifier with a full per-class covariance — the quadratic-boundary classifier.

    Fits each class's prior, mean vector, and full covariance matrix (each a single-scan
    aggregate), then classifies by the maximum-a-posteriori multivariate-Gaussian log-likelihood.
    Because each class keeps its own covariance, the decision boundary between two classes is
    quadratic, which lets it separate classes that differ in *spread* or *orientation*, not just
    location — the case `GaussianNB` and a linear model both miss. Reproduces scikit-learn's
    ``QuadraticDiscriminantAnalysis`` predictions.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.discriminant import QuadraticDiscriminantAnalysis
            >>> ds = bt.from_pydict(
            ...     {"a": [0.0, 1.0, 0.5, 8.0, 9.0, 8.5], "b": [0.0, 1.0, 0.5, 8.0, 9.0, 8.5],
            ...      "y": [0, 0, 0, 1, 1, 1]}
            ... )
            >>> model = QuadraticDiscriminantAnalysis(["a", "b"], "y").fit(ds)
            >>> query = bt.from_pydict({"a": [0.2, 8.8], "b": [0.2, 8.8]})
            >>> model.predict(query).to_pydict()["prediction"]
            [0, 1]

    Args:
        features: The numeric feature columns.
        target: The class label column.
        output_column: The name of the predicted-class column `predict` appends.
    """

    __slots__ = (
        "classes_",
        "features",
        "log_prior_",
        "means_",
        "output_column",
        "precision_",
        "target",
    )

    def __init__(
        self, features: Sequence[str], target: str, *, output_column: str = "prediction"
    ) -> None:
        self.features = list(features)
        if len(self.features) < 1:
            raise PlanError("QuadraticDiscriminantAnalysis needs at least one feature column.")
        self.target = target
        self.output_column = output_column
        self.classes_: list[object] = []
        self.means_: dict[object, list[float]] = {}
        self.precision_: dict[object, object] = {}
        self.log_prior_: dict[object, float] = {}

    def fit(self, ds: Dataset) -> QuadraticDiscriminantAnalysis:
        """Learn each class's prior, mean, and covariance, then cache the inverse and log-det.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.discriminant import QuadraticDiscriminantAnalysis
                >>> ds = bt.from_pydict({"a": [0.0, 1.0, 9.0, 10.0], "b": [0.0, 1.0, 9.0, 10.0],
                ...                      "y": [0, 0, 1, 1]})
                >>> sorted(QuadraticDiscriminantAnalysis(["a", "b"], "y").fit(ds).classes_)
                [0, 1]

        Args:
            ds: The training dataset.

        Returns:
            ``self``, fitted.

        Raises:
            ColumnNotFoundError: If a named column is missing.
        """
        import numpy as np

        from batcher.ml.stats.multivariate import covariance_matrix
        from batcher.plan.functions.aggregate import mean as mean_

        for name in (*self.features, self.target):
            if name not in ds.columns:
                from batcher._internal.errors import ColumnNotFoundError, unknown_message

                raise ColumnNotFoundError(
                    unknown_message("column", name, ds.columns, hint="Pass an existing column.")
                )
        total = ds.count()
        labels = [
            v.as_py() for v in ds.select(self.target).distinct().collect().column(self.target)
        ]
        self.classes_, self.means_, self.precision_, self.log_prior_ = [], {}, {}, {}
        for label in labels:
            subset = ds.filter(col(self.target) == label)
            count = subset.count()
            means = subset.agg(**{name: mean_(col(name)) for name in self.features}).collect()
            covariance = covariance_matrix(subset, self.features).to_pydict()
            matrix = np.array([covariance[name] for name in self.features], dtype=float).T
            self.classes_.append(label)
            self.means_[label] = [float(means.column(name)[0].as_py()) for name in self.features]
            sign, logdet = np.linalg.slogdet(matrix)
            inverse = np.linalg.pinv(matrix)
            self.precision_[label] = (inverse, float(logdet if sign > 0 else 0.0))
            self.log_prior_[label] = math.log(count / total)
        return self

    def _score(self, label: object):
        """The multivariate-Gaussian log-posterior expression for one class."""
        inverse, logdet = self.precision_[label]
        means = self.means_[label]
        centered = [col(name) - lit(means[i]) for i, name in enumerate(self.features)]
        quadratic = lit(0.0)
        for i in range(len(self.features)):
            for j in range(len(self.features)):
                weight = float(inverse[i, j])
                if weight != 0.0:
                    quadratic = quadratic + lit(weight) * centered[i] * centered[j]
        return lit(self.log_prior_[label] - 0.5 * logdet) - lit(0.5) * quadratic

    def predict(self, ds: Dataset) -> Dataset:
        """Append the maximum-a-posteriori class label for each row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.discriminant import QuadraticDiscriminantAnalysis
                >>> ds = bt.from_pydict({"a": [0.0, 1.0, 9.0, 10.0], "b": [0.0, 1.0, 9.0, 10.0],
                ...                      "y": [0, 0, 1, 1]})
                >>> model = QuadraticDiscriminantAnalysis(["a", "b"], "y").fit(ds)
                >>> model.predict(ds).to_pydict()["prediction"]
                [0, 0, 1, 1]

        Args:
            ds: The dataset to classify.

        Returns:
            A new lazy `Dataset` with the predicted-class column appended.
        """
        require_fitted(self, self.classes_)
        prediction = argmax_prediction(self.classes_, self._score)
        return ds.with_columns(**{self.output_column: prediction})


class LinearDiscriminantAnalysis:
    """A Gaussian classifier with one shared covariance — the linear-boundary discriminant.

    Assumes every class has the same covariance, differing only in mean. That assumption collapses
    the per-class quadratic terms of `QuadraticDiscriminantAnalysis` and leaves a *linear* decision
    boundary, which is more stable when the classes are few or the data is scarce (one pooled
    covariance is estimated from all the rows rather than one per class from a fraction of them).
    It is fitted from per-class mean aggregates plus the pooled within-class covariance, and
    reproduces scikit-learn's ``LinearDiscriminantAnalysis`` predictions.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.discriminant import LinearDiscriminantAnalysis
            >>> ds = bt.from_pydict(
            ...     {"a": [0.0, 1.0, 0.5, 8.0, 9.0, 8.5], "b": [0.0, 1.0, 0.5, 8.0, 9.0, 8.5],
            ...      "y": [0, 0, 0, 1, 1, 1]}
            ... )
            >>> model = LinearDiscriminantAnalysis(["a", "b"], "y").fit(ds)
            >>> query = bt.from_pydict({"a": [0.2, 8.8], "b": [0.2, 8.8]})
            >>> model.predict(query).to_pydict()["prediction"]
            [0, 1]

    Args:
        features: The numeric feature columns.
        target: The class label column.
        output_column: The name of the predicted-class column `predict` appends.
    """

    __slots__ = ("bias_", "classes_", "features", "output_column", "target", "weights_")

    def __init__(
        self, features: Sequence[str], target: str, *, output_column: str = "prediction"
    ) -> None:
        self.features = list(features)
        if len(self.features) < 1:
            raise PlanError("LinearDiscriminantAnalysis needs at least one feature column.")
        self.target = target
        self.output_column = output_column
        self.classes_: list[object] = []
        self.weights_: dict[object, list[float]] = {}
        self.bias_: dict[object, float] = {}

    def fit(self, ds: Dataset) -> LinearDiscriminantAnalysis:
        """Learn the per-class means and the pooled covariance, then form the linear discriminants.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.discriminant import LinearDiscriminantAnalysis
                >>> ds = bt.from_pydict({"a": [0.0, 1.0, 9.0, 10.0], "b": [0.0, 1.0, 9.0, 10.0],
                ...                      "y": [0, 0, 1, 1]})
                >>> sorted(LinearDiscriminantAnalysis(["a", "b"], "y").fit(ds).classes_)
                [0, 1]

        Args:
            ds: The training dataset.

        Returns:
            ``self``, fitted.

        Raises:
            ColumnNotFoundError: If a named column is missing.
        """
        import numpy as np

        from batcher.ml.stats.multivariate import covariance_matrix
        from batcher.plan.functions.aggregate import mean as mean_

        for name in (*self.features, self.target):
            if name not in ds.columns:
                from batcher._internal.errors import ColumnNotFoundError, unknown_message

                raise ColumnNotFoundError(
                    unknown_message("column", name, ds.columns, hint="Pass an existing column.")
                )
        total = ds.count()
        labels = [
            v.as_py() for v in ds.select(self.target).distinct().collect().column(self.target)
        ]
        d = len(self.features)
        means: dict[object, object] = {}
        priors: dict[object, float] = {}
        scatter = np.zeros((d, d))
        for label in labels:
            subset = ds.filter(col(self.target) == label)
            count = subset.count()
            mean_row = subset.agg(**{name: mean_(col(name)) for name in self.features}).collect()
            means[label] = np.array(
                [float(mean_row.column(name)[0].as_py()) for name in self.features]
            )
            priors[label] = count / total
            if count > 1:
                covariance = covariance_matrix(subset, self.features).to_pydict()
                matrix = np.array([covariance[name] for name in self.features], dtype=float).T
                scatter += matrix * (count - 1)
        pooled = scatter / max(total - len(labels), 1)
        precision = np.linalg.pinv(pooled)
        self.classes_, self.weights_, self.bias_ = [], {}, {}
        for label in labels:
            weight = precision @ means[label]
            self.classes_.append(label)
            self.weights_[label] = [float(w) for w in weight]
            self.bias_[label] = float(-0.5 * means[label] @ weight + math.log(priors[label]))
        return self

    def _score(self, label: object):
        """The linear discriminant score expression for one class."""
        return linear_score(self.features, self.weights_[label], self.bias_[label])

    def predict(self, ds: Dataset) -> Dataset:
        """Append the maximum-discriminant class label for each row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.discriminant import LinearDiscriminantAnalysis
                >>> ds = bt.from_pydict({"a": [0.0, 1.0, 9.0, 10.0], "b": [0.0, 1.0, 9.0, 10.0],
                ...                      "y": [0, 0, 1, 1]})
                >>> model = LinearDiscriminantAnalysis(["a", "b"], "y").fit(ds)
                >>> model.predict(ds).to_pydict()["prediction"]
                [0, 0, 1, 1]

        Args:
            ds: The dataset to classify.

        Returns:
            A new lazy `Dataset` with the predicted-class column appended.
        """
        require_fitted(self, self.classes_)
        prediction = argmax_prediction(self.classes_, self._score)
        return ds.with_columns(**{self.output_column: prediction})
