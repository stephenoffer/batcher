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


def class_moments(
    ds: Dataset, features: Sequence[str], target: str
) -> tuple[list[object], dict[object, int], dict[object, list[float]], dict[object, object]]:
    """Every class's count, mean vector and covariance matrix, from one grouped pass.

    Both discriminants used to filter the frame per class and run a count, a mean aggregate
    and a covariance over each — three scans per class, so a ten-class fit read the data
    thirty-one times. A covariance is recoverable from sums and sums of products, and those
    are ordinary mergeable aggregates, so every class's moments ride a single
    ``group_by(target)`` and only the tiny ``d x d`` arithmetic happens on the driver.

    The covariance uses the ``E[xy] - E[x]E[y]`` form with the sample (``n - 1``) divisor,
    matching what a per-class `covariance_matrix` returned.

    Args:
        ds: The training data.
        features: The numeric feature columns.
        target: The class label column.

    Returns:
        ``(classes, counts, means, covariances)`` keyed by class label; `covariances` holds
        one ``d x d`` NumPy array per class, and a class with a single row maps to ``None``
        because a sample covariance is undefined there.
    """
    import numpy as np

    from batcher.plan.functions.aggregate import sum as sum_

    width = len(features)
    aggregates: dict[str, object] = {"__bt_n": col(features[0]).count()}
    for i, name in enumerate(features):
        aggregates[f"__bt_s{i}"] = sum_(col(name).cast("float64"))
    for i in range(width):
        for j in range(i, width):
            left = col(features[i]).cast("float64")
            right = col(features[j]).cast("float64")
            aggregates[f"__bt_p{i}_{j}"] = sum_(left * right)
    grouped = ds.group_by(target).agg(**aggregates).collect()

    classes: list[object] = []
    counts: dict[object, int] = {}
    means: dict[object, list[float]] = {}
    covariances: dict[object, object] = {}
    for row in range(grouped.num_rows):
        label = grouped.column(target)[row].as_py()
        count = int(grouped.column("__bt_n")[row].as_py() or 0)
        if not count:
            continue
        sums = [float(grouped.column(f"__bt_s{i}")[row].as_py() or 0.0) for i in range(width)]
        classes.append(label)
        counts[label] = count
        means[label] = [s / count for s in sums]
        if count < 2:
            covariances[label] = None
            continue
        matrix = np.zeros((width, width))
        for i in range(width):
            for j in range(i, width):
                product = float(grouped.column(f"__bt_p{i}_{j}")[row].as_py() or 0.0)
                value = (product - sums[i] * sums[j] / count) / (count - 1)
                matrix[i, j] = matrix[j, i] = value
        covariances[label] = matrix
    return classes, counts, means, covariances


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

        for name in (*self.features, self.target):
            if name not in ds.columns:
                from batcher._internal.errors import ColumnNotFoundError, unknown_message

                raise ColumnNotFoundError(
                    unknown_message("column", name, ds.columns, hint="Pass an existing column.")
                )
        labels, counts, means, covariances = class_moments(ds, self.features, self.target)
        total = sum(counts.values())
        self.classes_, self.means_, self.precision_, self.log_prior_ = [], {}, {}, {}
        for label in labels:
            count = counts[label]
            if count < 2:
                # A full per-class covariance divides by `count - 1`, so a class with a single
                # example produces a NaN matrix and `pinv` then raised
                # ``LinAlgError: SVD did not converge`` — a solver message for what is really a
                # rare class. This is the ordinary shape of imbalanced data, so name the class.
                from batcher._internal.errors import PlanError

                raise PlanError(
                    f"QuadraticDiscriminantAnalysis needs at least 2 examples per class to "
                    f"estimate a covariance, but class {label!r} has {count}. Drop or merge the "
                    f"rare class, or use LinearDiscriminantAnalysis, which pools one covariance "
                    f"across all classes."
                )
            matrix = covariances[label]
            self.classes_.append(label)
            self.means_[label] = means[label]
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

        for name in (*self.features, self.target):
            if name not in ds.columns:
                from batcher._internal.errors import ColumnNotFoundError, unknown_message

                raise ColumnNotFoundError(
                    unknown_message("column", name, ds.columns, hint="Pass an existing column.")
                )
        labels, counts, class_means, covariances = class_moments(ds, self.features, self.target)
        total = sum(counts.values())
        d = len(self.features)
        means: dict[object, object] = {}
        priors: dict[object, float] = {}
        scatter = np.zeros((d, d))
        for label in labels:
            means[label] = np.asarray(class_means[label])
            priors[label] = counts[label] / total
            matrix = covariances[label]
            if matrix is not None:
                scatter += matrix * (counts[label] - 1)
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
