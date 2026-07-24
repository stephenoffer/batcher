"""Native linear models — ordinary and ridge regression trained inside the engine.

A linear model does not need a training framework: its normal equations are built entirely from
first and second moments, and those are mergeable aggregates. So Batcher fits ordinary least
squares and ridge regression in a single scan — the means and the covariance matrix over the
features and the target — and the only driver-side work is solving one small ``d x d`` system.
Prediction is then a linear-combination expression that lowers to Rust.

The point is not to replace a boosted tree; it is that a baseline linear model, a feature's
linear contribution, or a quick residual comes for free from the same columnar pass as
everything else, with no round-trip to a separate library and no per-row Python.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml._estimator import argmax_prediction, linear_score, require_fitted
from batcher.plan.expr_ir.constructors import col, lit, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["LinearRegression", "LogisticRegression", "Ridge", "RidgeClassifier"]


def _solve(
    ds: Dataset, features: Sequence[str], target: str, alpha: float
) -> tuple[list[float], float]:
    """Fit linear coefficients and intercept from the moments of `features` and `target`."""
    import numpy as np

    from batcher.ml.stats.multivariate import covariance_matrix
    from batcher.plan.functions.aggregate import mean as mean_

    columns = [*features, target]
    for name in columns:
        if name not in ds.columns:
            from batcher._internal.errors import ColumnNotFoundError, unknown_message

            raise ColumnNotFoundError(
                unknown_message("column", name, ds.columns, hint="Pass an existing column.")
            )
    means = ds.agg(**{name: mean_(col(name)) for name in columns}).collect()
    center = {name: float(means.column(name)[0].as_py()) for name in columns}
    n = ds.count()
    covariance = covariance_matrix(ds, columns).to_pydict()
    matrix = np.array([covariance[name] for name in columns], dtype=float).T
    d = len(features)
    sxx = matrix[:d, :d]
    sxy = matrix[:d, d]
    system = (n - 1) * sxx + alpha * np.eye(d)
    coefficients = np.linalg.solve(system, (n - 1) * sxy)
    intercept = center[target] - sum(
        coefficients[i] * center[name] for i, name in enumerate(features)
    )
    return [float(c) for c in coefficients], float(intercept)


class LinearRegression:
    """Ordinary least squares, fitted in one scan from the feature/target moments.

    The normal-equations solution ``beta = cov(X)^-1 cov(X, y)`` with an intercept, computed from
    aggregates so the whole fit is a single pass over the data. Reproduces scikit-learn's
    ``LinearRegression`` coefficients and intercept exactly. `predict` appends the linear
    prediction as a new column, so it composes with every other lazy operation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.linear import LinearRegression
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [3.0, 5.0, 7.0, 9.0]})
            >>> model = LinearRegression(["x"], "y").fit(ds)
            >>> round(model.coef_[0], 6), round(model.intercept_, 6)
            (2.0, 1.0)

    Args:
        features: The predictor columns.
        target: The column to predict.
        output_column: The name of the prediction column `predict` appends.
    """

    __slots__ = ("_alpha", "coef_", "features", "intercept_", "output_column", "target")

    def __init__(
        self, features: Sequence[str], target: str, *, output_column: str = "prediction"
    ) -> None:
        self.features = list(features)
        if not self.features:
            raise PlanError("LinearRegression needs at least one feature column.")
        self.target = target
        self.output_column = output_column
        self._alpha = 0.0
        self.coef_: list[float] = []
        self.intercept_: float = 0.0

    def fit(self, ds: Dataset) -> LinearRegression:
        """Learn the coefficients and intercept from the data's moments.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.linear import LinearRegression
                >>> ds = bt.from_pydict({"x": [0.0, 1.0, 2.0], "y": [1.0, 3.0, 5.0]})
                >>> LinearRegression(["x"], "y").fit(ds).coef_
                [2.0]

        Args:
            ds: The training dataset.

        Returns:
            ``self``, fitted.
        """
        self.coef_, self.intercept_ = _solve(ds, self.features, self.target, self._alpha)
        return self

    def predict(self, ds: Dataset) -> Dataset:
        """Append the linear prediction as a new column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.linear import LinearRegression
                >>> ds = bt.from_pydict({"x": [1.0, 2.0], "y": [3.0, 5.0]})
                >>> model = LinearRegression(["x"], "y").fit(ds)
                >>> model.predict(bt.from_pydict({"x": [5.0]})).to_pydict()["prediction"]
                [11.0]

        Args:
            ds: The dataset to score.

        Returns:
            A new lazy `Dataset` with the prediction column appended.
        """
        require_fitted(self, self.coef_)
        expression = linear_score(self.features, self.coef_, self.intercept_)
        return ds.with_columns(**{self.output_column: expression})


class Ridge(LinearRegression):
    """L2-regularized least squares, fitted in one scan.

    Ridge adds an ``alpha * ||beta||^2`` penalty to the least-squares objective, which shrinks the
    coefficients and stabilizes the fit when the features are collinear — exactly the case where
    ordinary least squares blows up. The solution is ``(X'X + alpha I)^-1 X'y``, still built from
    the same one-scan moments. Reproduces scikit-learn's ``Ridge`` coefficients exactly.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.linear import Ridge
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [3.0, 5.0, 7.0, 9.0]})
            >>> model = Ridge(["x"], "y", alpha=1.0).fit(ds)
            >>> abs(model.coef_[0]) < 2.0  # shrunk below the OLS slope of 2
            True

    Args:
        features: The predictor columns.
        target: The column to predict.
        alpha: The L2 penalty strength; 0 recovers ordinary least squares.
        output_column: The name of the prediction column `predict` appends.
    """

    __slots__ = ()

    def __init__(
        self,
        features: Sequence[str],
        target: str,
        *,
        alpha: float = 1.0,
        output_column: str = "prediction",
    ) -> None:
        super().__init__(features, target, output_column=output_column)
        if alpha < 0:
            raise PlanError(f"alpha must be non-negative, got {alpha}.")
        self._alpha = alpha


class LogisticRegression:
    """Binary logistic regression, fitted in the engine by iteratively reweighted least squares.

    A native GLM classifier: it models ``P(y=1) = sigmoid(intercept + beta . x)`` and fits the
    coefficients by Newton-Raphson (IRLS). Each iteration is one scan — the gradient and the
    Hessian are sums of per-row products, so they are aggregates — and the small Newton solve
    runs on the driver. Reproduces scikit-learn's unpenalized ``LogisticRegression`` coefficients.

    `predict_proba` appends the probability of the positive class; `predict` thresholds it at 0.5
    to a 0/1 label. Both are single-pass expressions.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.linear import LogisticRegression
            >>> ds = bt.from_pydict(
            ...     {"x": [-2.0, -1.0, 1.0, 2.0], "y": [0, 0, 1, 1]}
            ... )
            >>> model = LogisticRegression(["x"], "y").fit(ds)
            >>> model.predict(bt.from_pydict({"x": [-5.0, 5.0]})).to_pydict()["prediction"]
            [0, 1]

    Args:
        features: The predictor columns.
        target: The 0/1 target column.
        max_iter: The maximum number of IRLS iterations.
        tol: The convergence tolerance on the coefficient update's max absolute change.
        output_column: The name of the output column `predict`/`predict_proba` appends.
    """

    __slots__ = (
        "coef_",
        "features",
        "intercept_",
        "max_iter",
        "n_iter_",
        "output_column",
        "target",
        "tol",
    )

    def __init__(
        self,
        features: Sequence[str],
        target: str,
        *,
        max_iter: int = 100,
        tol: float = 1e-8,
        output_column: str = "prediction",
    ) -> None:
        self.features = list(features)
        if not self.features:
            raise PlanError("LogisticRegression needs at least one feature column.")
        self.target = target
        self.max_iter = max_iter
        self.tol = tol
        self.output_column = output_column
        self.coef_: list[float] = []
        self.intercept_: float = 0.0
        self.n_iter_: int = 0

    def _linear_predictor(self, coefficients: list[float], intercept: float):
        """The expression ``intercept + beta . x`` for the current coefficients."""
        expression = lit(intercept)
        for weight, name in zip(coefficients, self.features, strict=True):
            expression = expression + lit(weight) * col(name)
        return expression

    def fit(self, ds: Dataset) -> LogisticRegression:
        """Fit the coefficients by iteratively reweighted least squares.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.linear import LogisticRegression
                >>> ds = bt.from_pydict({"x": [-2.0, -1.0, 1.0, 2.0], "y": [0, 0, 1, 1]})
                >>> model = LogisticRegression(["x"], "y").fit(ds)
                >>> model.coef_[0] > 0  # a higher x means a higher probability of class 1
                True

        Args:
            ds: The training dataset.

        Returns:
            ``self``, fitted.

        Raises:
            ColumnNotFoundError: If a named column is missing.
        """
        import numpy as np

        from batcher.plan.functions.aggregate import sum as sum_

        for name in (*self.features, self.target):
            if name not in ds.columns:
                from batcher._internal.errors import ColumnNotFoundError, unknown_message

                raise ColumnNotFoundError(
                    unknown_message("column", name, ds.columns, hint="Pass an existing column.")
                )
        terms = [lit(1.0), *[col(name) for name in self.features]]
        m = len(terms)
        beta = np.zeros(m)
        for iteration in range(self.max_iter):
            eta = self._linear_predictor(list(beta[1:]), float(beta[0]))
            probability = lit(1.0) / (lit(1.0) + (-eta).exp())
            residual = col(self.target) - probability
            weight = probability * (lit(1.0) - probability)
            aggregates = {}
            for j in range(m):
                aggregates[f"g{j}"] = sum_(residual * terms[j])
                for k in range(j, m):
                    aggregates[f"h{j}_{k}"] = sum_(weight * terms[j] * terms[k])
            row = ds.agg(**aggregates).collect()
            gradient = np.array([float(row.column(f"g{j}")[0].as_py()) for j in range(m)])
            hessian = np.zeros((m, m))
            for j in range(m):
                for k in range(j, m):
                    value = float(row.column(f"h{j}_{k}")[0].as_py())
                    hessian[j, k] = hessian[k, j] = value
            step = np.linalg.solve(hessian + 1e-10 * np.eye(m), gradient)
            beta = beta + step
            self.n_iter_ = iteration + 1
            if np.max(np.abs(step)) < self.tol:
                break
        self.intercept_ = float(beta[0])
        self.coef_ = [float(c) for c in beta[1:]]
        return self

    def predict_proba(self, ds: Dataset) -> Dataset:
        """Append the predicted probability of the positive class.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.linear import LogisticRegression
                >>> ds = bt.from_pydict({"x": [-2.0, 2.0], "y": [0, 1]})
                >>> model = LogisticRegression(["x"], "y").fit(ds)
                >>> proba = model.predict_proba(ds).to_pydict()["prediction"]
                >>> proba[0] < 0.5 < proba[1]
                True

        Args:
            ds: The dataset to score.

        Returns:
            A new lazy `Dataset` with the probability column appended.
        """
        require_fitted(self, self.coef_)
        eta = self._linear_predictor(self.coef_, self.intercept_)
        probability = lit(1.0) / (lit(1.0) + (-eta).exp())
        return ds.with_columns(**{self.output_column: probability})

    def predict(self, ds: Dataset) -> Dataset:
        """Append the predicted 0/1 label (the positive class where the probability reaches 0.5).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.linear import LogisticRegression
                >>> ds = bt.from_pydict({"x": [-2.0, 2.0], "y": [0, 1]})
                >>> model = LogisticRegression(["x"], "y").fit(ds)
                >>> model.predict(ds).to_pydict()["prediction"]
                [0, 1]

        Args:
            ds: The dataset to score.

        Returns:
            A new lazy `Dataset` with the 0/1 label column appended.
        """
        require_fitted(self, self.coef_)
        eta = self._linear_predictor(self.coef_, self.intercept_)
        label = when(eta >= lit(0.0)).then(lit(1)).otherwise(lit(0))
        return ds.with_columns(**{self.output_column: label})


class RidgeClassifier:
    """Classification by ridge regression on one-vs-rest ``+1`` / ``-1`` targets.

    Treats classification as regression: for each class it fits a ridge regression whose target is
    ``+1`` for that class and ``-1`` for the rest, then labels a row with the class whose regression
    scores it highest. The L2 penalty makes it stable under collinear features, and because it is a
    closed-form least-squares fit it trains in a single scan per class with no iteration.
    Reproduces scikit-learn's ``RidgeClassifier``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.linear import RidgeClassifier
            >>> ds = bt.from_pydict(
            ...     {"x": [0.0, 0.5, 5.0, 5.5], "y": [0, 0, 1, 1]}
            ... )
            >>> model = RidgeClassifier(["x"], "y").fit(ds)
            >>> model.predict(bt.from_pydict({"x": [0.2, 5.2]})).to_pydict()["prediction"]
            [0, 1]

    Args:
        features: The predictor columns.
        target: The class label column.
        alpha: The L2 penalty strength.
        output_column: The name of the predicted-class column `predict` appends.
    """

    __slots__ = ("alpha", "bias_", "classes_", "features", "output_column", "target", "weights_")

    def __init__(
        self,
        features: Sequence[str],
        target: str,
        *,
        alpha: float = 1.0,
        output_column: str = "prediction",
    ) -> None:
        self.features = list(features)
        if not self.features:
            raise PlanError("RidgeClassifier needs at least one feature column.")
        if alpha < 0:
            raise PlanError(f"alpha must be non-negative, got {alpha}.")
        self.target = target
        self.alpha = alpha
        self.output_column = output_column
        self.classes_: list[object] = []
        self.weights_: dict[object, list[float]] = {}
        self.bias_: dict[object, float] = {}

    def fit(self, ds: Dataset) -> RidgeClassifier:
        """Fit a one-vs-rest ridge regression for every class.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.linear import RidgeClassifier
                >>> ds = bt.from_pydict({"x": [0.0, 1.0, 9.0, 10.0], "y": [0, 0, 1, 1]})
                >>> sorted(RidgeClassifier(["x"], "y").fit(ds).classes_)
                [0, 1]

        Args:
            ds: The training dataset.

        Returns:
            ``self``, fitted.

        Raises:
            ColumnNotFoundError: If a named column is missing.
        """
        if self.target not in ds.columns:
            from batcher._internal.errors import ColumnNotFoundError, unknown_message

            raise ColumnNotFoundError(
                unknown_message("column", self.target, ds.columns, hint="Pass an existing column.")
            )
        labels = [
            v.as_py() for v in ds.select(self.target).distinct().collect().column(self.target)
        ]
        self.classes_, self.weights_, self.bias_ = [], {}, {}
        for label in labels:
            indicator = when(col(self.target) == lit(label)).then(lit(1.0)).otherwise(lit(-1.0))
            one_vs_rest = ds.with_columns(__bt_ind=indicator)
            coefficients, intercept = _solve(one_vs_rest, self.features, "__bt_ind", self.alpha)
            self.classes_.append(label)
            self.weights_[label] = coefficients
            self.bias_[label] = intercept
        return self

    def _score(self, label: object):
        """The ridge regression score expression for one class."""
        return linear_score(self.features, self.weights_[label], self.bias_[label])

    def predict(self, ds: Dataset) -> Dataset:
        """Append the highest-scoring class label for each row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.linear import RidgeClassifier
                >>> ds = bt.from_pydict({"x": [0.0, 1.0, 9.0, 10.0], "y": [0, 0, 1, 1]})
                >>> model = RidgeClassifier(["x"], "y").fit(ds)
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
