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
from batcher.ml._estimator import (
    argmax_prediction,
    linear_score,
    require_fitted,
    require_numeric,
    require_rows,
)
from batcher.plan.expr_ir.constructors import col, lit, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from batcher.api.dataset import Dataset

__all__ = ["LinearRegression", "LogisticRegression", "Ridge", "RidgeClassifier"]


def _solve(
    ds: Dataset, features: Sequence[str], target: str, alpha: float, estimator: object
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
    require_numeric(estimator, ds, features)
    require_numeric(estimator, ds, [target], role="target")
    d = len(features)
    n = ds.count()
    require_rows(estimator, n, 2, because="the covariance it is built from divides by n - 1")
    means = ds.agg(**{name: mean_(col(name)) for name in columns}).collect()
    center = {name: float(means.column(name)[0].as_py()) for name in columns}
    covariance = covariance_matrix(ds, columns).to_pydict()
    matrix = np.array([covariance[name] for name in columns], dtype=float).T
    sxx = matrix[:d, :d]
    sxy = matrix[:d, d]
    system = (n - 1) * sxx + alpha * np.eye(d)
    try:
        coefficients = np.linalg.solve(system, (n - 1) * sxy)
    except np.linalg.LinAlgError as exc:
        # Singular means the features are linearly dependent, which `LinAlgError` never says.
        raise PlanError(
            f"the features {list(features)} are linearly dependent, so the least-squares "
            f"system has no unique solution. Drop the redundant column (a duplicate, a "
            f"constant, or a one-hot keeping every level), or use Ridge(alpha=...)."
        ) from exc
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
        self.coef_, self.intercept_ = _solve(ds, self.features, self.target, self._alpha, self)
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
        require_numeric(self, ds, self.features)
        # A non-binary target is the one input this fit cannot detect from its own arithmetic.
        # IRLS treats `target - probability` as a residual, so labels of 0/1/2 converge to a
        # model that predicts one class for nearly every row: fitted, plausible, and wrong,
        # with nothing raised. Three distinct values are enough to prove it, so the check is
        # bounded rather than a full scan of the column.
        seen = [
            v.as_py()
            for v in ds.select(self.target).distinct().limit(3).collect().column(self.target)
        ]
        extra = sorted((repr(v) for v in seen if v is not None and v not in (0, 1)), key=str)
        if extra:
            raise PlanError(
                f"LogisticRegression fits a binary target, but {self.target!r} contains "
                f"{extra[0]}. Encode it as 0/1 for a two-class problem, or use "
                "OneVsRestClassifier(LogisticRegression, features, target) to fit one binary "
                "model per class."
            )
        terms = [lit(1.0), *[col(name) for name in self.features]]
        m = len(terms)
        require_rows(self, ds.count(), m, because="IRLS needs one row per fitted term")
        beta = np.zeros(m)
        for iteration in range(self.max_iter):
            eta = self._linear_predictor(list(beta[1:]), float(beta[0]))
            probability = lit(1.0) / (lit(1.0) + (-eta).exp())
            # Cast because a boolean label column is one of the commonest ways to spell a
            # binary target, and Arrow will not subtract a float from a boolean: the IRLS
            # residual raised ``Invalid arithmetic operation: Boolean - Float64`` from inside
            # the engine, naming neither the column nor the fix.
            residual = col(self.target).cast("float64") - probability
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
            coefficients, intercept = _solve(
                one_vs_rest, self.features, "__bt_ind", self.alpha, self
            )
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


#: The ceiling on features for `RidgeCV`, which lowers to one aggregate per pair of terms.
#:
#: The whole cross-validation is one grouped aggregate of ``(d + 1)(d + 4) / 2`` sums. That is
#: 252 at twenty features and grows quadratically, so the bound keeps a mis-specified call
#: from building a plan with tens of thousands of aggregates instead of appearing to hang.
MAX_CV_FEATURES = 20


def _moment_blocks(row: dict[str, object], terms: int) -> tuple[float, NDArray, NDArray]:
    """Split one aggregate row into ``(n, sums, cross-products)`` over the fitted terms."""
    import numpy as np

    count = float(row["__bt_n"])
    sums = np.array([float(row[f"__bt_s{i}"] or 0.0) for i in range(terms)])
    products = np.zeros((terms, terms))
    for i in range(terms):
        for j in range(i, terms):
            value = float(row[f"__bt_p{i}_{j}"] or 0.0)
            products[i, j] = products[j, i] = value
    return count, sums, products


def _ridge_from_moments(count: float, sums, products, alpha: float, features: int):
    """Solve the ridge normal equations from moments alone, touching no rows.

    This is what makes the search cheap: centering, the penalty, and the solve are all
    arithmetic on a ``(d + 1) x (d + 1)`` matrix, so a new `alpha` costs a small `solve` and
    not another pass over the data.
    """
    import numpy as np

    centered = products - np.outer(sums, sums) / count
    system = centered[:features, :features] + alpha * np.eye(features)
    coefficients = np.linalg.solve(system, centered[:features, features])
    means = sums / count
    intercept = means[features] - float(coefficients @ means[:features])
    return coefficients, intercept


def _sse_from_moments(count: float, sums, products, coefficients, intercept: float) -> float:
    """The held-out sum of squared errors, expanded into the same moments.

    ``sum (y - b.x - c)^2`` multiplies out to ``Syy - 2b'Sxy - 2cSy + b'Sxx b + 2cb'Sx + nc^2``,
    every term of which is already in the fold's moment block. Scoring a fold therefore reads
    no rows either, which is the half of the saving that a naive search cannot avoid: it has
    to score each candidate on each fold, and each of those is a scan.
    """
    d = len(coefficients)
    sxx = products[:d, :d]
    sxy = products[:d, d]
    syy = products[d, d]
    sx = sums[:d]
    sy = sums[d]
    return float(
        syy
        - 2.0 * coefficients @ sxy
        - 2.0 * intercept * sy
        + coefficients @ sxx @ coefficients
        + 2.0 * intercept * (coefficients @ sx)
        + count * intercept * intercept
    )


def _fold_moments(
    estimator: object,
    ds: Dataset,
    features: Sequence[str],
    target: str,
    cv: int,
    seed: int,
):
    """The per-fold moment blocks and their total, from one grouped aggregate.

    Shared by every cross-validated linear model here - ridge, the lasso, the elastic net -
    because they all fit from the same moments and all score from them. Writing it twice
    would be the copy-paste the contract forbids, so it lives with the model that needed it
    first and the L1 module imports it.

    Args:
        estimator: The estimator doing the fitting, named in any error.
        ds: The training dataset.
        features: The predictor columns.
        target: The column being predicted.
        cv: How many folds to split into.
        seed: Seed for the content-hash fold assignment.

    Returns:
        The per-fold ``(count, sums, products)`` blocks, and the same three summed over all
        of them.

    Raises:
        PlanError: If a column is not numeric, or too few rows remain to fit.
        ColumnNotFoundError: If a named column is missing.
    """
    from batcher.api.dataset._build import split_key
    from batcher.plan.functions.aggregate import sum as sum_

    for name in (*features, target):
        if name not in ds.columns:
            from batcher._internal.errors import ColumnNotFoundError, unknown_message

            raise ColumnNotFoundError(
                unknown_message("column", name, ds.columns, hint="Pass an existing column.")
            )
    require_numeric(estimator, ds, features)
    require_numeric(estimator, ds, [target], role="target")

    d = len(features)
    names = [*features, target]
    columns = [col(n).cast("float64") for n in names]
    fold = (split_key(ds, names, seed) * lit(float(cv))).floor().cast("int64")
    aggregates: dict[str, object] = {"__bt_n": col(target).count()}
    for i, left in enumerate(columns):
        aggregates[f"__bt_s{i}"] = sum_(left)
        for j in range(i, len(columns)):
            aggregates[f"__bt_p{i}_{j}"] = sum_(left * columns[j])
    grouped = ds.with_columns(__bt_fold=fold).group_by("__bt_fold").agg(**aggregates).collect()

    blocks = [
        _moment_blocks({k: grouped.column(k)[r].as_py() for k in aggregates}, d + 1)
        for r in range(grouped.num_rows)
    ]
    total_n = sum(b[0] for b in blocks)
    require_rows(estimator, int(total_n), d + 2, because="a linear fit needs more rows than terms")
    return blocks, (total_n, sum(b[1] for b in blocks), sum(b[2] for b in blocks))


def _cv_error(blocks, total_n: float, total_sums, total_products, alpha: float, d: int) -> float:
    """The mean held-out squared error for one penalty, over folds already reduced to moments.

    Each fold's *training* moments are the total minus that fold's own, which is the whole
    reason one aggregate suffices: moments add, so leaving a fold out is a subtraction rather
    than another pass.
    """
    error, held = 0.0, 0.0
    for count, sums, products in blocks:
        train_n = total_n - count
        if train_n <= d + 1 or count <= 0:
            # A fold holding almost everything, or nothing, cannot score a candidate.
            # Skipping it beats both a singular solve and a silently optimistic zero.
            continue
        coefficients, intercept = _ridge_from_moments(
            train_n, total_sums - sums, total_products - products, alpha, d
        )
        error += _sse_from_moments(count, sums, products, coefficients, intercept)
        held += count
    return error / held if held else float("inf")


class RidgeCV:
    """Ridge regression with the penalty chosen by cross-validation, in one pass over the data.

    Searching a penalty normally costs a fit per candidate per fold. Ridge does not need that.
    Its normal equations are built entirely from the first and second moments of the features
    and the target, and **those moments do not depend on alpha** — so every candidate can be
    solved from the same numbers. The held-out error expands into the same moments too, so
    scoring a candidate on a fold reads no rows either.

    What that leaves is a single grouped aggregate: the moments per fold, computed once. Every
    fold's training moments are the total minus that fold's, because moments are additive, and
    every ``(fold, alpha)`` pair is then arithmetic on small matrices. Five folds and twenty
    candidates cost one pass, not a hundred.

    That additivity is the same property that makes the operator distributable, so the search
    behaves identically on one node and on a cluster. Folds are assigned by hashing the row's
    own values, so a row lands in the same fold however the data is partitioned.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import RidgeCV
            >>> ds = bt.from_pydict(
            ...     {"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ...      "y": [2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.9, 16.2]}
            ... )
            >>> model = RidgeCV(["x"], "y", alphas=(0.01, 1.0, 100.0), cv=4).fit(ds)
            >>> model.alpha_ in (0.01, 1.0, 100.0)
            True
            >>> round(model.predict(ds).to_pydict()["prediction"][0], 1)
            2.1

    Args:
        features: The predictor columns.
        target: The column to predict.
        alphas: The candidate L2 penalties to choose between.
        cv: How many cross-validation folds to score each candidate on.
        seed: Seed for the content-hash fold assignment.
        output_column: The name of the prediction column `predict` appends.

    Raises:
        PlanError: If `features` is empty or longer than `MAX_CV_FEATURES`, if `alphas` is
            empty or holds a negative penalty, or if `cv` is below two.
    """

    __slots__ = (
        "alpha_",
        "alphas",
        "coef_",
        "cv",
        "features",
        "intercept_",
        "output_column",
        "scores_",
        "seed",
        "target",
    )

    def __init__(
        self,
        features: Sequence[str],
        target: str,
        *,
        alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
        cv: int = 5,
        seed: int = 0,
        output_column: str = "prediction",
    ) -> None:
        self.features = list(features)
        if not self.features:
            raise PlanError("RidgeCV needs at least one feature column.")
        if len(self.features) > MAX_CV_FEATURES:
            raise PlanError(
                f"RidgeCV lowers to one aggregate per pair of terms, so {len(self.features)} "
                f"features would build a plan of "
                f"{(len(self.features) + 1) * (len(self.features) + 4) // 2} aggregates. "
                f"The ceiling is {MAX_CV_FEATURES}; select features first with "
                "SelectKBest, or fit Ridge at a fixed alpha."
            )
        self.alphas = [float(a) for a in alphas]
        if not self.alphas:
            raise PlanError("RidgeCV needs at least one candidate in alphas.")
        if any(a < 0 for a in self.alphas):
            raise PlanError(f"RidgeCV: every alpha must be non-negative, got {self.alphas}.")
        if cv < 2:
            raise PlanError(f"RidgeCV needs at least two folds to hold one out, got {cv}.")
        self.target = target
        self.cv = cv
        self.seed = seed
        self.output_column = output_column
        self.coef_: list[float] = []
        self.intercept_: float = 0.0
        self.alpha_: float = 0.0
        self.scores_: dict[float, float] = {}

    def fit(self, ds: Dataset) -> RidgeCV:
        """Score every candidate on every fold from a single grouped aggregate.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import RidgeCV
                >>> ds = bt.from_pydict(
                ...     {"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ...      "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
                ... )
                >>> model = RidgeCV(["x"], "y", alphas=(0.1, 10.0), cv=3).fit(ds)
                >>> sorted(model.scores_) == [0.1, 10.0]
                True

        Args:
            ds: The training dataset.

        Returns:
            ``self``, fitted at the best-scoring alpha over all the data.

        Raises:
            PlanError: If a feature or the target is not numeric, or too few rows remain.
            ColumnNotFoundError: If a named column is missing.
        """
        import numpy as np

        d = len(self.features)
        blocks, totals = _fold_moments(self, ds, self.features, self.target, self.cv, self.seed)
        total_n, total_sums, total_products = totals

        self.scores_ = {
            alpha: _cv_error(blocks, total_n, total_sums, total_products, alpha, d)
            for alpha in self.alphas
        }

        self.alpha_ = min(self.alphas, key=lambda a: (self.scores_[a], a))
        coefficients, intercept = _ridge_from_moments(
            total_n, total_sums, total_products, self.alpha_, d
        )
        self.coef_ = [float(c) for c in np.asarray(coefficients)]
        self.intercept_ = float(intercept)
        return self

    def predict(self, ds: Dataset) -> Dataset:
        """Append the prediction from the fit at the chosen penalty.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import RidgeCV
                >>> ds = bt.from_pydict(
                ...     {"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]}
                ... )
                >>> model = RidgeCV(["x"], "y", alphas=(0.001,), cv=2).fit(ds)
                >>> round(model.predict(ds).to_pydict()["prediction"][1], 3)
                4.0

        Args:
            ds: The dataset to score.

        Returns:
            A new lazy `Dataset` with the prediction column appended.
        """
        require_fitted(self, self.coef_)
        expression = linear_score(self.features, self.coef_, self.intercept_)
        return ds.with_columns(**{self.output_column: expression})
