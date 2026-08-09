"""L1-regularized linear models — sparse coefficient selection by coordinate descent.

Where ridge regression (`batcher.ml.linear.Ridge`) shrinks every coefficient toward zero, an L1
penalty drives some of them *exactly* to zero, so the fit doubles as feature selection: the model
picks the columns that matter and discards the rest. That is what makes the lasso and the elastic
net the tools of choice for a wide, correlated feature table where interpretability matters.

The fit is efficient because coordinate descent only needs two summaries of the data — the
centered Gram matrix and the feature-target covariances — both computed in a single scan. The
descent then runs entirely on the driver over those small matrices, and because the objective is
strictly convex it converges to the one global minimizer, so the coefficients match scikit-learn's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml._estimator import (
    linear_score,
    require_fitted,
    require_numeric,
    require_rows,
)
from batcher.plan.expr_ir.constructors import col

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from batcher.api.dataset import Dataset

__all__ = ["ElasticNet", "Lasso"]


def _soft_threshold(value: float, amount: float) -> float:
    """The soft-thresholding operator ``sign(value) * max(|value| - amount, 0)``."""
    if value > amount:
        return value - amount
    if value < -amount:
        return value + amount
    return 0.0


def _descend(
    gram, xty, alpha: float, l1_ratio: float, max_iter: int, tol: float
) -> tuple[NDArray, int]:
    """Coordinate descent for the elastic net, over a Gram matrix rather than over rows.

    Pulled out of `ElasticNet.fit` so the cross-validated search can reuse it. Everything it
    touches is ``d x d``, which is what lets a whole penalty path be explored without going
    back to the data.

    Args:
        gram: The centered feature covariance matrix.
        xty: The centered feature-target covariances.
        alpha: The overall penalty strength.
        l1_ratio: The share of the penalty that is L1; 1.0 is the lasso.
        max_iter: The sweep ceiling.
        tol: The coefficient change below which the sweep stops.

    Returns:
        The fitted coefficients and the number of sweeps taken.
    """
    import numpy as np

    d = len(xty)
    beta = np.zeros(d)
    l1 = alpha * l1_ratio
    l2 = alpha * (1.0 - l1_ratio)
    taken = 0
    for iteration in range(max_iter):
        change = 0.0
        for j in range(d):
            if gram[j, j] + l2 == 0:
                continue
            rho = xty[j] - gram[j] @ beta + gram[j, j] * beta[j]
            new = _soft_threshold(rho, l1) / (gram[j, j] + l2)
            change = max(change, abs(new - beta[j]))
            beta[j] = new
        taken = iteration + 1
        if change < tol:
            break
    return beta, taken


class ElasticNet:
    """Linear regression with a combined L1 and L2 penalty, fitted by coordinate descent.

    Minimizes ``(1/2n)||y - Xb||^2 + alpha * l1_ratio * ||b||_1 + (alpha/2)(1 - l1_ratio)||b||^2``.
    The L1 term selects features by zeroing their coefficients; the L2 term keeps the fit stable
    when features are correlated (the case pure lasso handles badly, arbitrarily picking one of a
    correlated group). Set `l1_ratio` to 1 for a pure lasso, toward 0 for something closer to ridge.
    Reproduces scikit-learn's ``ElasticNet`` coefficients, sparsity pattern, and intercept.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.sparse_linear import ElasticNet
            >>> ds = bt.from_pydict(
            ...     {"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 0.0, 1.0, 0.0],
            ...      "y": [2.0, 4.0, 6.0, 8.0]}
            ... )
            >>> model = ElasticNet(["a", "b"], "y", alpha=0.05).fit(ds)
            >>> model.coef_[0] > model.coef_[1]  # 'a' drives y, 'b' is noise
            True

    Args:
        features: The predictor columns.
        target: The column to predict.
        alpha: The overall penalty strength; 0 recovers ordinary least squares.
        l1_ratio: The share of the penalty that is L1, in ``[0, 1]``.
        max_iter: The maximum number of coordinate-descent sweeps.
        tol: The convergence tolerance on the largest coefficient change in a sweep.
        output_column: The name of the prediction column `predict` appends.
    """

    __slots__ = (
        "alpha",
        "coef_",
        "features",
        "intercept_",
        "l1_ratio",
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
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-7,
        output_column: str = "prediction",
    ) -> None:
        self.features = list(features)
        if not self.features:
            raise PlanError("ElasticNet needs at least one feature column.")
        if alpha < 0:
            raise PlanError(f"alpha must be non-negative, got {alpha}.")
        if not 0.0 <= l1_ratio <= 1.0:
            raise PlanError(f"l1_ratio must be in [0, 1], got {l1_ratio}.")
        self.target = target
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.output_column = output_column
        self.coef_: list[float] = []
        self.intercept_: float = 0.0
        self.n_iter_: int = 0

    def fit(self, ds: Dataset) -> ElasticNet:
        """Learn the coefficients by coordinate descent over the centered Gram matrix.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.sparse_linear import ElasticNet
                >>> ds = bt.from_pydict(
                ...     {"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0], "y": [1.0, 2.0, 3.0]}
                ... )
                >>> len(ElasticNet(["a", "b"], "y", alpha=0.1).fit(ds).coef_)
                2

        Args:
            ds: The training dataset.

        Returns:
            ``self``, fitted.

        Raises:
            ColumnNotFoundError: If a named column is missing.
        """
        import numpy as np

        from batcher.plan.functions.aggregate import covar_pop, mean

        columns = [*self.features, self.target]
        for name in columns:
            if name not in ds.columns:
                from batcher._internal.errors import ColumnNotFoundError, unknown_message

                raise ColumnNotFoundError(
                    unknown_message("column", name, ds.columns, hint="Pass an existing column.")
                )
        require_numeric(self, ds, self.features)
        require_numeric(self, ds, [self.target], role="target")
        d = len(self.features)
        require_rows(self, ds.count(), 2, because="its moments divide by n - 1")
        aggregates: dict[str, object] = {
            f"__m{i}": mean(col(name)) for i, name in enumerate(columns)
        }
        for i in range(d):
            aggregates[f"__gy{i}"] = covar_pop(col(self.features[i]), col(self.target))
            for j in range(i, d):
                aggregates[f"__g{i}_{j}"] = covar_pop(col(self.features[i]), col(self.features[j]))
        row = ds.agg(**aggregates).collect()
        means = [float(row.column(f"__m{i}")[0].as_py()) for i in range(d + 1)]
        gram = np.zeros((d, d))
        xty = np.zeros(d)
        for i in range(d):
            xty[i] = float(row.column(f"__gy{i}")[0].as_py())
            for j in range(i, d):
                gram[i, j] = gram[j, i] = float(row.column(f"__g{i}_{j}")[0].as_py())
        beta, self.n_iter_ = _descend(gram, xty, self.alpha, self.l1_ratio, self.max_iter, self.tol)
        self.coef_ = [float(b) for b in beta]
        self.intercept_ = means[d] - sum(beta[i] * means[i] for i in range(d))
        return self

    def predict(self, ds: Dataset) -> Dataset:
        """Append the linear prediction as a new column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.sparse_linear import ElasticNet
                >>> ds = bt.from_pydict(
                ...     {"a": [1.0, 2.0, 3.0], "b": [0.0, 0.0, 0.0], "y": [2.0, 4.0, 6.0]}
                ... )
                >>> model = ElasticNet(["a", "b"], "y", alpha=0.01).fit(ds)
                >>> len(model.predict(ds).to_pydict()["prediction"])
                3

        Args:
            ds: The dataset to score.

        Returns:
            A new lazy `Dataset` with the prediction column appended.
        """
        require_fitted(self, self.coef_)
        expression = linear_score(self.features, self.coef_, self.intercept_)
        return ds.with_columns(**{self.output_column: expression})


class Lasso(ElasticNet):
    """Linear regression with a pure L1 penalty — the ``l1_ratio = 1`` elastic net.

    Minimizes ``(1/2n)||y - Xb||^2 + alpha * ||b||_1``, driving the coefficients of irrelevant
    features to exactly zero. It is the sharpest of the linear feature selectors, though on a group
    of correlated features it keeps one arbitrarily and zeros the rest; `ElasticNet` with a smaller
    `l1_ratio` is steadier there. Reproduces scikit-learn's ``Lasso``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.sparse_linear import Lasso
            >>> ds = bt.from_pydict(
            ...     {"a": [1.0, 2.0, 3.0, 4.0], "b": [0.5, 0.2, 0.9, 0.1],
            ...      "y": [2.0, 4.0, 6.0, 8.0]}
            ... )
            >>> model = Lasso(["a", "b"], "y", alpha=0.1).fit(ds)
            >>> model.coef_[1] == 0.0  # the noise feature is zeroed out
            True

    Args:
        features: The predictor columns.
        target: The column to predict.
        alpha: The L1 penalty strength.
        max_iter: The maximum number of coordinate-descent sweeps.
        tol: The convergence tolerance on the largest coefficient change in a sweep.
        output_column: The name of the prediction column `predict` appends.
    """

    __slots__ = ()

    def __init__(
        self,
        features: Sequence[str],
        target: str,
        *,
        alpha: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-7,
        output_column: str = "prediction",
    ) -> None:
        super().__init__(
            features,
            target,
            alpha=alpha,
            l1_ratio=1.0,
            max_iter=max_iter,
            tol=tol,
            output_column=output_column,
        )


class ElasticNetCV:
    """Elastic net with the penalty chosen by cross-validation, in one pass over the data.

    The same saving `RidgeCV` gets, for the same reason. Coordinate descent never looks at a
    row: it works from the centered Gram matrix and the feature-target covariances, and those
    are moments that do not depend on the penalty. The held-out squared error expands into the
    same moments. So the whole search is one grouped aggregate of the moments per fold, and
    every ``(fold, alpha)`` pair is a descent over small matrices.

    Each fold's training moments are the total minus that fold's own, because moments add.
    That additivity is what makes the operator distributable, so the search behaves identically
    on one node and across a cluster, and folds come from a content hash of the row so a row
    lands in the same fold however the data is partitioned.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import ElasticNetCV
            >>> ds = bt.from_pydict(
            ...     {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ...      "b": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            ...      "y": [2.0, 4.1, 5.9, 8.0, 10.1, 12.0, 13.9, 16.1]}
            ... )
            >>> model = ElasticNetCV(["a", "b"], "y", alphas=(0.001, 1.0), cv=4).fit(ds)
            >>> model.alpha_ in (0.001, 1.0)
            True

    Args:
        features: The predictor columns.
        target: The column to predict.
        alphas: The candidate penalties to choose between.
        l1_ratio: The share of the penalty that is L1; 1.0 is the lasso, 0.0 is ridge.
        cv: How many cross-validation folds to score each candidate on.
        max_iter: The coordinate-descent sweep ceiling.
        tol: The coefficient change below which a descent stops.
        seed: Seed for the content-hash fold assignment.
        output_column: The name of the prediction column `predict` appends.

    Raises:
        PlanError: If `features` is empty or over `MAX_CV_FEATURES`, if `alphas` is empty or
            holds a negative penalty, if `l1_ratio` is outside ``[0, 1]``, or if `cv` is
            below two.
    """

    __slots__ = (
        "alpha_",
        "alphas",
        "coef_",
        "cv",
        "features",
        "intercept_",
        "l1_ratio",
        "max_iter",
        "output_column",
        "scores_",
        "seed",
        "target",
        "tol",
    )

    def __init__(
        self,
        features: Sequence[str],
        target: str,
        *,
        alphas: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0),
        l1_ratio: float = 0.5,
        cv: int = 5,
        max_iter: int = 1000,
        tol: float = 1e-4,
        seed: int = 0,
        output_column: str = "prediction",
    ) -> None:
        from batcher.ml.linear import MAX_CV_FEATURES

        self.features = list(features)
        if not self.features:
            raise PlanError(f"{type(self).__name__} needs at least one feature column.")
        if len(self.features) > MAX_CV_FEATURES:
            raise PlanError(
                f"{type(self).__name__} lowers to one aggregate per pair of terms, so "
                f"{len(self.features)} features exceed the ceiling of {MAX_CV_FEATURES}. "
                "Select features first with SelectKBest, or fit at a fixed alpha."
            )
        self.alphas = [float(a) for a in alphas]
        if not self.alphas:
            raise PlanError(f"{type(self).__name__} needs at least one candidate in alphas.")
        if any(a < 0 for a in self.alphas):
            raise PlanError(
                f"{type(self).__name__}: every alpha must be non-negative, got {self.alphas}."
            )
        if not 0.0 <= l1_ratio <= 1.0:
            raise PlanError(f"l1_ratio must be between 0 and 1, got {l1_ratio}.")
        if cv < 2:
            raise PlanError(
                f"{type(self).__name__} needs at least two folds to hold one out, got {cv}."
            )
        self.target = target
        self.l1_ratio = float(l1_ratio)
        self.cv = cv
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.output_column = output_column
        self.coef_: list[float] = []
        self.intercept_: float = 0.0
        self.alpha_: float = 0.0
        self.scores_: dict[float, float] = {}

    def _solve(self, count: float, sums, products, alpha: float):
        """Descend to the coefficients for one penalty, from moments alone."""
        import numpy as np

        d = len(self.features)
        centered = products - np.outer(sums, sums) / count
        beta, _ = _descend(
            centered[:d, :d] / count,
            centered[:d, d] / count,
            alpha,
            self.l1_ratio,
            self.max_iter,
            self.tol,
        )
        means = sums / count
        return beta, float(means[d] - beta @ means[:d])

    def fit(self, ds: Dataset) -> ElasticNetCV:
        """Score every candidate on every fold from a single grouped aggregate.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import ElasticNetCV
                >>> ds = bt.from_pydict(
                ...     {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ...      "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
                ... )
                >>> model = ElasticNetCV(["a"], "y", alphas=(0.01, 1.0), cv=3).fit(ds)
                >>> sorted(model.scores_) == [0.01, 1.0]
                True

        Args:
            ds: The training dataset.

        Returns:
            ``self``, fitted at the best-scoring alpha over all the data.

        Raises:
            PlanError: If a feature or the target is not numeric, or too few rows remain.
            ColumnNotFoundError: If a named column is missing.
        """
        from batcher.ml.linear import _fold_moments, _sse_from_moments

        blocks, totals = _fold_moments(self, ds, self.features, self.target, self.cv, self.seed)
        total_n, total_sums, total_products = totals
        d = len(self.features)

        self.scores_ = {}
        for alpha in self.alphas:
            error, held = 0.0, 0.0
            for count, sums, products in blocks:
                train_n = total_n - count
                if train_n <= d + 1 or count <= 0:
                    continue
                beta, intercept = self._solve(
                    train_n, total_sums - sums, total_products - products, alpha
                )
                error += _sse_from_moments(count, sums, products, beta, intercept)
                held += count
            self.scores_[alpha] = error / held if held else float("inf")

        self.alpha_ = min(self.alphas, key=lambda a: (self.scores_[a], a))
        beta, intercept = self._solve(total_n, total_sums, total_products, self.alpha_)
        self.coef_ = [float(b) for b in beta]
        self.intercept_ = intercept
        return self

    def predict(self, ds: Dataset) -> Dataset:
        """Append the prediction from the fit at the chosen penalty.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import ElasticNetCV
                >>> ds = bt.from_pydict(
                ...     {"a": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]}
                ... )
                >>> model = ElasticNetCV(["a"], "y", alphas=(0.0001,), cv=2).fit(ds)
                >>> round(model.predict(ds).to_pydict()["prediction"][1], 2)
                4.0

        Args:
            ds: The dataset to score.

        Returns:
            A new lazy `Dataset` with the prediction column appended.
        """
        require_fitted(self, self.coef_)
        expression = linear_score(self.features, self.coef_, self.intercept_)
        return ds.with_columns(**{self.output_column: expression})


class LassoCV(ElasticNetCV):
    """Lasso with the penalty chosen by cross-validation, in one pass over the data.

    `ElasticNetCV` with `l1_ratio` fixed at 1.0, so the whole penalty is L1 and the search
    selects features as well as a strength: a larger alpha zeroes more coefficients outright.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import LassoCV
            >>> ds = bt.from_pydict(
            ...     {"signal": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ...      "noise": [0.3, -0.1, 0.2, -0.4, 0.1, 0.5, -0.2, 0.0],
            ...      "y": [2.0, 4.1, 5.9, 8.0, 10.1, 12.0, 13.9, 16.1]}
            ... )
            >>> model = LassoCV(["signal", "noise"], "y", alphas=(0.001, 0.1), cv=4).fit(ds)
            >>> model.l1_ratio
            1.0

    Args:
        features: The predictor columns.
        target: The column to predict.
        alphas: The candidate L1 penalties to choose between.
        cv: How many cross-validation folds to score each candidate on.
        max_iter: The coordinate-descent sweep ceiling.
        tol: The coefficient change below which a descent stops.
        seed: Seed for the content-hash fold assignment.
        output_column: The name of the prediction column `predict` appends.
    """

    __slots__ = ()

    def __init__(
        self,
        features: Sequence[str],
        target: str,
        *,
        alphas: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0),
        cv: int = 5,
        max_iter: int = 1000,
        tol: float = 1e-4,
        seed: int = 0,
        output_column: str = "prediction",
    ) -> None:
        super().__init__(
            features,
            target,
            alphas=alphas,
            l1_ratio=1.0,
            cv=cv,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
            output_column=output_column,
        )
