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
from batcher.plan.expr_ir.constructors import col, lit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["ElasticNet", "Lasso"]


def _soft_threshold(value: float, amount: float) -> float:
    """The soft-thresholding operator ``sign(value) * max(|value| - amount, 0)``."""
    if value > amount:
        return value - amount
    if value < -amount:
        return value + amount
    return 0.0


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
        d = len(self.features)
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
        beta = np.zeros(d)
        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1.0 - self.l1_ratio)
        for iteration in range(self.max_iter):
            change = 0.0
            for j in range(d):
                if gram[j, j] + l2 == 0:
                    continue
                rho = xty[j] - gram[j] @ beta + gram[j, j] * beta[j]
                new = _soft_threshold(rho, l1) / (gram[j, j] + l2)
                change = max(change, abs(new - beta[j]))
                beta[j] = new
            self.n_iter_ = iteration + 1
            if change < self.tol:
                break
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
        if not self.coef_:
            raise PlanError(f"{type(self).__name__} must be fitted before predict.")
        expression = lit(self.intercept_)
        for weight, name in zip(self.coef_, self.features, strict=True):
            expression = expression + lit(weight) * col(name)
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
