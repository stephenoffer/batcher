"""Generalized linear models fitted by iteratively reweighted least squares.

A GLM keeps the linear predictor ``intercept + beta . x`` but wraps it in a link function so the
model fits a target that is not Gaussian — a count, a rate, a positive amount. There is no closed
form, but each Fisher-scoring step *is* one: the weighted gradient and Hessian are sums of per-row
products, so they are mergeable aggregates, and the small solve runs on the driver.

`TweedieRegressor` with a log link is the general form; `PoissonRegressor` (Tweedie power 1, the
count GLM) and `GammaRegressor` (power 2, the positive-continuous GLM) are the two special cases
that come up often enough to name. The logit GLM (`LogisticRegression`) lives beside the linear
models it most resembles in `batcher.ml.linear`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml._estimator import linear_score, require_fitted
from batcher.plan.expr_ir.constructors import col, lit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["GammaRegressor", "PoissonRegressor", "TweedieRegressor"]


class TweedieRegressor:
    """Tweedie regression with a log link — the GLM spanning counts and positive amounts.

    One `power` parameter selects the target's distribution: 1 is Poisson (counts), 2 is gamma
    (positive amounts), and a power in ``(1, 2)`` is the compound Poisson-gamma of a target that is
    *exactly zero* for many rows and positive for the rest — an insurance pure premium, a
    per-customer spend, any "mostly zero, some positive" quantity. With a log link it models
    ``E[y] = exp(intercept + beta . x)``, fitted by Fisher-scoring IRLS with the Tweedie weight
    ``mu^(2 - power)`` and an L2 penalty matching scikit-learn's ``TweedieRegressor``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.glm import TweedieRegressor
            >>> ds = bt.from_pydict({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 3.0, 8.0]})
            >>> model = TweedieRegressor(["x"], "y", power=1.5, alpha=0.0).fit(ds)
            >>> model.coef_[0] > 0
            True

    Args:
        features: The predictor columns.
        target: The non-negative target.
        power: The Tweedie power in ``[1, 2]`` — 1 is Poisson, 2 is gamma, between is compound.
        alpha: The L2 penalty strength (scikit-learn's convention, scaled by the row count).
        max_iter: The maximum number of IRLS iterations.
        tol: The convergence tolerance on the coefficient update's max absolute change.
        output_column: The name of the predicted-mean column `predict` appends.
    """

    __slots__ = (
        "alpha",
        "coef_",
        "features",
        "intercept_",
        "max_iter",
        "n_iter_",
        "output_column",
        "power",
        "target",
        "tol",
    )

    def __init__(
        self,
        features: Sequence[str],
        target: str,
        *,
        power: float = 1.5,
        alpha: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-8,
        output_column: str = "prediction",
    ) -> None:
        self.features = list(features)
        if not self.features:
            raise PlanError(f"{type(self).__name__} needs at least one feature column.")
        if alpha < 0:
            raise PlanError(f"alpha must be non-negative, got {alpha}.")
        if not 1.0 <= power <= 2.0:
            raise PlanError(f"power must be in [1, 2] for the log-link Tweedie, got {power}.")
        self.target = target
        self.power = power
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.output_column = output_column
        self.coef_: list[float] = []
        self.intercept_: float = 0.0
        self.n_iter_: int = 0

    def _eta(self, coefficients: list[float], intercept: float):
        """The linear predictor expression ``intercept + beta . x``."""
        return linear_score(self.features, coefficients, intercept)

    def fit(self, ds: Dataset) -> TweedieRegressor:
        """Fit the coefficients by Fisher-scoring iteratively reweighted least squares.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.glm import TweedieRegressor
                >>> ds = bt.from_pydict({"x": [0.0, 1.0, 2.0], "y": [0.0, 2.0, 5.0]})
                >>> TweedieRegressor(["x"], "y", power=1.5, alpha=0.0).fit(ds).n_iter_ > 0
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
        n = ds.count()
        penalty = self.alpha * n
        beta = np.zeros(m)
        for iteration in range(self.max_iter):
            eta = self._eta(list(beta[1:]), float(beta[0]))
            mu = eta.exp()
            weight = mu.pow(lit(2.0 - self.power))
            working = eta + (col(self.target) - mu) / mu
            aggregates = {}
            for j in range(m):
                aggregates[f"b{j}"] = sum_(weight * working * terms[j])
                for k in range(j, m):
                    aggregates[f"a{j}_{k}"] = sum_(weight * terms[j] * terms[k])
            row = ds.agg(**aggregates).collect()
            matrix = np.zeros((m, m))
            rhs = np.zeros(m)
            for j in range(m):
                rhs[j] = float(row.column(f"b{j}")[0].as_py())
                for k in range(j, m):
                    value = float(row.column(f"a{j}_{k}")[0].as_py())
                    matrix[j, k] = matrix[k, j] = value
            matrix[1:, 1:] += penalty * np.eye(m - 1)
            new_beta = np.linalg.solve(matrix + 1e-12 * np.eye(m), rhs)
            self.n_iter_ = iteration + 1
            if np.max(np.abs(new_beta - beta)) < self.tol:
                beta = new_beta
                break
            beta = new_beta
        self.intercept_ = float(beta[0])
        self.coef_ = [float(c) for c in beta[1:]]
        return self

    def predict(self, ds: Dataset) -> Dataset:
        """Append the predicted mean ``exp(intercept + beta . x)`` as a new column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.glm import TweedieRegressor
                >>> ds = bt.from_pydict({"x": [0.0, 1.0, 2.0], "y": [0.0, 2.0, 5.0]})
                >>> model = TweedieRegressor(["x"], "y", power=1.5, alpha=0.0).fit(ds)
                >>> all(v > 0 for v in model.predict(ds).to_pydict()["prediction"])
                True

        Args:
            ds: The dataset to score.

        Returns:
            A new lazy `Dataset` with the predicted-mean column appended.
        """
        require_fitted(self, self.coef_)
        return ds.with_columns(**{self.output_column: self._eta(self.coef_, self.intercept_).exp()})


class PoissonRegressor(TweedieRegressor):
    """Poisson regression for a non-negative count target — the ``power = 1`` Tweedie GLM.

    Models ``E[y] = exp(intercept + beta . x)`` with a log link, which keeps the predicted rate
    positive and multiplicative in the features — the right shape for event counts, claim
    frequencies, or arrivals, where ordinary least squares would happily predict a negative count.
    Matches scikit-learn's ``PoissonRegressor``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.glm import PoissonRegressor
            >>> ds = bt.from_pydict({"x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 2.0, 4.0, 8.0]})
            >>> model = PoissonRegressor(["x"], "y", alpha=0.0).fit(ds)
            >>> model.coef_[0] > 0  # more x means a higher expected count
            True

    Args:
        features: The predictor columns.
        target: The non-negative count target.
        alpha: The L2 penalty strength (scikit-learn's convention, scaled by the row count).
        max_iter: The maximum number of IRLS iterations.
        tol: The convergence tolerance on the coefficient update's max absolute change.
        output_column: The name of the predicted-rate column `predict` appends.
    """

    __slots__ = ()

    def __init__(
        self,
        features: Sequence[str],
        target: str,
        *,
        alpha: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-8,
        output_column: str = "prediction",
    ) -> None:
        super().__init__(
            features,
            target,
            power=1.0,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            output_column=output_column,
        )


class GammaRegressor(TweedieRegressor):
    """Gamma regression for a positive, right-skewed target — the ``power = 2`` Tweedie GLM.

    Models a positive amount whose variance grows with the square of its mean — a claim size, a
    duration, a spend — where the constant-variance assumption of least squares is wrong. With a
    log link it models ``E[y] = exp(intercept + beta . x)``, matching scikit-learn's
    ``GammaRegressor``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.glm import GammaRegressor
            >>> ds = bt.from_pydict({"x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 2.0, 4.0, 8.0]})
            >>> model = GammaRegressor(["x"], "y", alpha=0.0).fit(ds)
            >>> model.coef_[0] > 0
            True

    Args:
        features: The predictor columns.
        target: The positive continuous target.
        alpha: The L2 penalty strength (scikit-learn's convention, scaled by the row count).
        max_iter: The maximum number of IRLS iterations.
        tol: The convergence tolerance on the coefficient update's max absolute change.
        output_column: The name of the predicted-mean column `predict` appends.
    """

    __slots__ = ()

    def __init__(
        self,
        features: Sequence[str],
        target: str,
        *,
        alpha: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-8,
        output_column: str = "prediction",
    ) -> None:
        super().__init__(
            features,
            target,
            power=2.0,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            output_column=output_column,
        )
