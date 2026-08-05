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

import warnings
from typing import TYPE_CHECKING

from batcher._internal.errors import DataWarning, PlanError
from batcher.ml._estimator import (
    linear_score,
    require_fitted,
    require_numeric,
    require_rows,
)
from batcher.plan.expr_ir.constructors import col, lit, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["GammaRegressor", "HuberRegressor", "PoissonRegressor", "TweedieRegressor"]

# How many times a diverging Fisher-scoring step is halved before the fit gives up. Eight
# halvings shrink the step to 1/256 of the Newton direction, which is far past the point a
# genuinely descending step would be found.
_MAX_STEP_HALVINGS = 8


def _safe_step(matrix, rhs, beta, alpha: float):
    """One Fisher-scoring step, damped so a diverging fit fails loudly instead of returning NaN.

    IRLS under a log link is not globally convergent: with no regularization the linear
    predictor can grow until ``exp(eta)`` overflows, and every subsequent aggregate is then
    NaN. Unguarded, the loop burns its whole iteration budget and returns NaN coefficients,
    which `predict` happily turns into a column of NaN — a wrong answer that never raises.

    Damping the step keeps the ordinary case converging; the explicit non-finite check turns
    the pathological case into a typed error naming the knob that fixes it.
    """
    import numpy as np

    if not (np.all(np.isfinite(matrix)) and np.all(np.isfinite(rhs))):
        raise PlanError(
            "GLM fit diverged: the weighted normal equations are no longer finite. The linear "
            "predictor overflowed `exp`, usually because the target spans many orders of "
            f"magnitude at alpha={alpha}. Raise `alpha` (1.0 is a safe starting point) or "
            "rescale the features."
        )
    step = np.linalg.solve(matrix + 1e-12 * np.eye(matrix.shape[0]), rhs) - beta
    for halving in range(_MAX_STEP_HALVINGS + 1):
        candidate = beta + step / (2.0**halving)
        # A step is usable only if it stays finite *and* keeps `exp(eta)` in range, which is
        # what the magnitude bound below stands in for.
        if np.all(np.isfinite(candidate)) and np.max(np.abs(candidate)) < 700.0:
            return candidate
    raise PlanError(
        "GLM fit diverged: no damped Fisher-scoring step kept the coefficients finite after "
        f"{_MAX_STEP_HALVINGS} halvings at alpha={alpha}. Raise `alpha` (1.0 is a safe "
        "starting point) or rescale the features."
    )


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
        "converged_",
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
        self.converged_: bool = False

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
        require_numeric(self, ds, self.features)
        require_numeric(self, ds, [self.target], role="target")
        self.converged_ = False
        terms = [lit(1.0), *[col(name) for name in self.features]]
        m = len(terms)
        n = ds.count()
        require_rows(self, n, m, because="IRLS needs one row per fitted term")
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
            new_beta = _safe_step(matrix, rhs, beta, self.alpha)
            self.n_iter_ = iteration + 1
            if np.max(np.abs(new_beta - beta)) < self.tol:
                beta = new_beta
                self.converged_ = True
                break
            beta = new_beta
        if not self.converged_:
            warnings.warn(
                f"{type(self).__name__} did not converge in {self.max_iter} iterations at "
                f"alpha={self.alpha}; the coefficients are the last iterate, not a fitted "
                "optimum. Raise `max_iter`, raise `alpha`, or rescale the features.",
                DataWarning,
                stacklevel=2,
            )
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


class HuberRegressor:
    """Least squares that a handful of wild target values cannot dominate.

    Squared error grows with the square of the residual, so one row that is off by a hundred
    counts as much as ten thousand rows that are off by one. That is why a single mistyped
    price or a stuck sensor can visibly tilt an ordinary fit, and why the fit gives no sign of
    it: the coefficients move, the residuals look worse everywhere, and nothing is reported.

    Huber's loss is squared near zero and *linear* past a threshold, so a far-away row keeps a
    bounded influence instead of a growing one. The fit is the same iteratively reweighted
    least squares the GLMs here use - each pass computes a weight per row and solves one small
    weighted system - so it is a handful of aggregates and no per-row Python.

    `epsilon` is where the loss turns linear, in units of the residual scale. Smaller is more
    robust and less efficient on clean data; 1.35 is the conventional default, chosen to
    retain about 95% of least squares' efficiency when the errors really are normal.

    The scale is re-estimated from the current residuals on every pass, because fixing it
    from the starting least-squares fit does not work: those residuals are already stretched
    by the very rows being guarded against, so the cutoff sits beyond them and every weight
    stays at one. Measured on a six-row example with one bad value, a fixed scale returned a
    "robust" slope of 51.9 against a true 2.

    That costs two extra aggregates per pass and can hit `max_iter` on a degenerate input -
    a dataset small enough that the retained rows fit *exactly* drives the scale towards
    zero, so the weights never quite settle. The fit warns when it stops on the cap rather
    than on convergence, and the coefficients are the last iterate.

    The scale comes from the median absolute deviation rather than being iterated
    alongside the coefficients. That keeps the fit to one aggregate per iteration and is the
    same simplification `sklearn.linear_model.HuberRegressor` avoids by fitting scale jointly,
    so coefficients here agree closely with sklearn's but are not bit-identical to them.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import HuberRegressor, LinearRegression
            >>> ds = bt.from_pydict(
            ...     {"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ...      "y": [2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 90.0]}
            ... )
            >>> round(LinearRegression(["x"], "y").fit(ds).coef_[0], 1)  # one bad row
            8.2
            >>> round(HuberRegressor(["x"], "y").fit(ds).coef_[0], 1)
            2.0

    Args:
        features: The predictor columns.
        target: The column to predict.
        epsilon: Where the loss turns linear, in residual-scale units. Must exceed 1.
        alpha: An L2 penalty on the coefficients, as for the GLMs here.
        max_iter: The reweighting ceiling.
        tol: The coefficient change below which the fit stops.
        output_column: The name of the prediction column `predict` appends.

    Raises:
        PlanError: If `features` is empty, `epsilon` is not above 1, or `alpha` is negative.
    """

    __slots__ = (
        "alpha",
        "coef_",
        "converged_",
        "epsilon",
        "features",
        "intercept_",
        "max_iter",
        "n_iter_",
        "output_column",
        "scale_",
        "target",
        "tol",
    )

    def __init__(
        self,
        features: Sequence[str],
        target: str,
        *,
        epsilon: float = 1.35,
        alpha: float = 0.0,
        max_iter: int = 100,
        tol: float = 1e-8,
        output_column: str = "prediction",
    ) -> None:
        self.features = list(features)
        if not self.features:
            raise PlanError("HuberRegressor needs at least one feature column.")
        if epsilon <= 1.0:
            raise PlanError(
                f"epsilon is where the loss turns linear, in residual-scale units, and must "
                f"be above 1; got {epsilon}. Below that the loss is linear everywhere and "
                "the fit is a median regression, not a Huber one."
            )
        if alpha < 0:
            raise PlanError(f"alpha must be non-negative, got {alpha}.")
        self.target = target
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.max_iter = max_iter
        self.tol = tol
        self.output_column = output_column
        self.coef_: list[float] = []
        self.intercept_: float = 0.0
        self.scale_: float = 1.0
        self.n_iter_: int = 0
        self.converged_: bool = False

    def fit(self, ds: Dataset) -> HuberRegressor:
        """Reweight and re-solve until the coefficients stop moving.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import HuberRegressor
                >>> ds = bt.from_pydict(
                ...     {"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]}
                ... )
                >>> model = HuberRegressor(["x"], "y").fit(ds)
                >>> round(model.coef_[0], 6), round(model.intercept_, 6)
                (2.0, 0.0)

        Args:
            ds: The training dataset.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If a column is not numeric, or there are fewer rows than terms.
            ColumnNotFoundError: If a named column is missing.
        """
        import numpy as np

        for name in (*self.features, self.target):
            if name not in ds.columns:
                from batcher._internal.errors import ColumnNotFoundError, unknown_message

                raise ColumnNotFoundError(
                    unknown_message("column", name, ds.columns, hint="Pass an existing column.")
                )
        require_numeric(self, ds, self.features)
        require_numeric(self, ds, [self.target], role="target")

        terms = [lit(1.0), *[col(name) for name in self.features]]
        m = len(terms)
        n = ds.count()
        require_rows(self, n, m, because="the weighted least squares needs one row per term")
        self.converged_ = False
        beta = _least_squares(ds, self.features, self.target, self.alpha, self)
        penalty = self.alpha * n
        for iteration in range(self.max_iter):
            # The scale is re-estimated from the *current* residuals, not fixed from the
            # starting fit. Fixing it does not work: the starting fit is ordinary least
            # squares, which the outliers have already pulled, so its residual spread is wide
            # enough that the cutoff sits beyond the outliers and every weight stays at one.
            # The fit then never moves off least squares - measured on a six-row example with
            # one bad value, the "robust" slope came back as 51.9 against the true 2.
            self.scale_ = _residual_scale(ds, self.features, self.target, beta)
            cutoff = self.epsilon * self.scale_
            residual = (col(self.target) - _linear_expression(terms, beta)).abs()
            # Inside the cutoff the row keeps its full weight and the loss is squared; outside
            # it the weight falls as 1/|r|, which is exactly what makes the loss linear there
            # and bounds how far one row can pull the fit.
            weight = when(residual <= lit(cutoff)).then(lit(1.0)).otherwise(lit(cutoff) / residual)
            matrix, rhs = _weighted_system(ds, terms, col(self.target), weight)
            matrix[1:, 1:] += penalty * np.eye(m - 1)
            new_beta = np.linalg.solve(matrix + 1e-12 * np.eye(m), rhs)
            self.n_iter_ = iteration + 1
            change = float(np.max(np.abs(new_beta - beta)))
            beta = new_beta
            if change < self.tol:
                self.converged_ = True
                break
        if not self.converged_:
            warnings.warn(
                f"HuberRegressor did not converge in {self.max_iter} iterations; the "
                "coefficients are the last iterate, not a fitted optimum. Raise `max_iter` "
                "or rescale the features.",
                DataWarning,
                stacklevel=2,
            )
        self.intercept_ = float(beta[0])
        self.coef_ = [float(c) for c in beta[1:]]
        return self

    def predict(self, ds: Dataset) -> Dataset:
        """Append the robust linear prediction as a new column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import HuberRegressor
                >>> ds = bt.from_pydict(
                ...     {"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]}
                ... )
                >>> model = HuberRegressor(["x"], "y").fit(ds)
                >>> round(model.predict(ds).to_pydict()["prediction"][2], 6)
                6.0

        Args:
            ds: The dataset to score.

        Returns:
            A new lazy `Dataset` with the prediction column appended.
        """
        require_fitted(self, self.coef_)
        return ds.with_columns(
            **{self.output_column: linear_score(self.features, self.coef_, self.intercept_)}
        )


def _weighted_system(ds: Dataset, terms: list, target, weight):
    """The weighted normal equations ``(X'WX, X'Wy)``, from one aggregate over the data.

    Pulled out of the reweighting loop so each pass reads as what it is: build the weights,
    solve one small system. Every entry is a `sum` over the rows, so the whole system is one
    grouped aggregate and the driver only ever sees a ``(m, m)`` matrix.
    """
    import numpy as np

    from batcher.plan.functions.aggregate import sum as sum_

    m = len(terms)
    aggregates = {}
    for j in range(m):
        aggregates[f"b{j}"] = sum_(weight * target * terms[j])
        for k in range(j, m):
            aggregates[f"a{j}_{k}"] = sum_(weight * terms[j] * terms[k])
    row = ds.agg(**aggregates).collect()
    matrix = np.zeros((m, m))
    rhs = np.zeros(m)
    for j in range(m):
        rhs[j] = float(row.column(f"b{j}")[0].as_py() or 0.0)
        for k in range(j, m):
            value = float(row.column(f"a{j}_{k}")[0].as_py() or 0.0)
            matrix[j, k] = matrix[k, j] = value
    return matrix, rhs


def _linear_expression(terms: list, beta) -> object:
    """``beta . terms`` as one expression, for a coefficient vector including the intercept."""
    expression = lit(float(beta[0])) * terms[0]
    for index in range(1, len(terms)):
        expression = expression + lit(float(beta[index])) * terms[index]
    return expression


def _least_squares(ds: Dataset, features: Sequence[str], target: str, alpha: float, owner: object):
    """An ordinary ridge fit, used as the starting point the reweighting refines."""
    import numpy as np

    from batcher.ml.linear import _solve

    coefficients, intercept = _solve(ds, list(features), target, alpha, owner)
    return np.array([intercept, *coefficients])


def _residual_scale(ds: Dataset, features: Sequence[str], target: str, beta) -> float:
    """The residual scale, from the median absolute deviation about the median.

    The median rather than the standard deviation, because the scale is what decides which
    rows count as outliers and a standard deviation is itself dominated by them - the estimate
    would widen to accommodate the very rows it is meant to identify. 1.4826 rescales a normal
    distribution's MAD to its standard deviation, so `epsilon` keeps its usual meaning.
    """
    from batcher.plan.functions.aggregate import median

    terms = [lit(1.0), *[col(name) for name in features]]
    residual = col(target) - _linear_expression(terms, beta)
    centre = ds.agg(m=median(residual)).collect().column("m")[0].as_py()
    spread = (
        ds.agg(s=median((residual - lit(float(centre or 0.0))).abs()))
        .collect()
        .column("s")[0]
        .as_py()
    )
    scale = 1.4826 * float(spread or 0.0)
    # A perfect fit, or a target with more than half its rows identical, gives a zero spread.
    # Every residual is then inside any cutoff, which is the correct answer - the fit has no
    # outliers to down-weight - and a positive floor keeps the weight expression finite.
    return scale if scale > 0.0 else 1.0
