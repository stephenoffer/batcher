"""The Yeo-Johnson power transform and its one-pass maximum-likelihood fit.

Split from the other shape transforms because of how it is *fitted*, not what it computes.
Finding the power that makes a column most Gaussian is a maximum-likelihood problem, and
every implementation solves it with an optimizer that re-reads the data once per iteration.

That is unnecessary here. The profile likelihood at a fixed lambda is
``-n/2 * ln(var(transformed)) + (lambda - 1) * sum(sign(x) * ln(|x| + 1))``, and every term
of it is an aggregate. Evaluating the whole candidate grid is therefore *one* aggregate
carrying forty-one variance terms — one scan, whatever the grid's resolution — after which
the driver picks the maximum from forty-one numbers.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from batcher.ml.preprocessors.base import Preprocessor, columns_arg, fit_aggregate
from batcher.plan.expr_ir.constructors import col, lit, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset
    from batcher.plan.expr_ir import Expr

__all__ = ["BoxCoxTransformer", "PowerTransformer", "box_cox", "yeo_johnson"]


#: Candidate Yeo-Johnson lambdas. The profile likelihood is evaluated at all of them in one
#: pass, so the grid's width costs nothing in scans; -2..2 in steps of 0.1 is the range
#: scikit-learn's optimizer searches and 0.1 is finer than the parameter is identifiable to.
_LAMBDA_GRID = tuple(round(-2.0 + 0.1 * i, 1) for i in range(41))


def _best_lambda(cell: dict[str, float], name: str) -> float:
    """The grid lambda maximizing the profile likelihood for one column.

    Shared by both transforms because the objective is the same function of the aggregates:
    the family only changes how the variance and the Jacobian terms were *measured*, not how
    the maximum is picked out of them.
    """
    count = cell.get(f"{name}__n") or 0
    jacobian = cell.get(f"{name}__jac") or 0.0
    best_lambda, best_score = 0.0, -math.inf
    for index, lam in enumerate(_LAMBDA_GRID):
        variance = cell.get(f"{name}__v{index}")
        if variance is None or variance <= 0.0 or not math.isfinite(variance):
            continue
        score = -0.5 * count * math.log(variance) + (lam - 1.0) * jacobian
        if math.isfinite(score) and score > best_score:
            best_lambda, best_score = lam, score
    return best_lambda


class _GridPowerTransformer(Preprocessor):
    """The one-pass grid fit both power transforms share.

    `PowerTransformer` and `BoxCoxTransformer` differ only in the transform they apply and
    the Jacobian term its likelihood carries; the fit, the standardization pass, and the
    projection are identical. They live here so the two families cannot drift apart — the
    likelihood is subtle enough that a fix applied to one copy and not the other would be
    invisible until a lambda came out wrong.

    A subclass supplies `_transform_expr` and `_jacobian`, and may override `_validate` to
    reject a column its family is undefined on. The fit measures each column's minimum
    whether or not the family needs it: it rides along in an aggregate already carrying
    forty-one variance terms, so it costs nothing measurable, and paying for it
    unconditionally is cheaper than a hook that exists for one subclass.
    """

    numeric_only = True

    __slots__ = ("columns", "lambdas_", "mean_", "scale_", "standardize")

    def __init__(self, columns: str | Sequence[str], *, standardize: bool = True) -> None:
        self.columns = columns_arg(columns, what=type(self).__name__)
        self.standardize = standardize
        self.lambdas_: dict[str, float] = {}
        self.mean_: dict[str, float] = {}
        self.scale_: dict[str, float] = {}

    @staticmethod
    def _transform_expr(value: Expr, lam: float) -> Expr:
        """The family's transform of `value` at power `lam`."""
        raise NotImplementedError

    @staticmethod
    def _jacobian(value: Expr) -> Expr:
        """The likelihood's Jacobian term, which is linear in lambda and so measured once."""
        raise NotImplementedError

    def _validate(self, cell: dict[str, float], name: str) -> None:
        """Reject a column the family is undefined on. The default admits everything."""

    def _fit_grid(self, ds: Dataset) -> None:
        """Evaluate every candidate lambda's likelihood in one pass and keep the maximum."""
        self._check_numeric(ds)
        aggregates: dict[str, Expr] = {}
        for name in self.columns:
            value = col(name)
            aggregates[f"{name}__n"] = value.count()
            aggregates[f"{name}__jac"] = self._jacobian(value)
            aggregates[f"{name}__min"] = value.min()
            for index, lam in enumerate(_LAMBDA_GRID):
                aggregates[f"{name}__v{index}"] = self._transform_expr(value, lam).var()
        cell = fit_aggregate(ds, aggregates)
        for name in self.columns:
            self._validate(cell, name)
            self.lambdas_[name] = _best_lambda(cell, name)
        self._fitted = True
        if self.standardize:
            self._fit_standardization(ds)

    def _fit_standardization(self, ds: Dataset) -> None:
        """Learn the transformed columns' mean and standard deviation, in one more pass."""
        transformed = self.transform(ds, standardize=False)
        aggregates = {}
        for name in self.columns:
            aggregates[f"{name}__m"] = col(name).mean()
            aggregates[f"{name}__s"] = col(name).std()
        cell = fit_aggregate(transformed, aggregates)
        for name in self.columns:
            mean = cell[f"{name}__m"]
            spread = cell[f"{name}__s"]
            self.mean_[name] = 0.0 if mean is None else float(mean)
            self.scale_[name] = 1.0 if not spread else float(spread)

    def _apply(self, ds: Dataset, standardize: bool | None) -> Dataset:
        """Project every fitted column through its transform, scaling when asked."""
        self._require_fitted()
        apply_scaling = self.standardize if standardize is None else standardize
        projections = {}
        for name in self.columns:
            expression = self._transform_expr(col(name), self.lambdas_[name])
            if apply_scaling and name in self.scale_:
                expression = (expression - lit(self.mean_[name])) / lit(self.scale_[name])
            projections[name] = expression
        return ds.with_columns(**projections)


def yeo_johnson(value: Expr, lam: float) -> Expr:
    """The Yeo-Johnson transform of `value` at power `lam`, as an expression.

    The four-branch definition: the sign of the value picks the branch, and ``lam`` being 0
    (or 2, on the negative side) picks the logarithmic limit rather than the power form.
    Unlike Box-Cox it is defined for negative values, which is why it is the default here —
    a transform that raises on a negative number is useless on a real feature column.

    Args:
        value: The column expression to transform.
        lam: The power parameter.

    Returns:
        The transformed expression.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors.power import yeo_johnson
            >>> ds = bt.from_pydict({"x": [0.0, 1.0]})
            >>> ds.with_columns(t=yeo_johnson(bt.col("x"), 0.0)).to_pydict()["t"]
            [0.0, 0.6931471805599453]
    """
    positive = value >= lit(0.0)
    if lam == 0.0:
        upper = (value + lit(1.0)).ln()
    else:
        upper = ((value + lit(1.0)).pow(lit(lam)) - lit(1.0)) / lit(lam)
    if lam == 2.0:
        lower = -(lit(1.0) - value).ln()
    else:
        lower = -((lit(1.0) - value).pow(lit(2.0 - lam)) - lit(1.0)) / lit(2.0 - lam)
    return when(positive).then(upper).otherwise(lower)


class PowerTransformer(_GridPowerTransformer):
    """Make a column more Gaussian with a maximum-likelihood Yeo-Johnson power transform.

    The right answer for a skewed positive column — revenue, duration, count — where a
    linear model or a distance metric is downstream. Unlike a plain ``log1p`` it *chooses*
    the strength of the transform from the data, and unlike Box-Cox it accepts negative
    values.

    The lambda that maximizes the profile likelihood is normally found by an iterative
    optimizer, one pass over the data per iteration. Here every candidate lambda's
    likelihood is an aggregate, so the whole grid is evaluated **in a single pass** and the
    fit costs one scan regardless of how fine the grid is. The likelihood is
    ``-n/2 * ln(var(transformed)) + (lambda - 1) * sum(sign(x) * ln(|x| + 1))``, whose second
    term is linear in lambda and so is computed once.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import PowerTransformer
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]})
            >>> pre = PowerTransformer("x").fit(ds)
            >>> pre.lambdas_["x"] < 0.5
            True

    Args:
        columns: The numeric columns to transform (replaced in place).
        standardize: Also center and scale the transformed column to zero mean and unit
            variance, as scikit-learn does by default.
    """

    __slots__ = ()

    _transform_expr = staticmethod(yeo_johnson)

    @staticmethod
    def _jacobian(value: Expr) -> Expr:
        """``sum(sign(x) * ln(|x| + 1))`` — the Yeo-Johnson Jacobian, defined at any sign."""
        return (value.sign() * (value.abs() + lit(1.0)).ln()).sum()

    def fit(self, ds: Dataset) -> PowerTransformer:
        """Choose each column's lambda by profile likelihood, then learn its scaling.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import PowerTransformer
                >>> pre = PowerTransformer("x", standardize=False).fit(
                ...     bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0]})
                ... )
                >>> -2.0 <= pre.lambdas_["x"] <= 2.0
                True

        Args:
            ds: The dataset to learn from.

        Returns:
            ``self``, fitted.
        """
        self._fit_grid(ds)
        return self

    def transform(self, ds: Dataset, *, standardize: bool | None = None) -> Dataset:
        """Apply the fitted power transform (and standardization) to `ds`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import PowerTransformer
                >>> ds = bt.from_pydict({"x": [1.0, 2.0, 4.0, 8.0]})
                >>> out = PowerTransformer("x").fit_transform(ds).to_pydict()["x"]
                >>> abs(sum(out) / len(out)) < 1e-9
                True

        Args:
            ds: The dataset to transform.
            standardize: Override the constructor's `standardize` for this call; used
                internally to measure the un-standardized transform.

        Returns:
            A new lazy `Dataset` with the fitted columns transformed.
        """
        return self._apply(ds, standardize)


def box_cox(value: Expr, lam: float) -> Expr:
    """The Box-Cox transform of a strictly positive `value` at power `lam`, as an expression.

    ``(x**lam - 1) / lam`` for a nonzero `lam`, and ``ln(x)`` at the ``lam == 0`` limit. Box-Cox
    is only defined for positive values, which is what distinguishes it from `yeo_johnson`; on a
    column that can be zero or negative, use Yeo-Johnson instead.

    Args:
        value: The strictly positive column expression to transform.
        lam: The power parameter.

    Returns:
        The transformed expression.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors.power import box_cox
            >>> ds = bt.from_pydict({"x": [1.0, math.e]})
            >>> ds.with_columns(t=box_cox(bt.col("x"), 0.0)).to_pydict()["t"]
            [0.0, 1.0]
    """
    if lam == 0.0:
        return value.ln()
    return (value.pow(lit(lam)) - lit(1.0)) / lit(lam)


class BoxCoxTransformer(_GridPowerTransformer):
    """Make a strictly positive column more Gaussian with a maximum-likelihood Box-Cox transform.

    The classic normalizing transform for a positive, right-skewed feature — a price, a
    duration, a count strictly above zero. Like `PowerTransformer` it chooses the power that
    maximizes the profile likelihood, but on the Box-Cox family rather than Yeo-Johnson, which
    is the transform most statistics tooling means by "the Box-Cox transform" and the one to
    match when reproducing an existing analysis. It rejects a non-positive column rather than
    silently producing NaNs; reach for `PowerTransformer` when the column can be zero or negative.

    Every candidate lambda's likelihood is an aggregate, so the whole grid is evaluated in a
    single pass and the fit costs one scan. The likelihood is
    ``-n/2 * ln(var(transformed)) + (lambda - 1) * sum(ln(x))``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import BoxCoxTransformer
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]})
            >>> pre = BoxCoxTransformer("x").fit(ds)
            >>> pre.lambdas_["x"] < 0.5
            True

    Args:
        columns: The strictly positive columns to transform (replaced in place).
        standardize: Also center and scale each transformed column to zero mean, unit variance.
    """

    __slots__ = ()

    _transform_expr = staticmethod(box_cox)

    @staticmethod
    def _jacobian(value: Expr) -> Expr:
        """``sum(ln(x))`` — the Box-Cox Jacobian, which needs a strictly positive column."""
        return value.ln().sum()

    def _validate(self, cell: dict[str, float], name: str) -> None:
        """Raise when a column holds a value Box-Cox is undefined on, naming the minimum."""
        from batcher._internal.errors import PlanError

        minimum = cell.get(f"{name}__min")
        if minimum is not None and float(minimum) <= 0.0:
            raise PlanError(
                f"BoxCoxTransformer needs strictly positive values; column {name!r} has a "
                f"minimum of {float(minimum)}. Use PowerTransformer (Yeo-Johnson) instead."
            )

    def fit(self, ds: Dataset) -> BoxCoxTransformer:
        """Choose each column's lambda by profile likelihood, then learn its scaling.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import BoxCoxTransformer
                >>> pre = BoxCoxTransformer("x", standardize=False).fit(
                ...     bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0]})
                ... )
                >>> -2.0 <= pre.lambdas_["x"] <= 2.0
                True

        Args:
            ds: The dataset to learn from.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If a fitted column contains a non-positive value.
        """
        self._fit_grid(ds)
        return self

    def transform(self, ds: Dataset, *, standardize: bool | None = None) -> Dataset:
        """Apply the fitted Box-Cox transform (and standardization) to `ds`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import BoxCoxTransformer
                >>> ds = bt.from_pydict({"x": [1.0, 2.0, 4.0, 8.0]})
                >>> out = BoxCoxTransformer("x").fit_transform(ds).to_pydict()["x"]
                >>> abs(sum(out) / len(out)) < 1e-9
                True

        Args:
            ds: The dataset to transform.
            standardize: Override the constructor's `standardize` for this call.

        Returns:
            A new lazy `Dataset` with the fitted columns transformed.
        """
        return self._apply(ds, standardize)
