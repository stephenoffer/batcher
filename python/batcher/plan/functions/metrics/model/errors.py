"""Regression error metrics as mergeable aggregate expressions.

Every function here is an *expression over aggregates* — ``mean``, ``sum``, ``count``,
``var`` composed with arithmetic — so it runs in the engine as one aggregate pass and a
projection, identical single-node and distributed. That is the whole point: model
evaluation stops being a `to_pandas()` on the driver and becomes a query.

Because they are expressions rather than results, they compose with `group_by`, which is
where the value is. ``ds.group_by("segment").agg(rmse=bt.rmse("y", "yhat"))`` gives per-
segment error over a billion rows in one pass — the query a scikit-learn `mean_squared_error`
cannot express at all.

Null handling follows the SQL convention the `regr_*` family already uses: a row is
included only when **both** the actual and the predicted value are present, achieved by
null-propagating arithmetic rather than an explicit filter, so the pairing survives
pushdown and partitioning.
"""

from __future__ import annotations

from batcher._internal.errors import PlanError, require_float
from batcher.plan.expr_ir.constructors import lit, when
from batcher.plan.expr_ir.core import AggExpr, Expr, IntoExpr, Lit
from batcher.plan.functions.aggregate import _as_column, count_if

__all__ = [
    "explained_variance",
    "huber_loss",
    "mae",
    "mape",
    "max_error",
    "mean_bias",
    "mean_percentage_error",
    "medae",
    "mse",
    "msle",
    "normalized_rmse",
    "pinball_loss",
    "r2",
    "rmse",
    "rmsle",
    "smape",
    "wape",
]


def _residual(y_true: IntoExpr, y_pred: IntoExpr) -> Expr:
    """``y_true - y_pred``, null on any row where either side is null."""
    return _as_column(y_true) - _as_column(y_pred)


def _mean_where(value: Expr, keep: Expr) -> Expr:
    """The mean of `value` over the rows `keep` selects, excluding every other row.

    Written as ``sum(masked) / count(kept)`` rather than as a null-producing CASE, because
    a metric that is undefined on some rows must drop them from the *denominator* too. A
    row excluded from the numerator but counted in the denominator silently deflates the
    metric, which is worse than either including it or erroring on it.

    Rows where `value` itself is null are dropped from both sides for the same reason.
    """
    kept = keep & value.is_not_null()
    return when(kept).then(value).otherwise(lit(0.0)).sum() / count_if(kept)


def mse(y_true: IntoExpr, y_pred: IntoExpr) -> Expr:
    """Mean squared error — ``mean((y - yhat)^2)``.

    Args:
        y_true: The observed values.
        y_pred: The predicted values.

    Returns:
        The mean squared error over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1.0, 2.0], "p": [1.0, 4.0]})
            >>> ds.agg(m=bt.mse("y", "p")).to_pydict()
            {'m': [2.0]}
    """
    error = _residual(y_true, y_pred)
    return (error * error).mean()


def rmse(y_true: IntoExpr, y_pred: IntoExpr) -> Expr:
    """Root mean squared error — ``sqrt(mse)``, in the units of the target.

    Args:
        y_true: The observed values.
        y_pred: The predicted values.

    Returns:
        The root mean squared error over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1.0, 2.0], "p": [1.0, 4.0]})
            >>> ds.agg(m=bt.rmse("y", "p")).to_pydict()
            {'m': [1.4142135623730951]}
    """
    return mse(y_true, y_pred).sqrt()


def mae(y_true: IntoExpr, y_pred: IntoExpr) -> Expr:
    """Mean absolute error — ``mean(|y - yhat|)``, the outlier-tolerant counterpart of `rmse`.

    Args:
        y_true: The observed values.
        y_pred: The predicted values.

    Returns:
        The mean absolute error over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1.0, 2.0], "p": [1.0, 4.0]})
            >>> ds.agg(m=bt.mae("y", "p")).to_pydict()
            {'m': [1.0]}
    """
    return _residual(y_true, y_pred).abs().mean()


def medae(y_true: IntoExpr, y_pred: IntoExpr) -> AggExpr:
    """Median absolute error — ``median(|y - yhat|)``, insensitive to a heavy tail.

    Args:
        y_true: The observed values.
        y_pred: The predicted values.

    Returns:
        The median absolute error over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1.0, 2.0, 3.0], "p": [1.0, 4.0, 3.0]})
            >>> ds.agg(m=bt.medae("y", "p")).to_pydict()
            {'m': [0.0]}
    """
    return _residual(y_true, y_pred).abs().median()


def max_error(y_true: IntoExpr, y_pred: IntoExpr) -> AggExpr:
    """The largest single absolute error — the worst case, which a mean hides.

    Args:
        y_true: The observed values.
        y_pred: The predicted values.

    Returns:
        The maximum absolute error over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1.0, 2.0], "p": [1.0, 4.0]})
            >>> ds.agg(m=bt.max_error("y", "p")).to_pydict()
            {'m': [2.0]}
    """
    return _residual(y_true, y_pred).abs().max()


def mean_bias(y_true: IntoExpr, y_pred: IntoExpr) -> Expr:
    """Mean signed error — ``mean(yhat - y)``. Positive means the model over-predicts.

    The metric every error magnitude hides: a model can have an excellent `rmse` and still
    be systematically 3% high, which matters as soon as the predictions are summed.

    Args:
        y_true: The observed values.
        y_pred: The predicted values.

    Returns:
        The mean signed error over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1.0, 2.0], "p": [2.0, 4.0]})
            >>> ds.agg(m=bt.mean_bias("y", "p")).to_pydict()
            {'m': [1.5]}
    """
    return (_as_column(y_pred) - _as_column(y_true)).mean()


def mape(y_true: IntoExpr, y_pred: IntoExpr) -> Expr:
    """Mean absolute percentage error — ``mean(|y - yhat| / |y|)``, as a fraction.

    Rows where the actual value is zero are **excluded**, not counted as infinite error:
    the ratio is undefined there and including them would make the metric meaningless
    rather than large. Use `smape` or `wape` when zeros are common.

    Args:
        y_true: The observed values.
        y_pred: The predicted values.

    Returns:
        The mean absolute percentage error as a fraction (0.1 is 10%).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [10.0, 100.0], "p": [11.0, 90.0]})
            >>> ds.agg(m=bt.mape("y", "p")).to_pydict()
            {'m': [0.1]}
    """
    actual = _as_column(y_true)
    ratio = _residual(y_true, y_pred).abs() / actual.abs()
    return _mean_where(ratio, actual != lit(0.0))


def smape(y_true: IntoExpr, y_pred: IntoExpr) -> Expr:
    """Symmetric MAPE — ``mean(2|y - yhat| / (|y| + |yhat|))``, bounded in ``[0, 2]``.

    The forecasting-competition metric: unlike `mape` it is finite when the actual is
    zero, and it does not punish over-prediction more heavily than under-prediction. A
    row where both values are zero contributes 0, not a division by zero.

    Args:
        y_true: The observed values.
        y_pred: The predicted values.

    Returns:
        The symmetric mean absolute percentage error, as a fraction.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [0.0, 100.0], "p": [0.0, 100.0]})
            >>> ds.agg(m=bt.smape("y", "p")).to_pydict()
            {'m': [0.0]}
    """
    actual, predicted = _as_column(y_true), _as_column(y_pred)
    denominator = actual.abs() + predicted.abs()
    ratio = (actual - predicted).abs() * lit(2.0) / denominator
    # A row where both values are zero is a perfect prediction, not a division by zero.
    return when(denominator == lit(0.0)).then(lit(0.0)).otherwise(ratio).mean()


def wape(y_true: IntoExpr, y_pred: IntoExpr) -> Expr:
    """Weighted absolute percentage error — ``sum(|y - yhat|) / sum(|y|)``.

    The demand-forecasting default, and the right choice when the series has zeros: it is
    a ratio of totals rather than a mean of ratios, so a single near-zero actual cannot
    dominate it the way it dominates `mape`.

    Args:
        y_true: The observed values.
        y_pred: The predicted values.

    Returns:
        The total absolute error divided by the total actual magnitude.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [0.0, 100.0], "p": [1.0, 90.0]})
            >>> ds.agg(m=bt.wape("y", "p")).to_pydict()
            {'m': [0.11]}
    """
    actual = _as_column(y_true)
    paired = (actual + _as_column(y_pred) * Lit(0)).abs()
    return _residual(y_true, y_pred).abs().sum() / paired.sum()


def r2(y_true: IntoExpr, y_pred: IntoExpr) -> Expr:
    """Coefficient of determination — ``1 - SS_res / SS_tot``, scikit-learn's ``r2_score``.

    The residual sum of squares is compared against the variance of the actuals, so 1.0 is
    a perfect fit, 0.0 matches predicting the mean, and a negative value is worse than
    that. `SS_tot` uses the engine's numerically stable variance aggregate rather than the
    naive sum of squares, which loses all precision on a large-magnitude target.

    Args:
        y_true: The observed values.
        y_pred: The predicted values.

    Returns:
        The coefficient of determination over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1.0, 2.0, 3.0], "p": [1.0, 2.0, 3.0]})
            >>> ds.agg(m=bt.r2("y", "p")).to_pydict()
            {'m': [1.0]}
    """
    error = _residual(y_true, y_pred)
    paired_true = _as_column(y_true) + _as_column(y_pred) * Lit(0)
    # var() is the sample variance (n-1); (n-1) * var is exactly the total sum of squares.
    total = paired_true.var() * (paired_true.count() - lit(1))
    return lit(1.0) - (error * error).sum() / total


def explained_variance(y_true: IntoExpr, y_pred: IntoExpr) -> Expr:
    """Explained variance — ``1 - var(y - yhat) / var(y)``.

    `r2` with the residual *mean* removed, so a model with a constant offset still scores
    well here. The gap between the two is exactly the squared bias, which makes the pair
    worth reporting together.

    Args:
        y_true: The observed values.
        y_pred: The predicted values.

    Returns:
        The explained-variance score over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1.0, 2.0, 3.0], "p": [2.0, 3.0, 4.0]})
            >>> ds.agg(m=bt.explained_variance("y", "p")).to_pydict()
            {'m': [1.0]}
    """
    paired_true = _as_column(y_true) + _as_column(y_pred) * Lit(0)
    return lit(1.0) - _residual(y_true, y_pred).var() / paired_true.var()


def msle(y_true: IntoExpr, y_pred: IntoExpr) -> Expr:
    """Mean squared logarithmic error — ``mean((ln(1+y) - ln(1+yhat))^2)``.

    The metric for a target spanning orders of magnitude, where being 10 off on a value of
    10 matters and being 10 off on a value of 10,000 does not. Defined only for
    non-negative values; a negative actual or prediction makes the row null rather than
    NaN, so one bad row does not destroy the whole metric.

    Args:
        y_true: The observed values (non-negative).
        y_pred: The predicted values (non-negative).

    Returns:
        The mean squared logarithmic error over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [0.0, 3.0], "p": [0.0, 3.0]})
            >>> ds.agg(m=bt.msle("y", "p")).to_pydict()
            {'m': [0.0]}
    """
    actual, predicted = _as_column(y_true), _as_column(y_pred)
    difference = (lit(1.0) + actual).ln() - (lit(1.0) + predicted).ln()
    return _mean_where(difference * difference, (actual >= lit(0.0)) & (predicted >= lit(0.0)))


def rmsle(y_true: IntoExpr, y_pred: IntoExpr) -> Expr:
    """Root mean squared logarithmic error — ``sqrt(msle)``.

    Args:
        y_true: The observed values (non-negative).
        y_pred: The predicted values (non-negative).

    Returns:
        The root mean squared logarithmic error over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [0.0, 3.0], "p": [0.0, 3.0]})
            >>> ds.agg(m=bt.rmsle("y", "p")).to_pydict()
            {'m': [0.0]}
    """
    return msle(y_true, y_pred).sqrt()


def huber_loss(y_true: IntoExpr, y_pred: IntoExpr, *, delta: float = 1.0) -> Expr:
    """Mean Huber loss — squared below `delta`, linear above it.

    The robust regression objective: it keeps the smooth gradient of squared error near
    zero while refusing to let a single outlier dominate, which is exactly the failure
    mode `rmse` has on real data.

    Args:
        y_true: The observed values.
        y_pred: The predicted values.
        delta: The threshold where the loss turns from quadratic to linear.

    Returns:
        The mean Huber loss over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [0.0, 0.0], "p": [0.5, 10.0]})
            >>> ds.agg(m=bt.huber_loss("y", "p")).to_pydict()
            {'m': [4.8125]}
    """
    magnitude = _residual(y_true, y_pred).abs()
    quadratic = magnitude * magnitude * lit(0.5)
    linear = (magnitude - lit(delta * 0.5)) * lit(delta)
    return when(magnitude <= lit(delta)).then(quadratic).otherwise(linear).mean()


def pinball_loss(y_true: IntoExpr, y_pred: IntoExpr, *, quantile: float = 0.5) -> Expr:
    """Mean pinball (quantile) loss — the objective a quantile regressor is scored on.

    Under-prediction is weighted by `quantile` and over-prediction by ``1 - quantile``, so
    a model trained for the 90th percentile is penalised nine times more for being under
    than for being over. At ``quantile=0.5`` it is exactly half the mean absolute error.

    Args:
        y_true: The observed values.
        y_pred: The predicted values.
        quantile: The target quantile in ``[0, 1]``.

    Returns:
        The mean pinball loss over the group.

    Raises:
        PlanError: If `quantile` is not a number in ``[0, 1]``. Past it the weights stop sharing a
            sign and the "loss" goes **negative**: ``90`` scored -890.0 where ``0.9`` scored 1.0.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [10.0], "p": [8.0]})
            >>> ds.agg(m=bt.pinball_loss("y", "p", quantile=0.9)).to_pydict()
            {'m': [1.8]}
    """
    quantile = require_float(quantile, func="pinball_loss", arg="quantile")
    if not 0.0 <= quantile <= 1.0:
        raise PlanError(f"pinball_loss quantile must be in [0, 1], got {quantile}")
    error = _residual(y_true, y_pred)
    under = error * lit(quantile)
    over = error * lit(quantile - 1.0)
    return when(error >= lit(0.0)).then(under).otherwise(over).mean()


def mean_percentage_error(y_true: IntoExpr, y_pred: IntoExpr) -> Expr:
    """The mean signed percentage error — a forecast's *bias*, as a fraction of the actual.

    ``mean((y - yhat) / y)`` over the rows where the actual is nonzero. Unlike `mape`, which takes
    the absolute value and so only measures magnitude, this keeps the sign, so a positive result
    means the forecast systematically *under*-predicts and a negative one that it over-predicts.
    Read it alongside `mape`: a small mean percentage error with a large `mape` is an unbiased but
    noisy forecast, while a large one of either sign is a correctable systematic offset.

    Args:
        y_true: The observed values.
        y_pred: The predicted values.

    Returns:
        The mean signed percentage error as a fraction (multiply by 100 for a percentage).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [10.0, 20.0, 40.0], "p": [11.0, 18.0, 44.0]})
            >>> round(ds.agg(m=bt.mean_percentage_error("y", "p")).to_pydict()["m"][0], 4)
            -0.0333
    """
    actual = _as_column(y_true)
    return _mean_where(_residual(y_true, y_pred) / actual, actual != lit(0.0))


def normalized_rmse(y_true: IntoExpr, y_pred: IntoExpr) -> Expr:
    """The root-mean-squared error scaled by the mean of the actuals — a unitless RMSE.

    ``rmse / mean(y)``: the same error the `rmse` measures, divided by the level of the series so it
    is comparable across targets on different scales. A normalized RMSE of 0.1 means the typical
    error is a tenth of the average value, whatever the units, which is what makes it the right RMSE
    to compare across products, regions, or series that a raw RMSE cannot.

    Args:
        y_true: The observed values.
        y_pred: The predicted values.

    Returns:
        The RMSE divided by the mean of the observed values.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [10.0, 20.0, 30.0, 40.0], "p": [12.0, 18.0, 33.0, 38.0]})
            >>> round(ds.agg(m=bt.normalized_rmse("y", "p")).to_pydict()["m"][0], 4)
            0.0917
    """
    return rmse(y_true, y_pred) / _as_column(y_true).mean()
