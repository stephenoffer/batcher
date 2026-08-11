"""Time-series diagnostics — autocorrelation and the tests built on it.

A time series is the case a Pearson correlation cannot see: the signal is in how a column
relates to its own past. The autocorrelation function is that relationship at each lag, and it
is the input to the two standard tests of whether a series (or a model's residuals) still
carries structure the model has not captured. Each diagnostic here orders the column by a time
key, lags it with a window, and reduces the overlap to one aggregate.

Unlike the mergeable statistics elsewhere in `batcher.ml.stats`, an autocorrelation is an
inherently ordered, global computation — it needs the whole series in time order — so these
run over a single ordered window rather than as a partitionable aggregate. The formulas are
the Box-Jenkins definitions, checked against independent numpy references.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.ml.stats._special import chi2_sf
from batcher.ml.stats.hypothesis import TestResult
from batcher.plan.expr_ir.constructors import col, lit
from batcher.plan.expr_ir.nodes import lag
from batcher.plan.functions.aggregate import mean
from batcher.plan.functions.aggregate import sum as sum_

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = [
    "autocorrelation",
    "autocorrelations",
    "durbin_watson",
    "ljung_box",
    "mean_absolute_scaled_error",
    "partial_autocorrelation",
    "partial_autocorrelations",
]


def _mean_and_ss(ds: Dataset, column: str) -> tuple[float, float]:
    """The series mean and its total sum of squared deviations, in one pass each.

    Two passes rather than the one that ``sum(x^2) - n * mean^2`` would allow: that form
    subtracts two large nearly-equal numbers and loses every significant digit on a series
    with a large offset, which is the same catastrophic cancellation the rolling-variance
    expressions in `plan.expr_ir` carry their own correction for.
    """
    m = float(ds.agg(m=mean(col(column))).collect().column("m")[0].as_py())
    centered = col(column) - lit(m)
    ss = float(ds.agg(s=sum_(centered * centered)).collect().column("s")[0].as_py())
    return m, ss


def _acf(ds: Dataset, column: str, lags: Sequence[int], *, order_by: str) -> dict[int, float]:
    """Every requested lag's autocorrelation, in three executions however many lags there are.

    All the lag columns are built in one `with_columns`, so they share a single `Window` node
    — one partition, one ordering, one sort — and all the numerators are summed in one `agg`.
    Computing them a lag at a time instead costs three executions *and a full global sort* per
    lag, which is what made a 40-lag ACF (an ordinary seasonal diagnostic) 120 executions and
    40 sorts of the whole series.

    No `filter(lag.is_not_null())` is needed before the sum, and its absence is not a
    shortcut: the first `k` rows have a null lag, so their product is null, and `sum` skips
    nulls. Filtering first would remove exactly the rows that contribute nothing, which is why
    the per-lag version could afford it and why the fused one does not need it — the lags have
    different overlaps and could not share a filter anyway.
    """
    mean_value, ss_total = _mean_and_ss(ds, column)
    if ss_total == 0:
        return dict.fromkeys(lags, float("nan"))
    names = {k: f"__bt_lag{k}" for k in lags}
    lagged = ds.with_columns(
        **{name: lag(col(column), k).over(order_by=[order_by]) for k, name in names.items()}
    )
    centered = col(column) - lit(mean_value)
    row = lagged.agg(
        **{
            f"__bt_r{k}": sum_(centered * (col(name) - lit(mean_value)))
            for k, name in names.items()
        }
    ).collect()
    out: dict[int, float] = {}
    for k in names:
        # A lag at least as long as the series overlaps nothing, so the sum is null rather
        # than a number. That is `nan` — an undefined correlation — not a zero one.
        value = row.column(f"__bt_r{k}")[0].as_py()
        out[k] = float("nan") if value is None else float(value) / ss_total
    return out


def autocorrelation(ds: Dataset, column: str, k: int, *, order_by: str) -> float:
    """The autocorrelation of a series with itself at lag `k` (Box-Jenkins).

    How strongly a value resembles the value `k` steps earlier, in ``[-1, 1]``. A large lag-1
    autocorrelation means the series is smooth or trending; a large value at a seasonal lag
    means a repeating cycle. This is the r_k of the standard autocorrelation function, using
    the full-series mean and total sum of squares, so it agrees with the Box-Jenkins definition
    every time-series text uses.

    Args:
        ds: The dataset holding the series.
        column: The numeric series column.
        k: The lag, a positive number of steps.
        order_by: The time-ordering column that puts the series in sequence.

    Returns:
        The lag-`k` autocorrelation in ``[-1, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.timeseries import autocorrelation
            >>> ds = bt.from_pydict({"t": [0, 1, 2, 3, 4, 5], "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
            >>> round(autocorrelation(ds, "x", 1, order_by="t"), 4)
            0.5
    """
    if k <= 0:
        from batcher._internal.errors import PlanError

        raise PlanError(f"autocorrelation lag must be positive, got {k}.")
    return _acf(ds, column, [k], order_by=order_by)[k]


def autocorrelations(ds: Dataset, column: str, lags: int, *, order_by: str) -> dict[int, float]:
    """The autocorrelation at every lag from 1 to `lags` — the sample ACF.

    The whole autocorrelation function in one call, which is what you plot to read a series'
    memory: a slow decay says trending, a spike at lag 12 says monthly seasonality, a cut-off
    after lag `q` suggests a moving-average order.

    Args:
        ds: The dataset holding the series.
        column: The numeric series column.
        lags: The maximum lag to compute; every lag from 1 to this is returned.
        order_by: The time-ordering column.

    Returns:
        A ``{lag: autocorrelation}`` dict for lags ``1..lags``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.timeseries import autocorrelations
            >>> ds = bt.from_pydict({"t": [0, 1, 2, 3, 4, 5], "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
            >>> {k: round(v, 3) for k, v in autocorrelations(ds, "x", 2, order_by="t").items()}
            {1: 0.5, 2: 0.057}
    """
    if lags <= 0:
        from batcher._internal.errors import PlanError

        raise PlanError(f"autocorrelations needs a positive lag count, got {lags}.")
    return _acf(ds, column, range(1, lags + 1), order_by=order_by)


def ljung_box(ds: Dataset, column: str, lags: int, *, order_by: str) -> TestResult:
    """Test whether a series has any autocorrelation up to lag `lags` (Ljung-Box).

    The portmanteau test: instead of eyeballing the ACF lag by lag, it pools the first `lags`
    autocorrelations into one statistic, ``Q = n(n+2) * sum_k r_k^2 / (n - k)``, which is
    chi-squared with `lags` degrees of freedom under the null of no autocorrelation. A small
    p-value says the series (or, applied to residuals, the model's residuals) still has
    structure. The standard white-noise check on a forecasting model's residuals.

    Args:
        ds: The dataset holding the series.
        column: The numeric series column.
        lags: How many lags to pool into the statistic.
        order_by: The time-ordering column.

    Returns:
        A `TestResult` with the Q statistic, `lags` degrees of freedom, and the upper-tail
        p-value.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.timeseries import ljung_box
            >>> ds = bt.from_pydict({"t": list(range(20)), "x": [float(i) for i in range(20)]})
            >>> ljung_box(ds, "x", 4, order_by="t").pvalue < 0.05
            True
    """
    n = ds.count()
    acf = autocorrelations(ds, column, lags, order_by=order_by)
    q = n * (n + 2) * sum(r * r / (n - k) for k, r in acf.items())
    return TestResult(statistic=q, pvalue=chi2_sf(q, float(lags)), df=float(lags))


def durbin_watson(ds: Dataset, column: str, *, order_by: str) -> float:
    """The Durbin-Watson statistic for a residual column's serial correlation, in ``[0, 4]``.

    The regression diagnostic for autocorrelated residuals: ``DW = sum (e_t - e_{t-1})^2 /
    sum e_t^2``. It sits near 2 when residuals are independent, near 0 under strong positive
    autocorrelation (the model is missing a trend or a lag), and near 4 under negative
    autocorrelation. Apply it to a model's residual column to check the independence assumption
    ordinary least squares relies on.

    Args:
        ds: The dataset holding the residual series.
        column: The numeric residual column.
        order_by: The time-ordering column.

    Returns:
        The Durbin-Watson statistic in ``[0, 4]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.timeseries import durbin_watson
            >>> ds = bt.from_pydict({"t": [0, 1, 2, 3], "e": [1.0, -1.0, 1.0, -1.0]})
            >>> round(durbin_watson(ds, "e", order_by="t"), 3)
            3.0
    """
    lagged = ds.with_columns(__bt_prev=lag(col(column), 1).over(order_by=[order_by]))
    diff = col(column) - col("__bt_prev")
    numerator = (
        lagged.filter(col("__bt_prev").is_not_null())
        .agg(s=sum_(diff * diff))
        .collect()
        .column("s")[0]
        .as_py()
    )
    denominator = ds.agg(s=sum_(col(column) * col(column))).collect().column("s")[0].as_py()
    if not denominator:
        return float("nan")
    return float(numerator) / float(denominator)


def mean_absolute_scaled_error(
    ds: Dataset, y_true: str, y_pred: str, *, order_by: str, seasonality: int = 1
) -> float:
    """The forecast error scaled by the naive seasonal forecast's error (MASE).

    The scale-free accuracy metric for forecasting: the model's mean absolute error divided by
    the in-sample mean absolute error of the seasonal naive forecast (predict the value
    `seasonality` steps back). Below 1 means the model beats naive; above 1 means it does not.
    Because it is a ratio it is comparable across series on wildly different scales, which is
    exactly what a percentage error (undefined at zero) and a raw MAE (unit-bound) cannot do.

    Args:
        ds: The dataset holding the series and the forecast.
        y_true: The observed values.
        y_pred: The model's forecast.
        order_by: The time-ordering column.
        seasonality: The naive forecast's lag — 1 for a random-walk baseline, or the period
            (12 for monthly, 7 for daily) for a seasonal one.

    Returns:
        The mean absolute scaled error; below 1 beats the naive forecast.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.timeseries import mean_absolute_scaled_error
            >>> ds = bt.from_pydict(
            ...     {"t": [0, 1, 2, 3], "y": [1.0, 2.0, 3.0, 4.0], "p": [1.0, 2.0, 3.0, 4.0]}
            ... )
            >>> mean_absolute_scaled_error(ds, "y", "p", order_by="t")
            0.0
    """
    from batcher.plan.functions.aggregate import mean as mean_

    numerator = float(
        ds.agg(m=mean_((col(y_true) - col(y_pred)).abs())).collect().column("m")[0].as_py()
    )
    lagged = ds.with_columns(__bt_naive=lag(col(y_true), seasonality).over(order_by=[order_by]))
    denominator = (
        lagged.filter(col("__bt_naive").is_not_null())
        .agg(m=mean_((col(y_true) - col("__bt_naive")).abs()))
        .collect()
        .column("m")[0]
        .as_py()
    )
    if not denominator:
        return float("nan")
    return numerator / float(denominator)


def partial_autocorrelations(
    ds: Dataset, column: str, lags: int, *, order_by: str
) -> dict[int, float]:
    """The partial autocorrelation at every lag from 1 to `lags` — the sample PACF.

    Where the autocorrelation at lag `k` includes everything the intervening lags already explain,
    the *partial* autocorrelation strips that out: it is the correlation between a value and its
    lag-`k` self after removing the linear effect of lags 1 through ``k - 1``. That is what makes
    the PACF the tool for choosing an autoregressive order — for a true AR(p) process the PACF cuts
    off sharply after lag `p`, where the ACF only decays. Computed by the Durbin-Levinson recursion
    over the sample autocorrelations (the Yule-Walker estimate).

    Args:
        ds: The dataset holding the series.
        column: The numeric series column.
        lags: The maximum lag to compute; every lag from 1 to this is returned.
        order_by: The time-ordering column.

    Returns:
        A ``{lag: partial_autocorrelation}`` dict for lags ``1..lags``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.timeseries import partial_autocorrelations
            >>> ds = bt.from_pydict({"t": list(range(10)), "x": [float(i) for i in range(10)]})
            >>> pacf = partial_autocorrelations(ds, "x", 3, order_by="t")
            >>> round(pacf[1], 4)
            0.7
    """
    acf = autocorrelations(ds, column, lags, order_by=order_by)
    r = [1.0] + [acf[k] for k in range(1, lags + 1)]
    phi = [[0.0] * (lags + 1) for _ in range(lags + 1)]
    result: dict[int, float] = {}
    phi[1][1] = r[1]
    result[1] = r[1]
    for k in range(2, lags + 1):
        numerator = r[k] - sum(phi[k - 1][j] * r[k - j] for j in range(1, k))
        denominator = 1.0 - sum(phi[k - 1][j] * r[j] for j in range(1, k))
        phi[k][k] = numerator / denominator if denominator != 0 else float("nan")
        for j in range(1, k):
            phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j]
        result[k] = phi[k][k]
    return result


def partial_autocorrelation(ds: Dataset, column: str, k: int, *, order_by: str) -> float:
    """The partial autocorrelation of a series at lag `k` (Yule-Walker).

    The lag-`k` value of the partial autocorrelation function: the correlation between a value and
    its lag-`k` self once the intervening lags are accounted for. See `partial_autocorrelations`
    for the whole function and why it identifies an autoregressive order.

    Args:
        ds: The dataset holding the series.
        column: The numeric series column.
        k: The lag, a positive number of steps.
        order_by: The time-ordering column.

    Returns:
        The lag-`k` partial autocorrelation.

    Raises:
        PlanError: If `k` is not positive.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.timeseries import partial_autocorrelation
            >>> ds = bt.from_pydict({"t": list(range(10)), "x": [float(i) for i in range(10)]})
            >>> round(partial_autocorrelation(ds, "x", 1, order_by="t"), 4)
            0.7
    """
    if k <= 0:
        from batcher._internal.errors import PlanError

        raise PlanError(f"partial_autocorrelation lag must be positive, got {k}.")
    return partial_autocorrelations(ds, column, k, order_by=order_by)[k]
