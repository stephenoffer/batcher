"""Agreement and efficiency scores — how well two series match, beyond a plain correlation.

A correlation says two series move together; it says nothing about whether they are *equal*.
A prediction that is always exactly half the truth has a perfect correlation and is useless.
These metrics score agreement — correlation *and* the match in mean and spread — which is what
you actually want when comparing a simulation to an observation, a forecast to an outcome, or
two raters to each other.

Each is a single mergeable aggregate built from means, variances, and a covariance, so it runs
distributed and composes inside `group_by`. They come from the fields that live on this
distinction — `concordance_correlation` from method-comparison studies, `nash_sutcliffe_efficiency`
and `kling_gupta_efficiency` from hydrology, where "the model correlates well" and "the model is
right" are known to be different claims.
"""

from __future__ import annotations

from batcher.plan.expr_ir.constructors import lit
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, corr, covar_pop

__all__ = [
    "concordance_correlation",
    "kling_gupta_efficiency",
    "nash_sutcliffe_efficiency",
]


def _paired(a: IntoExpr, b: IntoExpr) -> tuple[Expr, Expr]:
    """Both columns masked to null wherever either is null, so the pairing is exact."""
    left, right = _as_column(a), _as_column(b)
    return left + right * lit(0.0), right + left * lit(0.0)


def concordance_correlation(y_true: IntoExpr, y_pred: IntoExpr) -> Expr:
    """Lin's concordance correlation coefficient — agreement, not just correlation.

    ``2 * cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y))^2)``. It is a Pearson correlation
    penalised by how far the two series' means and variances differ, so a prediction that
    tracks the truth perfectly but is shifted or scaled scores below 1 where a plain
    correlation would still say 1. The standard measure of agreement between two methods
    measuring the same thing. 1 is perfect concordance, 0 none, negative disagreement.

    Args:
        y_true: The observed values.
        y_pred: The predicted values.

    Returns:
        The concordance correlation in ``[-1, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1.0, 2.0, 3.0], "p": [1.0, 2.0, 3.0]})
            >>> ds.agg(m=bt.concordance_correlation("y", "p")).to_pydict()
            {'m': [1.0]}
    """
    observed, predicted = _paired(y_true, y_pred)
    covariance = covar_pop(observed, predicted)
    # Population variances (covar_pop(z, z)), so numerator and denominator share the same n
    # denominator — mixing a population covariance with a sample (n-1) variance is exactly the
    # bug that gives 2/3 instead of 1 for two identical series.
    var_observed = covar_pop(observed, observed)
    var_predicted = covar_pop(predicted, predicted)
    mean_diff = observed.mean() - predicted.mean()
    denominator = var_observed + var_predicted + mean_diff * mean_diff
    return lit(2.0) * covariance / denominator


def nash_sutcliffe_efficiency(y_true: IntoExpr, y_pred: IntoExpr) -> Expr:
    """Nash-Sutcliffe efficiency — ``1 - sum((y - yhat)^2) / sum((y - mean(y))^2)``.

    The hydrologist's R²: the residual sum of squares against the variance of the observations.
    1 is a perfect model, 0 is no better than predicting the observed mean, and negative means
    the mean would have been a better predictor. Identical in form to the coefficient of
    determination but named for the observed-versus-simulated framing it is used in.

    Args:
        y_true: The observed values.
        y_pred: The simulated/predicted values.

    Returns:
        The Nash-Sutcliffe efficiency; at most 1.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1.0, 2.0, 3.0], "p": [1.0, 2.0, 3.0]})
            >>> ds.agg(m=bt.nash_sutcliffe_efficiency("y", "p")).to_pydict()
            {'m': [1.0]}
    """
    observed, predicted = _paired(y_true, y_pred)
    error = observed - predicted
    total = observed.var() * (observed.count() - lit(1))
    return lit(1.0) - (error * error).sum() / total


def kling_gupta_efficiency(y_true: IntoExpr, y_pred: IntoExpr) -> Expr:
    """Kling-Gupta efficiency — agreement decomposed into correlation, bias, and variability.

    ``1 - sqrt((r - 1)^2 + (sigma_pred/sigma_obs - 1)^2 + (mean_pred/mean_obs - 1)^2)`` over the
    correlation ``r``, the ratio of standard deviations, and the ratio of means. Unlike
    Nash-Sutcliffe it weights the three ways a model can be wrong — mistiming, over/under
    dispersion, and bias — equally, which is why it has become the modern default for scoring a
    simulation against an observation. 1 is perfect; the ``-0.41`` of predicting the mean is the
    reference point below which the model adds nothing.

    Args:
        y_true: The observed values.
        y_pred: The simulated/predicted values.

    Returns:
        The Kling-Gupta efficiency; at most 1.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1.0, 2.0, 3.0, 4.0], "p": [1.0, 2.0, 3.0, 4.0]})
            >>> round(ds.agg(m=bt.kling_gupta_efficiency("y", "p")).to_pydict()["m"][0], 6)
            1.0
    """
    observed, predicted = _paired(y_true, y_pred)
    r = corr(observed, predicted)
    variability = predicted.std() / observed.std()
    bias = predicted.mean() / observed.mean()
    euclidean = (
        (r - lit(1.0)) * (r - lit(1.0))
        + (variability - lit(1.0)) * (variability - lit(1.0))
        + (bias - lit(1.0)) * (bias - lit(1.0))
    ).sqrt()
    return lit(1.0) - euclidean
