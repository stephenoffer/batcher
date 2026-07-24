"""Deviance losses for count and rate models — the right error metric for the wrong-shaped target.

RMSE assumes the error is symmetric and constant-variance. A count (claims, clicks, defects)
and a rate (cost per exposure) are neither: being 10 off on an expected 5 is a disaster and
being 10 off on an expected 1000 is nothing, and squared error cannot tell the two apart. The
deviance of the matching distribution can. These are the losses a Poisson, gamma, or Tweedie
model is actually fitted on, and therefore the honest way to score one.

Each is a mean over a per-row deviance contribution — a single aggregate — and each matches
scikit-learn's ``mean_*_deviance`` where it defines it. The deviance-explained score
(``d2_tweedie_score``) needs the target's own mean as a baseline and so is a Dataset function
in `batcher.ml.metrics`, not an expression.
"""

from __future__ import annotations

from batcher.plan.expr_ir.constructors import lit, when
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, count_if

__all__ = [
    "gamma_deviance",
    "poisson_deviance",
    "tweedie_deviance",
]


def poisson_deviance(y_true: IntoExpr, y_pred: IntoExpr) -> Expr:
    """Mean Poisson deviance — the loss a count model is fitted on.

    ``mean( 2 * (y*ln(y/yhat) - (y - yhat)) )``, with the ``y*ln(y/yhat)`` term taken as 0
    where ``y`` is 0 (its limit). The error metric for a non-negative *count* target, because
    it weights an error by the inverse of the expected value — a miss on a rare event costs
    far more than the same miss on a common one, which is what a count actually behaves like
    and what squared error ignores.

    Predictions must be strictly positive; a non-positive prediction makes the row null rather
    than infinite, so one bad prediction does not destroy the whole metric.

    Args:
        y_true: The observed non-negative counts.
        y_pred: The predicted rates (strictly positive).

    Returns:
        The mean Poisson deviance.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [0.0, 2.0, 3.0], "p": [1.0, 2.0, 3.0]})
            >>> round(ds.agg(m=bt.poisson_deviance("y", "p")).to_pydict()["m"][0], 6)
            0.666667
    """
    return tweedie_deviance(y_true, y_pred, power=1.0)


def gamma_deviance(y_true: IntoExpr, y_pred: IntoExpr) -> Expr:
    """Mean gamma deviance — the loss for a strictly positive, right-skewed target.

    ``mean( 2 * (ln(yhat/y) + y/yhat - 1) )``. The right metric for a positive continuous
    quantity whose variance grows with its mean — a claim size, a purchase amount, a duration.
    Where Poisson deviance suits a count, gamma deviance suits the money the count turns into.

    Both the target and the prediction must be strictly positive; a non-positive value makes
    the row null.

    Args:
        y_true: The observed positive values.
        y_pred: The predicted positive values.

    Returns:
        The mean gamma deviance.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1.0, 2.0, 4.0], "p": [1.0, 2.0, 4.0]})
            >>> ds.agg(m=bt.gamma_deviance("y", "p")).to_pydict()
            {'m': [0.0]}
    """
    return tweedie_deviance(y_true, y_pred, power=2.0)


def tweedie_deviance(y_true: IntoExpr, y_pred: IntoExpr, *, power: float = 1.5) -> Expr:
    """Mean Tweedie deviance — the family that spans Poisson, gamma, and the mix between them.

    The Tweedie ``power`` selects the distribution: 0 is squared error (normal), 1 is Poisson
    (counts), 2 is gamma (positive skew), and ``1 < power < 2`` is the compound Poisson-gamma
    that describes *pure premium* — a mostly-zero target with a heavy positive tail, exactly
    the shape of insurance loss per policy. That intermediate range is why Tweedie is the
    actuarial default and why it earns a parameter rather than three separate functions.

    Args:
        y_true: The observed values.
        y_pred: The predicted values (strictly positive for ``power >= 1``).
        power: The Tweedie power. 0 (normal), 1 (Poisson), ``(1, 2)`` (compound), 2 (gamma).

    Returns:
        The mean Tweedie deviance.

    Raises:
        PlanError: If `power` is in the undefined open interval ``(0, 1)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1.0, 2.0, 3.0], "p": [1.0, 2.0, 3.0]})
            >>> ds.agg(m=bt.tweedie_deviance("y", "p", power=1.5)).to_pydict()
            {'m': [0.0]}
    """
    from batcher._internal.errors import PlanError

    if 0.0 < power < 1.0:
        raise PlanError(
            f"Tweedie deviance is undefined for power in (0, 1), got {power}. Use 0 (normal), "
            "1 (Poisson), a value in (1, 2) (compound), or 2 (gamma)."
        )
    actual = _as_column(y_true)
    predicted = _as_column(y_pred)
    if power == 0.0:
        error = actual - predicted
        return (error * error).mean()
    if power == 1.0:
        # 2 * (y*ln(y/yhat) - (y - yhat)); the y*ln term is 0 at y == 0.
        log_term = (
            when(actual > lit(0.0)).then(actual * (actual / predicted).ln()).otherwise(lit(0.0))
        )
        contribution = lit(2.0) * (log_term - (actual - predicted))
        valid = predicted > lit(0.0)
    elif power == 2.0:
        # 2 * (ln(yhat/y) + y/yhat - 1).
        contribution = lit(2.0) * ((predicted / actual).ln() + actual / predicted - lit(1.0))
        valid = (actual > lit(0.0)) & (predicted > lit(0.0))
    else:
        p = power
        term1 = actual.pow(lit(2.0 - p)) / lit((1.0 - p) * (2.0 - p))
        term2 = actual * predicted.pow(lit(1.0 - p)) / lit(1.0 - p)
        term3 = predicted.pow(lit(2.0 - p)) / lit(2.0 - p)
        contribution = lit(2.0) * (term1 - term2 + term3)
        valid = (actual >= lit(0.0)) & (predicted > lit(0.0))
    # Mean over the valid rows only: an invalid row (a non-positive prediction) is dropped
    # from both the numerator and the denominator rather than turned into a NaN that would
    # poison the whole aggregate. `sum(masked) / count(valid)` is that mean.
    kept = valid & contribution.is_not_null()
    return when(kept).then(contribution).otherwise(lit(0.0)).sum() / count_if(kept)
