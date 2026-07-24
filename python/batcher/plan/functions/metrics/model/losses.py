"""Loss functions and proper scoring rules — the numbers a model is trained to minimize.

Where `classification` and `errors` describe how a fitted model behaves, these are the objectives
themselves: the probabilistic scores (Brier, log loss), the margin losses behind an SVM (hinge,
squared hinge), and the GLM deviances (Poisson, gamma, Tweedie) that generalize squared error to a
non-Gaussian target. They sit together because they answer one question — what did this prediction
cost — which is what makes a training curve comparable against a holdout number.

Each is an aggregate expression, so a loss can be computed per segment, per day, or per cohort in
the same scan that computes it overall.
"""

from __future__ import annotations

from typing import Any

from batcher.plan.expr_ir.constructors import lit, when
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, count_if
from batcher.plan.functions.metrics.model.classification import positive_mask

__all__ = [
    "brier_score",
    "gamma_deviance",
    "hinge_loss",
    "log_loss",
    "poisson_deviance",
    "squared_hinge_loss",
    "tweedie_deviance",
]


# Log loss is unbounded as a probability approaches 0 or 1, so a single confidently-wrong
# row would return infinity for the whole dataset. scikit-learn clips at 1e-15 for the same
# reason; matching that keeps the two comparable.
_LOG_LOSS_EPS = 1e-15


def log_loss(y_true: IntoExpr, y_score: IntoExpr, *, positive: Any = 1) -> Expr:
    """Binary cross-entropy — ``-mean(y*ln(p) + (1-y)*ln(1-p))``.

    The metric that scores *calibration*, not just ranking: a model that is right but
    over-confident is punished here and nowhere else. Lower is better; predicting the base
    rate for every row gives the entropy of the labels.

    Scores are clipped to ``[1e-15, 1 - 1e-15]`` before the logarithm, matching
    scikit-learn, so one confidently-wrong row cannot make the whole dataset's loss
    infinite.

    Args:
        y_true: The observed labels.
        y_score: The predicted probability of the positive class, in ``[0, 1]``.
        positive: The value that counts as the positive class.

    Returns:
        The mean binary cross-entropy (nats).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 0], "p": [0.5, 0.5]})
            >>> round(ds.agg(m=bt.log_loss("y", "p")).to_pydict()["m"][0], 6)
            0.693147
    """
    score = _as_column(y_score).clip(lit(_LOG_LOSS_EPS), lit(1.0 - _LOG_LOSS_EPS))
    positive_term = score.ln()
    negative_term = (lit(1.0) - score).ln()
    row_loss = when(positive_mask(y_true, positive)).then(positive_term).otherwise(negative_term)
    return -row_loss.mean()


def brier_score(y_true: IntoExpr, y_score: IntoExpr, *, positive: Any = 1) -> Expr:
    """Mean squared error of the predicted probability — ``mean((p - y)^2)``.

    The bounded, gentler alternative to `log_loss`: it also measures calibration, but a
    confidently-wrong row costs at most 1 rather than diverging. 0 is perfect.

    Args:
        y_true: The observed labels.
        y_score: The predicted probability of the positive class, in ``[0, 1]``.
        positive: The value that counts as the positive class.

    Returns:
        The Brier score in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 0], "p": [1.0, 0.0]})
            >>> ds.agg(m=bt.brier_score("y", "p")).to_pydict()
            {'m': [0.0]}
    """
    actual = when(positive_mask(y_true, positive)).then(lit(1.0)).otherwise(lit(0.0))
    error = _as_column(y_score) - actual
    return (error * error).mean()


def _margin(y_true: IntoExpr, y_score: IntoExpr, positive: Any) -> Expr:
    """The hinge margin ``max(0, 1 - sign(y) * score)`` per row, with ``y`` coded +1/-1."""
    signed = when(positive_mask(_as_column(y_true), positive)).then(lit(1.0)).otherwise(lit(-1.0))
    raw = lit(1.0) - signed * _as_column(y_score)
    return when(raw > lit(0.0)).then(raw).otherwise(lit(0.0))


def hinge_loss(y_true: IntoExpr, y_score: IntoExpr, *, positive: Any = 1) -> Expr:
    """The average hinge loss of a decision function — the support-vector-machine objective.

    ``mean(max(0, 1 - sign(y) * score))``: zero for a point classified correctly with a margin of
    at least 1, and growing linearly as a point moves toward and past the boundary. It is the loss
    a linear SVM minimizes, and the right score for a model that outputs a raw decision value
    rather than a probability. Matches scikit-learn's ``hinge_loss``.

    Args:
        y_true: The true labels (0/1 or boolean).
        y_score: The decision function — the signed distance to the boundary.
        positive: The label value that counts as the positive class.

    Returns:
        The mean hinge loss over the group, at least 0.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 1, 0, 0], "s": [2.0, 0.5, -2.0, -0.5]})
            >>> ds.agg(h=bt.hinge_loss("y", "s")).to_pydict()
            {'h': [0.25]}
    """
    return _margin(y_true, y_score, positive).mean()


def squared_hinge_loss(y_true: IntoExpr, y_score: IntoExpr, *, positive: Any = 1) -> Expr:
    """The average squared hinge loss — the smooth, margin-violation-punishing variant.

    ``mean(max(0, 1 - sign(y) * score)^2)``: the hinge loss squared, which is differentiable
    everywhere and penalizes a point deep on the wrong side far more steeply than the plain hinge.
    It is the objective of the L2-loss (squared-hinge) SVM, and the score to prefer when a few
    badly-misclassified points should dominate the total.

    Args:
        y_true: The true labels (0/1 or boolean).
        y_score: The decision function — the signed distance to the boundary.
        positive: The label value that counts as the positive class.

    Returns:
        The mean squared hinge loss over the group, at least 0.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 1, 0, 0], "s": [2.0, 0.0, -2.0, 0.0]})
            >>> ds.agg(h=bt.squared_hinge_loss("y", "s")).to_pydict()
            {'h': [0.5]}
    """
    margin = _margin(y_true, y_score, positive)
    return (margin * margin).mean()


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
