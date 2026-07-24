"""Probabilistic classification metrics — the calibration half of the metric surface.

`accuracy` and its relatives score a *decision*: they take a hard prediction and ask
whether it was right. These two score a *belief*: they take the predicted probability and
ask whether it was the right size. A model can be perfect on the first and badly wrong on
the second, and that gap is invisible until a prediction gets multiplied by a dollar
amount or fed into an expected-value calculation.

Both are single-pass aggregates, so they cost the same as any other metric inside `agg()`.
"""

from __future__ import annotations

from typing import Any

from batcher.plan.expr_ir.constructors import lit, when
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column
from batcher.plan.functions.metrics.classification import positive_mask

__all__ = ["brier_score", "log_loss"]

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
