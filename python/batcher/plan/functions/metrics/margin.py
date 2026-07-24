"""Margin losses — scoring a classifier by how far its decision function is on the right side.

A probability loss (`log_loss`, `brier_score`) scores a calibrated probability; a margin loss
scores a raw decision function — the signed distance to the boundary a support-vector machine or
a linear classifier produces. The loss is zero once a point is correctly classified with room to
spare (a margin of at least 1) and grows with how far a point is on the wrong side, which is the
objective these models actually optimize.

Each is a single mergeable aggregate over the label and the score, so it composes with
`group_by` and runs distributed exactly as the other metrics do. The label is coded to ``+1`` /
``-1`` internally, so pass an ordinary 0/1 (or boolean) label and set `positive` if the positive
class is not ``1``.
"""

from __future__ import annotations

from typing import Any

from batcher.plan.expr_ir.constructors import lit, when
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column
from batcher.plan.functions.metrics.classification import positive_mask

__all__ = ["hinge_loss", "squared_hinge_loss"]


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
