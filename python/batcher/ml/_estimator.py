"""The scaffolding every fitted estimator in `batcher.ml` shares.

The estimators here (linear, discriminant, naive-Bayes, mixture, cluster, GLM) all follow one
shape: `fit` runs a handful of grouped aggregates and stores per-class parameters, then `predict`
lowers those parameters to an `Expr` the engine evaluates in Rust. Two pieces of that shape were
being written out by hand in every estimator — the "was this fitted?" guard and the argmax over a
set of per-class score expressions — so they live here once instead of once per module.

Neither helper touches a row: `argmax_prediction` builds a nested `when` chain that the engine
evaluates column-wise, which is why an estimator can score a billion rows without the control
plane seeing one of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.constructors import col, lit, when

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from batcher.plan.expr_ir.core import Expr

T = TypeVar("T")

__all__ = ["argmax_prediction", "linear_score", "require_fitted"]


def require_fitted(estimator: object, state: T | None, method: str = "predict") -> T:
    """Return `state`, or raise if the estimator has not been fitted yet.

    Every estimator stores its learned parameters in a trailing-underscore attribute that is empty
    until `fit` runs. Calling `predict` first is the single most common mistake against this API,
    so it gets one typed error with the class name and the method in it rather than an
    `IndexError` from deep inside expression construction.

    Args:
        estimator: The estimator instance, used for the class name in the message.
        state: The learned-parameter attribute to check for emptiness.
        method: The method being guarded, named in the error message.

    Returns:
        The `state` value, guaranteed non-empty.

    Raises:
        PlanError: If `state` is `None` or empty.
    """
    if not state:
        name = type(estimator).__name__
        raise PlanError(f"{name} must be fitted before {method}().")
    return state


def linear_score(features: Sequence[str], weights: Sequence[float], intercept: float) -> Expr:
    """Build the linear predictor ``intercept + weights . features`` as one expression.

    The shared spine of every linear model here — least squares, ridge, the elastic net, the GLM
    link's inner term, and each class's discriminant score. The weights are folded in as literals
    at plan time, so what reaches the engine is a single arithmetic tree over the feature columns
    rather than a parameter lookup per row.

    Args:
        features: The feature column names, in the order the weights were fitted.
        weights: One coefficient per feature.
        intercept: The constant term.

    Returns:
        An expression evaluating to the linear score for each row.

    Raises:
        ValueError: If `features` and `weights` differ in length.
    """
    expression = lit(intercept)
    for weight, name in zip(weights, features, strict=True):
        expression = expression + lit(weight) * col(name)
    return expression


def argmax_prediction(labels: Sequence[T], score_of: Callable[[T], Expr]) -> Expr:
    """Build the expression selecting whichever label scores highest on each row.

    Folds the labels into a nested `when(score > best).then(label).otherwise(...)` chain, carrying
    the running maximum alongside the running argmax. That keeps the whole decision in one
    expression tree, so a classifier's `predict` is a single projection the JIT can compile rather
    than one pass per class.

    Args:
        labels: The fitted class labels, in a stable order. Ties resolve to the earliest label.
        score_of: Maps a label to its per-row score expression; higher wins.

    Returns:
        An expression evaluating to the highest-scoring label for each row.
    """
    prediction = lit(labels[0])
    best = score_of(labels[0])
    for label in labels[1:]:
        score = score_of(label)
        closer = score > best
        prediction = when(closer).then(lit(label)).otherwise(prediction)
        best = when(closer).then(score).otherwise(best)
    return prediction
