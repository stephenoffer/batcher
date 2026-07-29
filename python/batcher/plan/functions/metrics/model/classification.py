"""Classification metrics as mergeable aggregate expressions.

Two families live here, distinguished by what the prediction column holds:

**Threshold metrics** take a *hard* prediction (a label) and reduce to the four confusion
counts — ``tp``, ``fp``, ``fn``, ``tn`` — each a `count_if` over one predicate. Everything
from accuracy to Matthews' correlation is arithmetic over those four, which is why they
all cost exactly one aggregate pass no matter how many you ask for at once.

The probabilistic metrics that score a *calibration* rather than a decision — log loss and
the Brier score — live in the sibling `probabilistic` module, because they take a score
column rather than a label column and share none of the confusion arithmetic.

The metrics that genuinely need an ordering over the whole dataset — ROC AUC, PR AUC, the
KS statistic — are not here: they are `batcher.ml.metrics` Dataset functions built on a
window rank, because a rank is not an aggregate.

The positive class defaults to ``1``/``True`` and is configurable with `positive`, so a
string label column (``"churned"``) works without being re-encoded first.
"""

from __future__ import annotations

from typing import Any

from batcher.plan.expr_ir.constructors import lit, when
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, count_if

__all__ = [
    "accuracy",
    "balanced_accuracy",
    "cohen_kappa",
    "f1_score",
    "false_negative_rate",
    "false_negatives",
    "false_positive_rate",
    "false_positives",
    "fbeta_score",
    "matthews_corrcoef",
    "negative_predictive_value",
    "precision",
    "prevalence",
    "recall",
    "specificity",
    "true_negatives",
    "true_positives",
]


def positive_mask(column: IntoExpr, positive: Any) -> Expr:
    """Whether `column` equals the positive class.

    Shared with the rank-based metrics in `batcher.ml.metrics`, which need the same
    definition of "positive" and must not restate it — the two disagreeing would give a
    confusion matrix and an AUC computed against different labels.

    Args:
        column: The label column or expression.
        positive: The value that counts as the positive class.

    Returns:
        A boolean `Expr`, null where the label is null.
    """
    return _as_column(column) == lit(positive)


def true_positives(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """Rows predicted positive that are positive — the ``tp`` confusion cell.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        The true-positive count over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 1, 0], "p": [1, 0, 0]})
            >>> ds.agg(n=bt.true_positives("y", "p")).to_pydict()
            {'n': [1]}
    """
    return count_if(positive_mask(y_true, positive) & positive_mask(y_pred, positive))


def false_positives(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """Rows predicted positive that are negative — the ``fp`` confusion cell.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        The false-positive count over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 0, 0], "p": [1, 1, 0]})
            >>> ds.agg(n=bt.false_positives("y", "p")).to_pydict()
            {'n': [1]}
    """
    return count_if(~positive_mask(y_true, positive) & positive_mask(y_pred, positive))


def false_negatives(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """Rows predicted negative that are positive — the ``fn`` confusion cell.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        The false-negative count over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 1, 0], "p": [1, 0, 0]})
            >>> ds.agg(n=bt.false_negatives("y", "p")).to_pydict()
            {'n': [1]}
    """
    return count_if(positive_mask(y_true, positive) & ~positive_mask(y_pred, positive))


def true_negatives(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """Rows predicted negative that are negative — the ``tn`` confusion cell.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        The true-negative count over the group.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 0, 0], "p": [1, 1, 0]})
            >>> ds.agg(n=bt.true_negatives("y", "p")).to_pydict()
            {'n': [1]}
    """
    return count_if(~positive_mask(y_true, positive) & ~positive_mask(y_pred, positive))


def accuracy(y_true: IntoExpr, y_pred: IntoExpr) -> Expr:
    """Fraction of rows where the prediction equals the label.

    Multi-class safe (it compares values, not a positive class), and the one metric that
    is actively misleading on imbalanced data: at 1% positives, predicting "negative"
    always scores 0.99. Report it beside `balanced_accuracy` or `recall`.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.

    Returns:
        The accuracy in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 1, 0, 0], "p": [1, 0, 0, 0]})
            >>> ds.agg(m=bt.accuracy("y", "p")).to_pydict()
            {'m': [0.75]}
    """
    matched = count_if(_as_column(y_true) == _as_column(y_pred))
    return matched / count_if(_as_column(y_true).is_not_null() & _as_column(y_pred).is_not_null())


def precision(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """Of the rows predicted positive, the fraction that are — ``tp / (tp + fp)``.

    The cost-of-a-false-alarm metric: what fraction of the alerts you act on are real.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        The precision in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 0, 1], "p": [1, 1, 0]})
            >>> ds.agg(m=bt.precision("y", "p")).to_pydict()
            {'m': [0.5]}
    """
    tp = true_positives(y_true, y_pred, positive=positive)
    return tp / (tp + false_positives(y_true, y_pred, positive=positive))


def recall(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """Of the truly positive rows, the fraction found — ``tp / (tp + fn)``.

    Sensitivity, the true-positive rate. The cost-of-a-miss metric.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        The recall in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 0, 1], "p": [1, 1, 0]})
            >>> ds.agg(m=bt.recall("y", "p")).to_pydict()
            {'m': [0.5]}
    """
    tp = true_positives(y_true, y_pred, positive=positive)
    return tp / (tp + false_negatives(y_true, y_pred, positive=positive))


def specificity(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """Of the truly negative rows, the fraction called negative — ``tn / (tn + fp)``.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        The specificity in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [0, 0, 1], "p": [0, 1, 1]})
            >>> ds.agg(m=bt.specificity("y", "p")).to_pydict()
            {'m': [0.5]}
    """
    tn = true_negatives(y_true, y_pred, positive=positive)
    return tn / (tn + false_positives(y_true, y_pred, positive=positive))


def false_positive_rate(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """Of the truly negative rows, the fraction called positive — ``1 - specificity``.

    The x-axis of the ROC curve.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        The false-positive rate in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [0, 0, 1], "p": [0, 1, 1]})
            >>> ds.agg(m=bt.false_positive_rate("y", "p")).to_pydict()
            {'m': [0.5]}
    """
    return lit(1.0) - specificity(y_true, y_pred, positive=positive)


def negative_predictive_value(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """Of the rows predicted negative, the fraction that are — ``tn / (tn + fn)``.

    Precision's mirror, and the number that matters when a negative prediction is what
    triggers the action (a "safe to skip" classifier).

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        The negative predictive value in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [0, 1, 0], "p": [0, 0, 0]})
            >>> ds.agg(m=bt.negative_predictive_value("y", "p")).to_pydict()
            {'m': [0.6666666666666666]}
    """
    tn = true_negatives(y_true, y_pred, positive=positive)
    return tn / (tn + false_negatives(y_true, y_pred, positive=positive))


def f1_score(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """Harmonic mean of precision and recall — ``2tp / (2tp + fp + fn)``.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        The F1 score in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 0, 1], "p": [1, 1, 0]})
            >>> ds.agg(m=bt.f1_score("y", "p")).to_pydict()
            {'m': [0.5]}
    """
    return fbeta_score(y_true, y_pred, beta=1.0, positive=positive)


def fbeta_score(
    y_true: IntoExpr, y_pred: IntoExpr, *, beta: float = 1.0, positive: Any = 1
) -> Expr:
    """Weighted harmonic mean of precision and recall; `beta` > 1 favours recall.

    The dial `f1_score` fixes at 1. Use ``beta=2`` when a miss costs more than a false
    alarm (fraud, disease screening) and ``beta=0.5`` when the reverse holds.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        beta: How much more recall weighs than precision.
        positive: The value that counts as the positive class.

    Returns:
        The F-beta score in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 0, 1], "p": [1, 1, 0]})
            >>> ds.agg(m=bt.fbeta_score("y", "p", beta=2.0)).to_pydict()
            {'m': [0.5]}
    """
    tp = true_positives(y_true, y_pred, positive=positive)
    fp = false_positives(y_true, y_pred, positive=positive)
    fn = false_negatives(y_true, y_pred, positive=positive)
    squared = beta * beta
    return (lit(1.0 + squared) * tp) / (lit(1.0 + squared) * tp + lit(squared) * fn + fp)


def balanced_accuracy(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """The mean of recall and specificity — accuracy with the class imbalance removed.

    A majority-class predictor scores 0.5 here rather than the 0.99 plain `accuracy` gives
    it, which is why this is the honest headline number on imbalanced data.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        The balanced accuracy in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 0, 0, 0], "p": [0, 0, 0, 0]})
            >>> ds.agg(m=bt.balanced_accuracy("y", "p")).to_pydict()
            {'m': [0.5]}
    """
    sensitivity = recall(y_true, y_pred, positive=positive)
    return (sensitivity + specificity(y_true, y_pred, positive=positive)) / lit(2.0)


def matthews_corrcoef(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """Matthews correlation coefficient, in ``[-1, 1]`` — the balanced single number.

    The only common binary metric that uses all four confusion cells symmetrically, so it
    is high only when the model does well on both classes. 0 is chance, negative is
    anti-correlated.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        The Matthews correlation coefficient in ``[-1, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 1, 0, 0], "p": [1, 1, 0, 0]})
            >>> ds.agg(m=bt.matthews_corrcoef("y", "p")).to_pydict()
            {'m': [1.0]}
    """
    tp = true_positives(y_true, y_pred, positive=positive)
    fp = false_positives(y_true, y_pred, positive=positive)
    fn = false_negatives(y_true, y_pred, positive=positive)
    tn = true_negatives(y_true, y_pred, positive=positive)
    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).sqrt()
    return when(denominator == lit(0.0)).then(lit(0.0)).otherwise(numerator / denominator)


def cohen_kappa(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """Cohen's kappa — agreement corrected for the agreement chance alone would give.

    ``(observed - expected) / (1 - expected)``. 1 is perfect, 0 is chance-level, negative
    is worse than chance. The metric to reach for when comparing a model against a human
    annotator, where both are guessing some of the time.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        Cohen's kappa in ``[-1, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 1, 0, 0], "p": [1, 1, 0, 0]})
            >>> ds.agg(m=bt.cohen_kappa("y", "p")).to_pydict()
            {'m': [1.0]}
    """
    tp = true_positives(y_true, y_pred, positive=positive)
    fp = false_positives(y_true, y_pred, positive=positive)
    fn = false_negatives(y_true, y_pred, positive=positive)
    tn = true_negatives(y_true, y_pred, positive=positive)
    total = tp + fp + fn + tn
    observed = (tp + tn) / total
    expected = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total)
    return (observed - expected) / (lit(1.0) - expected)


def prevalence(y_true: IntoExpr, *, positive: Any = 1) -> Expr:
    """The fraction of rows that are actually positive — the base rate.

    Worth reporting alongside every other metric: precision, `accuracy`, and lift are all
    uninterpretable without it.

    Args:
        y_true: The observed labels.
        positive: The value that counts as the positive class.

    Returns:
        The positive base rate in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 0, 0, 0]})
            >>> ds.agg(m=bt.prevalence("y")).to_pydict()
            {'m': [0.25]}
    """
    column = _as_column(y_true)
    return count_if(column == lit(positive)) / count_if(column.is_not_null())


def false_negative_rate(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """The false negative rate — the share of actual positives the model misses (the miss rate).

    ``fn / (fn + tp) = 1 - recall``: of everything truly positive, the fraction called negative.
    It is the Type II error, the number that matters when a missed positive is the costly outcome
    (an undetected fraud, a missed diagnosis) and a high accuracy can still hide it.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        The false negative rate in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 1, 1, 0], "p": [1, 0, 0, 0]})
            >>> ds.agg(m=bt.false_negative_rate("y", "p")).to_pydict()
            {'m': [0.6666666666666666]}
    """
    tp = true_positives(y_true, y_pred, positive=positive)
    fn = false_negatives(y_true, y_pred, positive=positive)
    return fn / (tp + fn)
