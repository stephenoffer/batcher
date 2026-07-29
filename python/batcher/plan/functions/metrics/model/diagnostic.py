"""Diagnostic-test metrics — the epidemiology and medical-ML vocabulary of the confusion matrix.

Everything here is arithmetic over the same four confusion counts as `classification`, and
every one is a name a clinician, an epidemiologist, or a diagnostics team uses by default.
They are not obscure: a false-discovery rate is the complement of precision, informedness is
Youden's J, a likelihood ratio is how a test result updates the odds of disease. Batcher
already computes the four cells in one aggregate, so each of these is one more expression
over them and costs nothing extra to ask for alongside precision and recall.

The value of naming them explicitly is that the alternative is a reader re-deriving
``fp / (fp + tp)`` and getting the direction wrong. A metric with a standard name and a
standard definition is a metric nobody has to check.
"""

from __future__ import annotations

from typing import Any

from batcher.plan.expr_ir.constructors import lit
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, count_if
from batcher.plan.functions.metrics.model.classification import (
    false_negatives,
    false_positives,
    recall,
    specificity,
    true_negatives,
    true_positives,
)

__all__ = [
    "diagnostic_odds_ratio",
    "false_discovery_rate",
    "false_omission_rate",
    "fowlkes_mallows_index",
    "geometric_mean_score",
    "hamming_loss",
    "informedness",
    "jaccard_score",
    "markedness",
    "negative_likelihood_ratio",
    "positive_likelihood_ratio",
    "prevalence_threshold",
]


def jaccard_score(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """The Jaccard index of the positive class — ``tp / (tp + fp + fn)``.

    The overlap between the predicted-positive set and the actually-positive set, ignoring
    the true negatives entirely. That is the right measure when the negatives are the
    uninteresting majority — image segmentation, entity matching, any retrieval-shaped task
    where "how much of what matters did we get right" is the question and the vast negative
    set would swamp accuracy.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        The Jaccard score in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 1, 0, 1], "p": [1, 0, 0, 1]})
            >>> ds.agg(m=bt.jaccard_score("y", "p")).to_pydict()
            {'m': [0.6666666666666666]}
    """
    tp = true_positives(y_true, y_pred, positive=positive)
    fp = false_positives(y_true, y_pred, positive=positive)
    fn = false_negatives(y_true, y_pred, positive=positive)
    return tp / (tp + fp + fn)


def false_discovery_rate(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """The fraction of positive predictions that are wrong — ``fp / (fp + tp)``, ``1 - precision``.

    The number a screening program actually acts on: of everyone the test flagged, how many
    were flagged for nothing. It reads more naturally than precision when the cost is the
    follow-up on a false alarm — a biopsy, an audit, a page.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        The false-discovery rate in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 0, 1], "p": [1, 1, 0]})
            >>> ds.agg(m=bt.false_discovery_rate("y", "p")).to_pydict()
            {'m': [0.5]}
    """
    tp = true_positives(y_true, y_pred, positive=positive)
    fp = false_positives(y_true, y_pred, positive=positive)
    return fp / (fp + tp)


def false_omission_rate(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """The fraction of negative predictions that are wrong — ``fn / (fn + tn)``.

    The complement of the negative predictive value, and the risk that matters when a
    *negative* prediction triggers the action — a "safe to discharge", a "no fraud", a "skip
    review". It is how often that all-clear is mistaken.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        The false-omission rate in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 0, 0], "p": [0, 0, 0]})
            >>> ds.agg(m=bt.false_omission_rate("y", "p")).to_pydict()
            {'m': [0.3333333333333333]}
    """
    fn = false_negatives(y_true, y_pred, positive=positive)
    tn = true_negatives(y_true, y_pred, positive=positive)
    return fn / (fn + tn)


def positive_likelihood_ratio(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """How much a positive result raises the odds — ``sensitivity / (1 - specificity)``.

    The diagnostician's number: multiply the pre-test odds of disease by this to get the
    post-test odds after a positive result. Above 10 is a strongly convincing positive test;
    near 1 the result told you nothing. Unlike precision it does not depend on prevalence, so
    it transfers between populations with different base rates.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        The positive likelihood ratio, at least 0.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 1, 0, 0], "p": [1, 1, 0, 1]})
            >>> ds.agg(m=bt.positive_likelihood_ratio("y", "p")).to_pydict()
            {'m': [2.0]}
    """
    sensitivity = recall(y_true, y_pred, positive=positive)
    return sensitivity / (lit(1.0) - specificity(y_true, y_pred, positive=positive))


def negative_likelihood_ratio(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """How much a negative result lowers the odds — ``(1 - sensitivity) / specificity``.

    The mirror of `positive_likelihood_ratio`: multiply the pre-test odds by this after a
    negative result. Below 0.1 is a strongly convincing negative test — the result that lets
    you rule the condition out. Also prevalence-independent.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        The negative likelihood ratio, at least 0.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 1, 0, 0], "p": [1, 0, 0, 0]})
            >>> ds.agg(m=bt.negative_likelihood_ratio("y", "p")).to_pydict()
            {'m': [0.5]}
    """
    sensitivity = recall(y_true, y_pred, positive=positive)
    return (lit(1.0) - sensitivity) / specificity(y_true, y_pred, positive=positive)


def diagnostic_odds_ratio(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """The single-number test performance — ``(tp * tn) / (fp * fn)``.

    The ratio of the odds of a positive result in a true case to the odds in a false one, and
    the one number that summarizes a diagnostic test's discriminative power independent of
    prevalence. 1 means the test is useless; higher is better; it is the positive likelihood
    ratio divided by the negative one.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        The diagnostic odds ratio, at least 0.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 1, 0, 0], "p": [1, 1, 0, 0]})
            >>> ds.agg(m=bt.diagnostic_odds_ratio("y", "p")).to_pydict()
            {'m': [inf]}
    """
    tp = true_positives(y_true, y_pred, positive=positive)
    fp = false_positives(y_true, y_pred, positive=positive)
    fn = false_negatives(y_true, y_pred, positive=positive)
    tn = true_negatives(y_true, y_pred, positive=positive)
    return (tp * tn) / (fp * fn)


def informedness(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """Youden's J — ``sensitivity + specificity - 1``, the probability of an informed decision.

    How much better than a coin flip the classifier is, using both error rates symmetrically.
    0 is chance, 1 is perfect, negative is worse than chance. It is the height of the ROC
    curve above the diagonal at the operating point, and the quantity `best_threshold`'s
    ``youden`` objective maximizes.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        Informedness in ``[-1, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 1, 0, 0], "p": [1, 0, 0, 0]})
            >>> ds.agg(m=bt.informedness("y", "p")).to_pydict()
            {'m': [0.5]}
    """
    return (
        recall(y_true, y_pred, positive=positive)
        + specificity(y_true, y_pred, positive=positive)
        - lit(1.0)
    )


def markedness(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """``precision + negative_predictive_value - 1`` — informedness from the prediction's side.

    The counterpart of `informedness`: where that combines the two *recall*-side rates, this
    combines the two *precision*-side rates. Their geometric mean is the Matthews correlation
    coefficient, which is why reporting both explains where an MCC comes from.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        Markedness in ``[-1, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 0, 1, 0], "p": [1, 0, 0, 0]})
            >>> ds.agg(m=bt.markedness("y", "p")).to_pydict()
            {'m': [0.6666666666666665]}
    """
    tp = true_positives(y_true, y_pred, positive=positive)
    fp = false_positives(y_true, y_pred, positive=positive)
    fn = false_negatives(y_true, y_pred, positive=positive)
    tn = true_negatives(y_true, y_pred, positive=positive)
    precision = tp / (tp + fp)
    npv = tn / (tn + fn)
    return precision + npv - lit(1.0)


def fowlkes_mallows_index(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """The geometric mean of precision and recall — ``sqrt(precision * recall)``.

    A middle ground between F1's harmonic mean and a plain average, and the standard external
    measure of clustering agreement. It rewards a classifier only when it is good on *both*
    precision and recall, like F1, but weights a shortfall in either slightly less severely.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        The Fowlkes-Mallows index in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 1, 0, 1], "p": [1, 1, 0, 0]})
            >>> round(ds.agg(m=bt.fowlkes_mallows_index("y", "p")).to_pydict()["m"][0], 4)
            0.8165
    """
    tp = true_positives(y_true, y_pred, positive=positive)
    fp = false_positives(y_true, y_pred, positive=positive)
    fn = false_negatives(y_true, y_pred, positive=positive)
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    return (precision * sensitivity).sqrt()


def prevalence_threshold(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """The prevalence below which a positive result is more likely wrong than right.

    ``(sqrt(tpr * fpr) - fpr) / (tpr - fpr)`` over the true- and false-positive rates. Below
    this base rate, the test's positives are dominated by false alarms however good its
    sensitivity — the mathematical statement of why screening a rare condition floods the
    system with false positives. The number that tells you a test good in the clinic is
    useless as a population screen.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        The prevalence threshold in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 1, 1, 0, 0, 0], "p": [1, 1, 0, 0, 0, 1]})
            >>> round(ds.agg(m=bt.prevalence_threshold("y", "p")).to_pydict()["m"][0], 4)
            0.4142
    """
    tpr = recall(y_true, y_pred, positive=positive)
    fpr = lit(1.0) - specificity(y_true, y_pred, positive=positive)
    return ((tpr * fpr).sqrt() - fpr) / (tpr - fpr)


def hamming_loss(y_true: IntoExpr, y_pred: IntoExpr) -> Expr:
    """The fraction of labels predicted wrong — the multi-label error rate, ``mean(y != yhat)``.

    For a single label this is exactly ``1 - accuracy``. It earns its own name in the
    multi-label setting, where each row has been binarized into one indicator column per label
    and the loss is averaged over every ``(row, label)`` cell: a model that gets most labels
    right on most rows scores well even when no row is *entirely* correct, which is the honest
    reading when labels are independent.

    Args:
        y_true: The observed label (or one binarized label column).
        y_pred: The predicted label.

    Returns:
        The Hamming loss in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 0, 1, 1], "p": [1, 0, 0, 1]})
            >>> ds.agg(m=bt.hamming_loss("y", "p")).to_pydict()
            {'m': [0.25]}
    """
    left, right = _as_column(y_true), _as_column(y_pred)
    wrong = count_if(left != right)
    return wrong / count_if(left.is_not_null() & right.is_not_null())


def geometric_mean_score(y_true: IntoExpr, y_pred: IntoExpr, *, positive: Any = 1) -> Expr:
    """The geometric mean of sensitivity and specificity — ``sqrt(recall * specificity)``.

    The imbalanced-classification score that a high accuracy cannot fake: it is high only when
    *both* classes are recalled well, and collapses to zero the moment either is ignored. Where
    `balanced_accuracy` averages the two class recalls, the geometric mean multiplies them, so it
    punishes a lopsided classifier harder — the right single number when the rare class matters
    as much as the common one.

    Args:
        y_true: The observed labels.
        y_pred: The predicted labels.
        positive: The value that counts as the positive class.

    Returns:
        The geometric mean score in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [1, 1, 0, 0], "p": [1, 1, 0, 0]})
            >>> ds.agg(g=bt.geometric_mean_score("y", "p")).to_pydict()
            {'g': [1.0]}
    """
    sensitivity = recall(y_true, y_pred, positive=positive)
    return (sensitivity * specificity(y_true, y_pred, positive=positive)).sqrt()
