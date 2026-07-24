"""Fairness metrics — does the model treat groups differently, and how.

A model can be accurate overall and systematically worse for one group, and no aggregate
metric shows it. Fairness metrics compare the model's behaviour *across* a protected
attribute — approval rates, error rates, precision — and report the gap. Which gap matters
depends on the setting, and the disagreement between the definitions is itself the point:
you cannot minimize all of them at once, so naming them separately forces the choice to be
explicit rather than accidental.

Each is a grouped aggregate over the protected attribute, so a fairness audit over a billion
scored rows is one pass, and each returns the disparity between the best- and worst-treated
group so a single number can gate a release. `by_group` variants return the full per-group
table when the summary is not enough.

None of these is a verdict. A demographic-parity gap can be justified when the base rates
genuinely differ; an equal-opportunity gap usually cannot. The metric measures; the decision
about whether a gap is acceptable is not one a function can make.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher.plan.expr_ir.constructors import col
from batcher.plan.functions.aggregate import count_if
from batcher.plan.functions.metrics.classification import positive_mask

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = [
    "demographic_parity_difference",
    "disparate_impact_ratio",
    "equal_opportunity_difference",
    "equalized_odds_difference",
    "group_fairness_report",
    "predictive_parity_difference",
]


def _require(ds: Dataset, *names: str) -> None:
    """Raise a `ColumnNotFoundError` naming the closest real column for any missing name."""
    for name in names:
        if name not in ds.columns:
            from batcher._internal.errors import ColumnNotFoundError, unknown_message

            raise ColumnNotFoundError(
                unknown_message("column", name, ds.columns, hint="Pass an existing column.")
            )


def _group_rates(ds: Dataset, group: str, rate: Any) -> list[float]:
    """The per-group value of a rate expression, dropping groups where it is undefined."""
    rows = ds.group_by(group).agg(__bt_rate=rate).collect()
    values = rows.column("__bt_rate").to_pylist()
    return [float(v) for v in values if v is not None]


def demographic_parity_difference(
    ds: Dataset, group: str, y_pred: str, *, positive: Any = 1
) -> float:
    """The gap in positive-prediction rate between the most- and least-favoured group.

    Also called statistical parity. It asks only whether the model *selects* each group at
    the same rate, ignoring whether the selections are correct — the right lens for an
    allocation decision (a loan, an interview, an ad) where the concern is access rather than
    accuracy. 0 is exact parity.

    A non-zero gap is not automatically unfair: if the groups genuinely differ in the outcome,
    equal selection would itself be a distortion. This measures; it does not judge.

    Args:
        ds: The scored dataset.
        group: The protected-attribute column.
        y_pred: The predicted-label column.
        positive: The prediction value that counts as favourable.

    Returns:
        The largest minus the smallest group selection rate, in ``[0, 1]``.

    Raises:
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import demographic_parity_difference
            >>> ds = bt.from_pydict({"g": ["a", "a", "b", "b"], "p": [1, 1, 0, 1]})
            >>> demographic_parity_difference(ds, "g", "p")
            0.5
    """
    _require(ds, group, y_pred)
    rate = count_if(positive_mask(col(y_pred), positive)) / col(y_pred).count()
    rates = _group_rates(ds, group, rate)
    return max(rates) - min(rates) if rates else float("nan")


def disparate_impact_ratio(ds: Dataset, group: str, y_pred: str, *, positive: Any = 1) -> float:
    """The ratio of the lowest to the highest group selection rate — the 80% rule's number.

    The multiplicative form of demographic parity, and the one written into US hiring law: a
    ratio below 0.8 is the conventional threshold for adverse impact. 1.0 is exact parity;
    lower means the least-favoured group is selected proportionally less often.

    Args:
        ds: The scored dataset.
        group: The protected-attribute column.
        y_pred: The predicted-label column.
        positive: The prediction value that counts as favourable.

    Returns:
        The min-over-max selection-rate ratio, in ``[0, 1]``.

    Raises:
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import disparate_impact_ratio
            >>> ds = bt.from_pydict({"g": ["a", "a", "b", "b"], "p": [1, 1, 1, 0]})
            >>> disparate_impact_ratio(ds, "g", "p")
            0.5
    """
    _require(ds, group, y_pred)
    rate = count_if(positive_mask(col(y_pred), positive)) / col(y_pred).count()
    rates = _group_rates(ds, group, rate)
    if not rates or max(rates) == 0.0:
        return float("nan")
    return min(rates) / max(rates)


def equal_opportunity_difference(
    ds: Dataset, group: str, y_true: str, y_pred: str, *, positive: Any = 1
) -> float:
    """The gap in true-positive rate between groups — equal opportunity for the qualified.

    Restricts attention to the rows that *should* be selected (the actual positives) and asks
    whether the model finds them at the same rate in each group. This is usually the right
    fairness criterion when a positive outcome is a benefit and a miss is a harm: it demands
    that a qualified applicant has the same chance whatever their group, without forcing equal
    selection of the unqualified. 0 is equal opportunity.

    Args:
        ds: The scored dataset.
        group: The protected-attribute column.
        y_true: The true-label column.
        y_pred: The predicted-label column.
        positive: The value that counts as the positive class.

    Returns:
        The largest minus the smallest per-group true-positive rate, in ``[0, 1]``.

    Raises:
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import equal_opportunity_difference
            >>> ds = bt.from_pydict(
            ...     {"g": ["a", "a", "b", "b"], "y": [1, 1, 1, 1], "p": [1, 1, 1, 0]}
            ... )
            >>> equal_opportunity_difference(ds, "g", "y", "p")
            0.5
    """
    _require(ds, group, y_true, y_pred)
    is_positive = positive_mask(col(y_true), positive)
    tpr = count_if(is_positive & positive_mask(col(y_pred), positive)) / count_if(is_positive)
    rates = _group_rates(ds, group, tpr)
    return max(rates) - min(rates) if rates else float("nan")


def equalized_odds_difference(
    ds: Dataset, group: str, y_true: str, y_pred: str, *, positive: Any = 1
) -> float:
    """The larger of the true-positive-rate gap and the false-positive-rate gap across groups.

    The strictest of the common criteria: it requires the model to have both the same hit
    rate *and* the same false-alarm rate in every group, and reports whichever disparity is
    worse. Equalized odds is the right bar when both a miss and a false alarm carry real cost
    and neither can be traded across groups. 0 means the error profiles match.

    Args:
        ds: The scored dataset.
        group: The protected-attribute column.
        y_true: The true-label column.
        y_pred: The predicted-label column.
        positive: The value that counts as the positive class.

    Returns:
        The maximum of the TPR gap and the FPR gap, in ``[0, 1]``.

    Raises:
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import equalized_odds_difference
            >>> ds = bt.from_pydict(
            ...     {"g": ["a", "a", "b", "b"], "y": [1, 0, 1, 0], "p": [1, 0, 1, 1]}
            ... )
            >>> equalized_odds_difference(ds, "g", "y", "p")
            1.0
    """
    _require(ds, group, y_true, y_pred)
    is_positive = positive_mask(col(y_true), positive)
    is_predicted = positive_mask(col(y_pred), positive)
    tpr = count_if(is_positive & is_predicted) / count_if(is_positive)
    fpr = count_if(~is_positive & is_predicted) / count_if(~is_positive)
    tpr_rates = _group_rates(ds, group, tpr)
    fpr_rates = _group_rates(ds, group, fpr)
    tpr_gap = max(tpr_rates) - min(tpr_rates) if tpr_rates else float("nan")
    fpr_gap = max(fpr_rates) - min(fpr_rates) if fpr_rates else float("nan")
    return max(tpr_gap, fpr_gap)


def predictive_parity_difference(
    ds: Dataset, group: str, y_true: str, y_pred: str, *, positive: Any = 1
) -> float:
    """The gap in precision across groups — does a positive prediction mean the same thing.

    Predictive parity asks whether a positive prediction carries the same probability of being
    correct in every group. It is the criterion that matters when the people acted on are the
    *predicted* positives and the concern is that a flag means the same for everyone — the
    fairness definition at the centre of the COMPAS recidivism debate. 0 is predictive parity.

    Args:
        ds: The scored dataset.
        group: The protected-attribute column.
        y_true: The true-label column.
        y_pred: The predicted-label column.
        positive: The value that counts as the positive class.

    Returns:
        The largest minus the smallest per-group precision, in ``[0, 1]``.

    Raises:
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import predictive_parity_difference
            >>> ds = bt.from_pydict(
            ...     {"g": ["a", "a", "b", "b"], "y": [1, 1, 1, 0], "p": [1, 1, 1, 1]}
            ... )
            >>> predictive_parity_difference(ds, "g", "y", "p")
            0.5
    """
    _require(ds, group, y_true, y_pred)
    is_positive = positive_mask(col(y_true), positive)
    is_predicted = positive_mask(col(y_pred), positive)
    precision = count_if(is_positive & is_predicted) / count_if(is_predicted)
    rates = _group_rates(ds, group, precision)
    return max(rates) - min(rates) if rates else float("nan")


def group_fairness_report(
    ds: Dataset, group: str, y_true: str, y_pred: str, *, positive: Any = 1
) -> Dataset:
    """The per-group rates a fairness audit reads — one row per group, computed in one pass.

    Rather than a single disparity number, this returns the underlying rates for every group,
    which is what a review actually needs: the selection rate, the true- and false-positive
    rates, the precision, and the support. The disparity metrics are the max-minus-min of
    these columns; the table is where the disagreement between them becomes visible.

    Args:
        ds: The scored dataset.
        group: The protected-attribute column.
        y_true: The true-label column.
        y_pred: The predicted-label column.
        positive: The value that counts as the positive class.

    Returns:
        A `Dataset` of ``group``, ``support``, ``selection_rate``, ``true_positive_rate``,
        ``false_positive_rate``, ``precision``, ordered by group.

    Raises:
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import group_fairness_report
            >>> ds = bt.from_pydict(
            ...     {"g": ["a", "b"], "y": [1, 0], "p": [1, 1]}
            ... )
            >>> group_fairness_report(ds, "g", "y", "p").columns[:3]
            ['g', 'support', 'selection_rate']
    """
    _require(ds, group, y_true, y_pred)
    is_positive = positive_mask(col(y_true), positive)
    is_predicted = positive_mask(col(y_pred), positive)
    return (
        ds.group_by(group)
        .agg(
            support=col(y_true).count(),
            selection_rate=count_if(is_predicted) / col(y_pred).count(),
            true_positive_rate=count_if(is_positive & is_predicted) / count_if(is_positive),
            false_positive_rate=count_if(~is_positive & is_predicted) / count_if(~is_positive),
            precision=count_if(is_positive & is_predicted) / count_if(is_predicted),
        )
        .sort(group)
    )
