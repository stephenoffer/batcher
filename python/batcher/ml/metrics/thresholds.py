"""Choosing an operating point — the step between a good AUC and a deployed model.

A classifier does not output a decision; it outputs a score. Turning that into an action
needs a cutoff, and 0.5 is almost never the right one. It is right only when the classes are
balanced *and* a false positive costs exactly what a false negative costs, which is close to
never true: a missed fraud costs the transaction, a false alarm costs a phone call.

Two ways to pick the cutoff, and which you use depends on whether you can put numbers on the
outcomes:

`best_threshold`
    Maximize a metric — F1, or an F-beta weighted toward recall. Use it when the costs are
    real but not quantified.
`best_cost_threshold`
    Minimize expected cost, given what a false positive and a false negative actually cost.
    Use it whenever you *can* name those numbers, because it is the only choice that
    optimizes the thing you care about rather than a proxy for it.

Both are one `threshold_sweep` — a single pass over the scored rows — plus an argmax over a
few hundred candidate cutoffs on the driver. Neither re-scans per candidate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.metrics.tables import threshold_sweep

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["best_cost_threshold", "best_threshold", "expected_cost_curve"]

#: The metrics `best_threshold` can maximize, and how each is derived from the sweep row.
_OBJECTIVES = ("f1", "fbeta", "precision", "recall", "youden")


def _sweep_rows(
    ds: Dataset, y_true: str, y_score: str, thresholds: int, positive: Any
) -> list[dict[str, float]]:
    """The threshold sweep as a list of plain dicts, one per candidate cutoff."""
    table = threshold_sweep(
        ds, y_true, y_score, thresholds=thresholds, positive=positive
    ).to_pydict()
    names = list(table)
    return [
        {name: table[name][index] for name in names} for index in range(len(table["threshold"]))
    ]


def _objective_value(row: dict[str, float], objective: str, beta: float) -> float:
    """One sweep row's score under the chosen objective, with 0 for an undefined one."""
    precision = row["precision"] or 0.0
    recall = row["recall"] or 0.0
    if objective == "precision":
        return precision
    if objective == "recall":
        return recall
    if objective == "youden":
        # Youden's J: sensitivity + specificity - 1, which is where the ROC curve is
        # furthest from the diagonal. The threshold-free choice when costs are symmetric.
        negatives = row["fp"] + row["tn"]
        specificity = (row["tn"] / negatives) if negatives else 0.0
        return recall + specificity - 1.0
    squared = 1.0 if objective == "f1" else beta * beta
    denominator = squared * precision + recall
    return (1.0 + squared) * precision * recall / denominator if denominator else 0.0


def best_threshold(
    ds: Dataset,
    y_true: str,
    y_score: str,
    *,
    objective: str = "f1",
    beta: float = 1.0,
    thresholds: int = 100,
    positive: Any = 1,
) -> dict[str, float]:
    """The cutoff that maximizes `objective`, with the confusion counts it produces.

    0.5 is the right cutoff only when the classes are balanced *and* the two error types
    cost the same. On a 2%-positive dataset the F1-optimal cutoff is routinely below 0.1,
    and using 0.5 there means the model predicts "negative" for everything and looks
    excellent on accuracy.

    Objectives:

    ``"f1"``
        The balanced harmonic mean. The default when you have no further information.
    ``"fbeta"``
        Weighted toward recall by `beta`; ``beta=2`` when a miss costs more than a false
        alarm.
    ``"precision"`` / ``"recall"``
        Maximize one directly. Both are degenerate on their own — precision peaks where
        almost nothing is predicted positive — so pair them with a floor on the other.
    ``"youden"``
        Sensitivity plus specificity minus one, the point furthest from the ROC diagonal.
        The class-balance-independent choice.

    Args:
        ds: The scored dataset.
        y_true: The label column.
        y_score: The predicted probability, assumed to be in ``[0, 1]``.
        objective: What to maximize; see the list above.
        beta: The recall weight, for ``objective="fbeta"``.
        thresholds: How many candidate cutoffs to evaluate; the resolution of the answer.
        positive: The label value that counts as the positive class.

    Returns:
        A dict with ``threshold``, ``objective_value``, ``precision``, ``recall``, ``f1``,
        and the four confusion counts at that cutoff.

    Raises:
        PlanError: On an unknown `objective`, or fewer than 2 thresholds.
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import best_threshold
            >>> ds = bt.from_pydict({"y": [0, 0, 1, 1], "s": [0.1, 0.2, 0.8, 0.9]})
            >>> round(best_threshold(ds, "y", "s", thresholds=10)["f1"], 4)
            1.0
    """
    if objective not in _OBJECTIVES:
        from batcher._internal.errors import suggestion

        hint = suggestion(objective, _OBJECTIVES)
        tail = f" {hint}" if hint else ""
        raise PlanError(f"objective must be one of {sorted(_OBJECTIVES)}, got {objective!r}.{tail}")
    rows = _sweep_rows(ds, y_true, y_score, thresholds, positive)
    if not rows:
        raise PlanError("best_threshold needs at least one scored row")
    best = max(rows, key=lambda row: _objective_value(row, objective, beta))
    return {
        "threshold": best["threshold"],
        "objective_value": _objective_value(best, objective, beta),
        "precision": best["precision"],
        "recall": best["recall"],
        "f1": best["f1"],
        "tp": best["tp"],
        "fp": best["fp"],
        "fn": best["fn"],
        "tn": best["tn"],
    }


def expected_cost_curve(
    ds: Dataset,
    y_true: str,
    y_score: str,
    *,
    cost_false_positive: float,
    cost_false_negative: float,
    cost_true_positive: float = 0.0,
    thresholds: int = 100,
    positive: Any = 1,
) -> Dataset:
    """The total expected cost at every candidate cutoff, as a table.

    Turns "which threshold" into an arithmetic question. Give the cost of each error type in
    whatever unit matters — currency, minutes of analyst time, churned customers — and each
    row of the result says what operating at that cutoff would have cost on this data.

    `cost_true_positive` is there because acting on a correct positive is rarely free: a
    caught fraud still costs a review, a targeted offer still costs the discount. Leaving it
    at 0 measures error cost alone.

    Args:
        ds: The scored dataset.
        y_true: The label column.
        y_score: The predicted probability, assumed to be in ``[0, 1]``.
        cost_false_positive: What one false alarm costs.
        cost_false_negative: What one miss costs.
        cost_true_positive: What acting on one correct positive costs.
        thresholds: How many candidate cutoffs to evaluate.
        positive: The label value that counts as the positive class.

    Returns:
        A lazy `Dataset` of ``threshold``, ``tp``, ``fp``, ``fn``, ``tn``, ``total_cost``,
        ``cost_per_row``, ordered by ascending total cost.

    Raises:
        PlanError: If either error cost is negative.
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import expected_cost_curve
            >>> ds = bt.from_pydict({"y": [0, 0, 1, 1], "s": [0.1, 0.2, 0.8, 0.9]})
            >>> curve = expected_cost_curve(
            ...     ds, "y", "s", cost_false_positive=1.0, cost_false_negative=10.0
            ... )
            >>> curve.to_pydict()["total_cost"][0]
            0.0
    """
    import batcher as bt

    if cost_false_positive < 0 or cost_false_negative < 0:
        raise PlanError(
            f"costs must be non-negative, got fp={cost_false_positive}, fn={cost_false_negative}"
        )
    rows = _sweep_rows(ds, y_true, y_score, thresholds, positive)
    total_rows = float(sum((rows[0][c] for c in ("tp", "fp", "fn", "tn")), 0.0)) if rows else 0.0
    table: dict[str, list[Any]] = {
        "threshold": [],
        "tp": [],
        "fp": [],
        "fn": [],
        "tn": [],
        "total_cost": [],
        "cost_per_row": [],
    }
    for row in rows:
        cost = (
            row["fp"] * cost_false_positive
            + row["fn"] * cost_false_negative
            + row["tp"] * cost_true_positive
        )
        table["threshold"].append(row["threshold"])
        for cell in ("tp", "fp", "fn", "tn"):
            table[cell].append(row[cell])
        table["total_cost"].append(cost)
        table["cost_per_row"].append(cost / total_rows if total_rows else float("nan"))
    return bt.from_pydict(table).sort("total_cost")


def best_cost_threshold(
    ds: Dataset,
    y_true: str,
    y_score: str,
    *,
    cost_false_positive: float,
    cost_false_negative: float,
    cost_true_positive: float = 0.0,
    thresholds: int = 100,
    positive: Any = 1,
) -> dict[str, float]:
    """The cutoff minimizing expected cost, given what each error actually costs.

    The right way to pick an operating point whenever the costs can be named, because it
    optimizes the thing you care about rather than a proxy for it. F1 implicitly assumes a
    false positive and a false negative cost the same; at a 10:1 ratio the F1-optimal cutoff
    can cost more than twice the minimum.

    Args:
        ds: The scored dataset.
        y_true: The label column.
        y_score: The predicted probability, assumed to be in ``[0, 1]``.
        cost_false_positive: What one false alarm costs.
        cost_false_negative: What one miss costs.
        cost_true_positive: What acting on one correct positive costs.
        thresholds: How many candidate cutoffs to evaluate.
        positive: The label value that counts as the positive class.

    Returns:
        A dict with ``threshold``, ``total_cost``, ``cost_per_row``, and the four confusion
        counts at that cutoff.

    Raises:
        PlanError: If either error cost is negative.
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import best_cost_threshold
            >>> ds = bt.from_pydict({"y": [0, 0, 1, 1], "s": [0.1, 0.2, 0.8, 0.9]})
            >>> best_cost_threshold(
            ...     ds, "y", "s", cost_false_positive=1.0, cost_false_negative=10.0
            ... )["total_cost"]
            0.0
    """
    curve = expected_cost_curve(
        ds,
        y_true,
        y_score,
        cost_false_positive=cost_false_positive,
        cost_false_negative=cost_false_negative,
        cost_true_positive=cost_true_positive,
        thresholds=thresholds,
        positive=positive,
    ).to_pydict()
    if not curve["threshold"]:
        raise PlanError("best_cost_threshold needs at least one scored row")
    return {name: curve[name][0] for name in curve}
