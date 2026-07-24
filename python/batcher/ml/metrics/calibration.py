"""Calibration — whether a predicted probability means what it says.

A model can rank perfectly and still lie about its confidence. A fraud model that assigns
0.9 to a batch of transactions that turn out 60% fraudulent has an excellent AUC and a
useless probability, and every decision that multiplies that probability by a dollar amount
is wrong. Calibration is the property AUC cannot see, and these are the numbers that measure
it.

`expected_calibration_error`
    The headline number: the average gap between predicted confidence and observed accuracy,
    weighted by how many predictions fall in each confidence band. 0 is perfect; anything
    above ~0.05 is worth fixing before the probabilities are used as probabilities.
`maximum_calibration_error`
    The *worst* band's gap rather than the average. The number to watch when one region of
    the score range feeds a high-stakes decision — a good average can still hide a terrible
    band.
`brier_skill_score`
    The Brier score rescaled against always predicting the base rate, so it reads like R² —
    1 is perfect, 0 is no better than the base rate, negative is worse. The single number
    that says whether the probabilities add value at all.

All three reduce to one bucketed aggregate over the scored rows, so a calibration report
over a billion predictions is one pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.metrics.tables import calibration_curve
from batcher.plan.functions.metrics.classification import positive_mask

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = [
    "brier_skill_score",
    "expected_calibration_error",
    "maximum_calibration_error",
]


def _calibration_bins(
    ds: Dataset, y_true: str, y_score: str, bins: int, positive: Any
) -> tuple[list[float], list[float], list[float]]:
    """Per-bin ``(row_count, mean_predicted, observed_rate)`` from the calibration curve."""
    curve = calibration_curve(ds, y_true, y_score, bins=bins, positive=positive).to_pydict()
    return (
        [float(n) for n in curve["rows"]],
        [float(p) for p in curve["mean_predicted"]],
        [float(o) for o in curve["observed_rate"]],
    )


def expected_calibration_error(
    ds: Dataset,
    y_true: str,
    y_score: str,
    *,
    bins: int = 10,
    positive: Any = 1,
) -> float:
    """The support-weighted average gap between predicted confidence and observed accuracy.

    Bins the predictions by confidence, and in each bin compares the mean predicted
    probability against the actual positive rate. The ECE is the average of those gaps,
    weighted by how many predictions each bin holds — so a bias in a densely populated
    confidence band counts for more than the same bias in a rare one, which is the honest
    weighting.

    Args:
        ds: The scored dataset.
        y_true: The label column.
        y_score: The predicted probability, in ``[0, 1]``.
        bins: How many equal-width confidence bins to use.
        positive: The label value that counts as the positive class.

    Returns:
        The expected calibration error in ``[0, 1]``; 0 is perfectly calibrated.

    Raises:
        PlanError: If `bins` is less than 2.
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import expected_calibration_error
            >>> ds = bt.from_pydict({"y": [0, 0, 1, 1], "s": [0.0, 0.0, 1.0, 1.0]})
            >>> expected_calibration_error(ds, "y", "s", bins=2)
            0.0
    """
    if bins < 2:
        raise PlanError(f"expected_calibration_error needs at least 2 bins, got {bins}")
    counts, predicted, observed = _calibration_bins(ds, y_true, y_score, bins, positive)
    total = sum(counts)
    if total == 0:
        return float("nan")
    return sum(n * abs(p - o) for n, p, o in zip(counts, predicted, observed, strict=True)) / total


def maximum_calibration_error(
    ds: Dataset,
    y_true: str,
    y_score: str,
    *,
    bins: int = 10,
    positive: Any = 1,
) -> float:
    """The largest single confidence band's gap between prediction and reality.

    Where `expected_calibration_error` averages, this takes the maximum — the number to
    report when one region of the score range drives a high-stakes call and a good average
    could still hide a badly miscalibrated band. Empty bins are ignored rather than counted
    as a perfect zero.

    Args:
        ds: The scored dataset.
        y_true: The label column.
        y_score: The predicted probability, in ``[0, 1]``.
        bins: How many equal-width confidence bins to use.
        positive: The label value that counts as the positive class.

    Returns:
        The maximum calibration error in ``[0, 1]``.

    Raises:
        PlanError: If `bins` is less than 2.
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import maximum_calibration_error
            >>> ds = bt.from_pydict({"y": [0, 1, 1, 1], "s": [0.1, 0.2, 0.3, 0.4]})
            >>> round(maximum_calibration_error(ds, "y", "s", bins=2), 4)
            0.5
    """
    if bins < 2:
        raise PlanError(f"maximum_calibration_error needs at least 2 bins, got {bins}")
    counts, predicted, observed = _calibration_bins(ds, y_true, y_score, bins, positive)
    gaps = [abs(p - o) for n, p, o in zip(counts, predicted, observed, strict=True) if n > 0]
    return max(gaps) if gaps else float("nan")


def brier_skill_score(
    ds: Dataset,
    y_true: str,
    y_score: str,
    *,
    positive: Any = 1,
) -> float:
    """The Brier score rescaled against the base rate — reads like R² for probabilities.

    The Brier score alone is hard to read: 0.18 is good or bad depending on the base rate. The
    skill score fixes the reference, comparing the model's Brier score against the Brier score
    of always predicting the overall positive rate. 1 is perfect, 0 is no better than the base
    rate, negative means the model's probabilities are actively worse than a constant.

    Args:
        ds: The scored dataset.
        y_true: The label column.
        y_score: The predicted probability, in ``[0, 1]``.
        positive: The label value that counts as the positive class.

    Returns:
        The Brier skill score; at most 1.

    Raises:
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import brier_skill_score
            >>> ds = bt.from_pydict({"y": [0, 0, 1, 1], "s": [0.0, 0.0, 1.0, 1.0]})
            >>> brier_skill_score(ds, "y", "s")
            1.0
    """
    import batcher as bt
    from batcher.plan.expr_ir.constructors import col, lit, when

    for name in (y_true, y_score):
        if name not in ds.columns:
            from batcher._internal.errors import ColumnNotFoundError, unknown_message

            raise ColumnNotFoundError(
                unknown_message("column", name, ds.columns, hint="Pass an existing column.")
            )
    actual = when(positive_mask(col(y_true), positive)).then(lit(1.0)).otherwise(lit(0.0))
    error = col(y_score) - actual
    row = (
        ds.with_columns(__bt_actual=actual, __bt_sq=error * error)
        .agg(
            model_brier=bt.mean(col("__bt_sq")),
            base_rate=bt.mean(col("__bt_actual")),
        )
        .collect()
    )
    model_brier = row.column("model_brier")[0].as_py()
    base_rate = row.column("base_rate")[0].as_py()
    if model_brier is None or base_rate is None:
        return float("nan")
    # The reference Brier score of always predicting the base rate p is p(1-p).
    reference = base_rate * (1.0 - base_rate)
    if reference == 0.0:
        # A single-class dataset: the constant predictor is already perfect, so there is no
        # skill to measure. Report 0 (no improvement possible) rather than dividing by zero.
        return 0.0
    return 1.0 - model_brier / reference
