"""Isotonic calibration — a monotone step function fitted to the observed rates.

`PlattCalibrator` assumes the miscalibration is sigmoid-shaped, which is a strong assumption
and often a wrong one: a boosted tree's overconfidence is typically asymmetric, and a model
trained on a resampled class balance is shifted in a way no sigmoid describes. Isotonic
regression assumes only that a higher score should not mean a lower probability, and fits
the best non-decreasing step function under that constraint.

That flexibility is also its cost. With enough steps it will fit the calibration split's
noise, so it wants more data than Platt scaling — a few thousand rows rather than a few
hundred — and `n_bins` is the knob that trades one against the other.

The fit stays relational: the scores are bucketed by quantile and the positive rate per
bucket is one grouped aggregate, so the data pass is mergeable and distributed. Only the
pool-adjacent-violators pass runs on the driver, over at most `n_bins` values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor, column_arg
from batcher.ml.preprocessors.calibration.platt import binary_label, require_calibration_columns
from batcher.plan.expr_ir.constructors import col, lit

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["IsotonicCalibrator", "pool_adjacent_violators"]


def pool_adjacent_violators(values: list[float], weights: list[float]) -> list[float]:
    """Fit the best non-decreasing sequence to `values` by weighted least squares.

    The pool-adjacent-violators algorithm: walk left to right, and whenever a value is lower
    than the one before it, merge the two into their weighted mean and re-check backwards.
    The result is the closest non-decreasing sequence in weighted squared error, computed in
    one linear pass.

    Args:
        values: The observed value per bucket, in increasing bucket order.
        weights: How many observations each bucket's value is averaged over.

    Returns:
        The fitted non-decreasing values, one per input bucket.

    Examples:
        .. doctest::

            >>> from batcher.ml.preprocessors.calibration.isotonic import (
            ...     pool_adjacent_violators,
            ... )
            >>> pool_adjacent_violators([0.1, 0.9, 0.5], [1.0, 1.0, 1.0])
            [0.1, 0.7, 0.7]
    """
    # Each block is [weighted sum, total weight, how many buckets it covers].
    blocks: list[list[float]] = []
    for value, weight in zip(values, weights, strict=True):
        blocks.append([value * weight, weight, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            total, weight_sum, count = blocks.pop()
            blocks[-1][0] += total
            blocks[-1][1] += weight_sum
            blocks[-1][2] += count
        # The merge can leave the *new* block below its own predecessor, so the loop above
        # runs again on the next iteration through this same condition.
    out: list[float] = []
    for total, weight_sum, count in blocks:
        out.extend([total / weight_sum] * int(count))
    return out


class IsotonicCalibrator(Preprocessor):
    """Map raw scores onto calibrated probabilities with a fitted monotone step function.

    Like `PlattCalibrator`, fit this on a split the model did not train on. Unlike it, this
    makes no assumption about the *shape* of the miscalibration — only that a higher score
    should never mean a lower probability.

    A score below the first bucket's boundary takes the first fitted value and one above the
    last takes the last, so the calibration is defined everywhere rather than extrapolating
    a trend it has no evidence for.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import IsotonicCalibrator
            >>> ds = bt.from_pydict(
            ...     {"score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            ...      "label": [0, 0, 0, 0, 1, 1, 1, 1]}
            ... )
            >>> out = IsotonicCalibrator("score", "label", n_bins=2).fit_transform(ds)
            >>> out.to_pydict()["calibrated"]
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    Args:
        score_column: The column holding the model's raw score.
        label_column: The column holding the truth, used only by `fit`.
        output_column: Where to write the calibrated probability.
        positive: The label value counted as the positive class.
        n_bins: How many score buckets to fit. More bins follow the data more closely and
            overfit a small calibration split sooner.
    """

    __slots__ = (
        "label_column",
        "n_bins",
        "output_column",
        "positive",
        "score_column",
        "thresholds_",
        "values_",
    )

    def __init__(
        self,
        score_column: str,
        label_column: str,
        *,
        output_column: str = "calibrated",
        positive: object = 1,
        n_bins: int = 100,
    ) -> None:
        self.score_column = column_arg(score_column, what="IsotonicCalibrator")
        self.label_column = column_arg(label_column, what="IsotonicCalibrator")
        self.output_column = output_column
        self.positive = positive
        if n_bins < 2:
            raise PlanError(f"IsotonicCalibrator: n_bins must be at least 2, got {n_bins}")
        self.n_bins = n_bins
        self.thresholds_: list[float] = []
        self.values_: list[float] = []

    def fit(self, ds: Dataset) -> IsotonicCalibrator:
        """Bucket the scores, measure each bucket's positive rate, and make it monotone.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import IsotonicCalibrator
                >>> ds = bt.from_pydict(
                ...     {"score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                ...      "label": [0, 0, 0, 0, 1, 1, 1, 1]}
                ... )
                >>> IsotonicCalibrator("score", "label", n_bins=2).fit(ds).values_
                [0.0, 1.0]

        Args:
            ds: The held-out split carrying both the scores and the labels.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If the split is empty.
            ColumnNotFoundError: If the score or label column is missing.
        """
        from batcher.plan.functions.aggregate import mean

        require_calibration_columns(
            ds, self.score_column, self.label_column, what="IsotonicCalibrator"
        )
        prepared = ds.select(
            __bt_score=col(self.score_column).cast("float64"),
            __bt_label=binary_label(self.label_column, self.positive),
        ).filter(col("__bt_score").is_not_null())
        cuts = self._cut_points(prepared)
        # The bucket index is a sum of threshold indicators — one vectorized expression
        # rather than a search per row, the same shape `QuantileTransformer` uses.
        bucket = lit(0)
        for cut in cuts:
            bucket = bucket + (col("__bt_score") > lit(cut)).cast("int64")
        grouped = (
            prepared.with_columns(__bt_bin=bucket)
            .group_by("__bt_bin")
            .agg(__bt_rate=mean(col("__bt_label")), __bt_n=col("__bt_label").count())
            .sort("__bt_bin")
            .collect()
        )
        rates = [float(v) for v in grouped.column("__bt_rate").to_pylist()]
        weights = [float(v) for v in grouped.column("__bt_n").to_pylist()]
        if not rates:
            raise PlanError(
                "IsotonicCalibrator: the calibration split has no non-null scores, so there "
                "is nothing to calibrate."
            )
        present = [int(v) for v in grouped.column("__bt_bin").to_pylist()]
        # An empty bucket produces no group, so the thresholds are re-derived from the
        # buckets that actually carried rows. Keeping the full cut list would leave steps
        # with no evidence behind them sitting between two fitted values.
        self.thresholds_ = [cuts[i - 1] for i in present[1:]]
        self.values_ = pool_adjacent_violators(rates, weights)
        self._fitted = True
        return self

    def _cut_points(self, prepared: Dataset) -> list[float]:
        """The interior bucket boundaries, at evenly spaced quantiles of the score."""
        aggs = {
            f"q{i}": col("__bt_score").approx_quantile(i / self.n_bins)
            for i in range(1, self.n_bins)
        }
        row = prepared.agg(**aggs).collect()
        seen: list[float] = []
        for i in range(1, self.n_bins):
            value = row.column(f"q{i}")[0].as_py()
            if value is None:
                continue
            # A repeated quantile is a tie in the score, and a bucket boundary inside a tie
            # would split identical scores into different calibrated values.
            if not seen or float(value) > seen[-1]:
                seen.append(float(value))
        return seen

    def transform(self, ds: Dataset) -> Dataset:
        """Append the calibrated probability for each row, lazily.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import IsotonicCalibrator
                >>> train = bt.from_pydict(
                ...     {"score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                ...      "label": [0, 0, 0, 0, 1, 1, 1, 1]}
                ... )
                >>> pre = IsotonicCalibrator("score", "label", n_bins=2).fit(train)
                >>> pre.transform(bt.from_pydict({"score": [0.05]})).to_pydict()["calibrated"]
                [0.0]

        Args:
            ds: The dataset whose scores to calibrate.

        Returns:
            A new lazy `Dataset` with the calibrated column appended.
        """
        self._require_fitted()
        score = col(self.score_column).cast("float64")
        # A step function as a sum of increments: start at the first fitted value and add
        # each step's rise once the score passes its boundary. PAVA produces long runs of
        # equal values, so most increments are zero and fold away.
        calibrated = lit(self.values_[0])
        previous = self.values_[0]
        for cut, value in zip(self.thresholds_, self.values_[1:], strict=True):
            calibrated = calibrated + (score > lit(cut)).cast("float64") * lit(value - previous)
            previous = value
        return ds.with_columns(**{self.output_column: calibrated})
