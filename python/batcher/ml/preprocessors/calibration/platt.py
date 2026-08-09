"""Platt scaling — a one-dimensional logistic fit turning scores into probabilities.

A model's output being between 0 and 1 does not make it a probability. A boosted tree that
says 0.9 is usually right more than 90% of the time; an SVM's decision value is not on a
probability scale at all. That matters whenever the number is used for anything but ranking
— a cost-sensitive threshold, an expected-value calculation, a downstream model that treats
it as a rate — and none of those fail loudly when the number is wrong.

Platt scaling is the parametric fix: fit ``sigmoid(a * score + b)`` against the labels on a
held-out split and apply it to every future score. Two parameters, so it needs very little
data and cannot overfit much, at the cost of assuming the miscalibration is sigmoid-shaped.
When it is not, use `IsotonicCalibrator`.

The fit is `LogisticRegression` over the single score column, so it is the same IRLS the
rest of the package uses and inherits the same mergeable aggregates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor, column_arg
from batcher.plan.expr_ir.constructors import col, lit

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["PlattCalibrator"]


def require_calibration_columns(ds: Dataset, *names: str, what: str) -> None:
    """Raise unless every named column is present, naming the calibrator and the column.

    Args:
        ds: The dataset being fitted.
        names: The columns the calibrator needs.
        what: The calibrator's class name, for the message.

    Raises:
        ColumnNotFoundError: If a named column is missing.
    """
    available = ds.columns
    present = set(available)
    for name in names:
        if name not in present:
            from batcher._internal.errors import ColumnNotFoundError, unknown_message

            raise ColumnNotFoundError(
                unknown_message(
                    "column", name, available, hint=f"{what} needs the score and label columns."
                )
            )


def binary_label(label_column: str, positive: object) -> object:
    """The 0/1 expression a calibrator fits against, from a label column and its positive class.

    Args:
        label_column: The column holding the truth.
        positive: The value counted as the positive class.

    Returns:
        An expression evaluating to 1.0 for the positive class and 0.0 otherwise.
    """
    return (col(label_column) == lit(positive)).cast("float64")


class PlattCalibrator(Preprocessor):
    """Map raw scores onto calibrated probabilities with a fitted sigmoid.

    Fit on a split the model did not train on. Calibrating on the training split learns the
    model's overconfidence *on data it memorized*, which is not the overconfidence it will
    show in production, and the result is a calibration curve that looks perfect in
    development and is wrong in use.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import PlattCalibrator
            >>> ds = bt.from_pydict(
            ...     {"score": [0.1, 0.2, 0.8, 0.9], "label": [0, 0, 1, 1]}
            ... )
            >>> out = PlattCalibrator("score", "label").fit_transform(ds)
            >>> out.to_pydict()["calibrated"][0] < out.to_pydict()["calibrated"][3]
            True

    Args:
        score_column: The column holding the model's raw score.
        label_column: The column holding the truth, used only by `fit`.
        output_column: Where to write the calibrated probability.
        positive: The label value counted as the positive class.
        max_iter: The maximum number of IRLS iterations for the sigmoid fit.
    """

    __slots__ = (
        "coef_",
        "intercept_",
        "label_column",
        "max_iter",
        "output_column",
        "positive",
        "score_column",
    )

    def __init__(
        self,
        score_column: str,
        label_column: str,
        *,
        output_column: str = "calibrated",
        positive: object = 1,
        max_iter: int = 100,
    ) -> None:
        self.score_column = column_arg(score_column, what="PlattCalibrator")
        self.label_column = column_arg(label_column, what="PlattCalibrator")
        self.output_column = output_column
        self.positive = positive
        self.max_iter = max_iter
        self.coef_ = 0.0
        self.intercept_ = 0.0

    def fit(self, ds: Dataset) -> PlattCalibrator:
        """Fit the sigmoid's slope and intercept against the labels.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import PlattCalibrator
                >>> ds = bt.from_pydict(
                ...     {"score": [0.1, 0.2, 0.8, 0.9], "label": [0, 0, 1, 1]}
                ... )
                >>> PlattCalibrator("score", "label").fit(ds).coef_ > 0
                True

        Args:
            ds: The held-out split carrying both the scores and the labels.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If the split has only one class, which no sigmoid can separate.
            ColumnNotFoundError: If the score or label column is missing.
        """
        from batcher.ml.linear import LogisticRegression

        require_calibration_columns(
            ds, self.score_column, self.label_column, what="PlattCalibrator"
        )
        prepared = ds.select(
            __bt_score=col(self.score_column).cast("float64"),
            __bt_label=binary_label(self.label_column, self.positive),
        )
        classes = prepared.select("__bt_label").distinct().count()
        if classes < 2:
            raise PlanError(
                f"PlattCalibrator: the calibration split has only one class under "
                f"positive={self.positive!r}, so there is nothing to calibrate against. "
                "Check the split and the positive-class value."
            )
        model = LogisticRegression(["__bt_score"], "__bt_label", max_iter=self.max_iter).fit(
            prepared
        )
        self.coef_ = float(model.coef_[0])
        self.intercept_ = float(model.intercept_)
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Append the calibrated probability for each row, lazily.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import PlattCalibrator
                >>> train = bt.from_pydict(
                ...     {"score": [0.1, 0.2, 0.8, 0.9], "label": [0, 0, 1, 1]}
                ... )
                >>> pre = PlattCalibrator("score", "label").fit(train)
                >>> "calibrated" in pre.transform(bt.from_pydict({"score": [0.5]})).columns
                True

        Args:
            ds: The dataset whose scores to calibrate.

        Returns:
            A new lazy `Dataset` with the calibrated column appended.
        """
        self._require_fitted()
        linear = lit(self.intercept_) + lit(self.coef_) * col(self.score_column).cast("float64")
        probability = lit(1.0) / (lit(1.0) + (-linear).exp())
        return ds.with_columns(**{self.output_column: probability})
