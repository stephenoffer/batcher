"""`TransformedTargetRegressor` — fit on a reshaped target, predict on the original scale.

Squared error assumes the target's noise is symmetric and roughly constant. A price, a
duration, a claim amount and a count all violate that: they are non-negative, right-skewed,
and their spread grows with their level, so a regression fitted directly on them spends its
capacity on the long tail and under-predicts the body.

The standard fix is to fit on ``log1p(y)`` and exponentiate the prediction back. Doing that
by hand is three steps in three places, and the third — remembering to invert at serving
time — is the one that gets forgotten, which produces predictions wrong by a factor of *e*
with nothing to indicate it.

This wraps the pair so the inverse cannot be forgotten: `fit` transforms the target, and
`predict` inverts the model's output before returning it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.stats._shared import require_columns
from batcher.plan.expr_ir import col

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["TRANSFORMS", "TransformedTargetRegressor"]

#: The named target transforms, each with its exact inverse. ``log1p``/``expm1`` is the pair
#: to reach for on a non-negative, right-skewed target: unlike a bare log it is defined at
#: zero, which is where a count or an amount most often sits.
TRANSFORMS = ("log1p", "log", "sqrt")

_INVERSE = {"log1p": "expm1", "log": "exp", "sqrt": "square"}


class TransformedTargetRegressor:
    """Fit a regressor on a transformed target and invert its predictions automatically.

    The prediction is returned on the *original* scale, so a metric computed against the
    untransformed truth means what it says. That is the property worth having: comparing a
    model fitted on ``log1p(y)`` against one fitted on ``y`` is otherwise comparing two
    different quantities and calling the smaller number better.

    Inverting a mean in log space returns a *median*-like estimate on the original scale,
    not a mean — the transform is non-linear, so the two do not coincide. That is the
    standard, accepted behaviour of this technique and usually what you want on a skewed
    target, but it is the reason the result is biased low if you need an expectation.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import LinearRegression, TransformedTargetRegressor
            >>> train = bt.from_pydict(
            ...     {"x": [1.0, 2.0, 3.0, 4.0], "y": [1.718282, 6.389056, 19.085537, 53.598150]}
            ... )
            >>> model = TransformedTargetRegressor(
            ...     LinearRegression(["x"], "y"), target="y", transform="log1p"
            ... ).fit(train)
            >>> [round(v, 2) for v in model.predict(train).to_pydict()["prediction"]]
            [1.72, 6.39, 19.09, 53.6]

    Args:
        model: The regressor to fit on the transformed target. Its own `target` must name
            the same column.
        target: The target column to transform.
        transform: One of `TRANSFORMS`.
        output_column: The column the wrapped model writes its prediction into, which is
            also where the inverted prediction is returned.
    """

    __slots__ = ("model", "output_column", "target", "transform")

    def __init__(
        self,
        model: Any,
        *,
        target: str,
        transform: str = "log1p",
        output_column: str = "prediction",
    ) -> None:
        if not callable(getattr(model, "fit", None)) or not callable(
            getattr(model, "predict", None)
        ):
            raise PlanError(
                f"TransformedTargetRegressor needs a model with fit() and predict(), got "
                f"{type(model).__name__}."
            )
        if transform not in TRANSFORMS:
            raise PlanError(
                f"TransformedTargetRegressor: transform must be one of "
                f"{', '.join(TRANSFORMS)}, got {transform!r}"
            )
        self.model = model
        self.target = target
        self.transform = transform
        self.output_column = output_column

    def _forward(self, ds: Dataset) -> Dataset:
        """Replace the target column with its transform."""
        expression = getattr(col(self.target).cast("float64"), self.transform)()
        return ds.with_columns(**{self.target: expression})

    def fit(self, ds: Dataset) -> TransformedTargetRegressor:
        """Transform the target, then fit the wrapped model on it.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import LinearRegression, TransformedTargetRegressor
                >>> train = bt.from_pydict({"x": [1.0, 2.0], "y": [1.718282, 6.389056]})
                >>> wrapped = TransformedTargetRegressor(
                ...     LinearRegression(["x"], "y"), target="y"
                ... ).fit(train)
                >>> wrapped.model.coef_[0] > 0
                True

        Args:
            ds: The training data, with the target on its original scale.

        Returns:
            ``self``, fitted.

        Raises:
            ColumnNotFoundError: If the target column is missing.
        """
        require_columns(ds, self.target, hint="Pass the target column.")
        self.model.fit(self._forward(ds))
        return self

    def predict(self, ds: Dataset) -> Dataset:
        """Score with the wrapped model, then invert its output to the original scale.

        The target column is *not* transformed here — only `fit` needs that — so a frame
        being scored does not have to carry the target at all.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import LinearRegression, TransformedTargetRegressor
                >>> train = bt.from_pydict({"x": [1.0, 2.0], "y": [1.718282, 6.389056]})
                >>> wrapped = TransformedTargetRegressor(
                ...     LinearRegression(["x"], "y"), target="y"
                ... ).fit(train)
                >>> scored = wrapped.predict(bt.from_pydict({"x": [1.0]}))
                >>> round(scored.to_pydict()["prediction"][0], 3)
                1.718

        Args:
            ds: The dataset to score.

        Returns:
            A new lazy `Dataset` whose prediction column is on the original scale.
        """
        scored = self.model.predict(ds)
        inverse = getattr(col(self.output_column), _INVERSE[self.transform])()
        return scored.with_columns(**{self.output_column: inverse})
