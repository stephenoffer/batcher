"""Fitting one model per target, so a multi-target problem stays one object.

Plenty of real problems predict several things at once from the same features: demand in
each region, a sensor reading per axis, or a set of independent yes/no tags on the same
document. Every estimator here takes exactly one target column, so the caller is left
holding a list of models, a list of the columns each was fitted on, and the job of keeping
the two aligned - which is the same train/serve skew `Pipeline` exists to prevent, one level
up.

`MultiOutputRegressor` and `MultiOutputClassifier` own that list. One `fit`, one `predict`
appending one column per target, and one saved document.

What they do not do is make the fit cheaper. Each target is an independent model over the
same features, so the cost is one fit per target and this is a wrapper rather than an
optimization. Where a genuinely shared fit is possible it is worth preferring: `RidgeCV`
searches a whole penalty path in one pass because ridge's moments do not depend on the
penalty, whereas two different targets genuinely need two different fits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml._estimator import require_fitted
from batcher.ml.stats._shared import require_columns

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["MAX_TARGETS", "MultiOutputClassifier", "MultiOutputRegressor"]

#: The ceiling on how many targets a multi-output model will fit.
#:
#: One model is fitted per target, so a `targets` list pointed at the wrong thing - every
#: column of a wide table, say - would fit hundreds of models and look like a hang. The bound
#: turns that into an immediate error naming the count.
MAX_TARGETS = 100


class MultiOutputRegressor:
    """Fit one model per target column and predict them all in a single pass.

    The estimator is passed as a *class* rather than an instance, because each sub-model needs
    a different target column and a Batcher estimator binds its columns at construction. This
    is the same shape `OneVsRestClassifier` uses.

    Prediction appends one column per target, named ``{output_prefix}_{target}``, and is a
    single projection: every sub-model's expression is added to the same frame, so scoring ten
    targets reads the data once rather than ten times.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import LinearRegression, MultiOutputRegressor
            >>> ds = bt.from_pydict(
            ...     {"x": [1.0, 2.0, 3.0, 4.0],
            ...      "north": [2.0, 4.0, 6.0, 8.0],
            ...      "south": [10.0, 20.0, 30.0, 40.0]}
            ... )
            >>> model = MultiOutputRegressor(
            ...     LinearRegression, ["x"], ["north", "south"]
            ... ).fit(ds)
            >>> scored = model.predict(ds).to_pydict()
            >>> round(scored["prediction_north"][0], 6), round(scored["prediction_south"][0], 6)
            (2.0, 10.0)

    Args:
        estimator: The single-target estimator class to fit once per target.
        features: The predictor columns, shared by every target.
        targets: The columns to predict, one model each.
        params: Hyperparameters passed to every sub-model. The sub-model's `output_column` is
            managed here and must not be set.
        output_prefix: The stem of the appended prediction columns.
        max_targets: The ceiling on how many targets to fit, guarding a mis-aimed list.

    Raises:
        PlanError: If `features` or `targets` is empty, if `targets` repeats a column, or if
            there are more targets than `max_targets`.
    """

    __slots__ = (
        "estimator",
        "estimators_",
        "features",
        "max_targets",
        "output_prefix",
        "params",
        "targets",
    )

    def __init__(
        self,
        estimator: type,
        features: Sequence[str],
        targets: Sequence[str],
        *,
        params: dict[str, Any] | None = None,
        output_prefix: str = "prediction",
        max_targets: int = MAX_TARGETS,
    ) -> None:
        self.features = list(features)
        self.targets = list(targets)
        what = type(self).__name__
        if not self.features:
            raise PlanError(f"{what} needs at least one feature column.")
        if not self.targets:
            raise PlanError(f"{what} needs at least one target column.")
        if len(set(self.targets)) != len(self.targets):
            raise PlanError(
                f"{what}: targets repeats a column ({self.targets}). Each target is fitted "
                "once, and a duplicate would silently overwrite the first model's output "
                "column with the second's."
            )
        if len(self.targets) > max_targets:
            raise PlanError(
                f"{what} fits one model per target, so {len(self.targets)} targets would fit "
                f"that many models. The ceiling is {max_targets}; if this really is the "
                "target list, raise max_targets."
            )
        overlap = sorted(set(self.features) & set(self.targets))
        if overlap:
            raise PlanError(
                f"{what}: {overlap[0]!r} is both a feature and a target, so that model would "
                "predict a column from itself and score perfectly for no reason."
            )
        self.estimator = estimator
        self.params = dict(params or {})
        self.output_prefix = output_prefix
        self.max_targets = max_targets
        self.estimators_: list[Any] = []

    def _column(self, target: str) -> str:
        """The output column for one target."""
        return f"{self.output_prefix}_{target}"

    def fit(self, ds: Dataset) -> MultiOutputRegressor:
        """Fit one sub-model per target.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import LinearRegression, MultiOutputRegressor
                >>> ds = bt.from_pydict(
                ...     {"x": [1.0, 2.0, 3.0], "a": [1.0, 2.0, 3.0], "b": [3.0, 6.0, 9.0]}
                ... )
                >>> model = MultiOutputRegressor(LinearRegression, ["x"], ["a", "b"]).fit(ds)
                >>> len(model.estimators_)
                2

        Args:
            ds: The training dataset.

        Returns:
            ``self``, fitted.

        Raises:
            ColumnNotFoundError: If a named column is missing.
        """
        require_columns(ds, *self.features, *self.targets)
        self.estimators_ = [
            self.estimator(
                self.features, target, output_column=self._column(target), **self.params
            ).fit(ds)
            for target in self.targets
        ]
        return self

    def predict(self, ds: Dataset) -> Dataset:
        """Append one prediction column per target.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import LinearRegression, MultiOutputRegressor
                >>> ds = bt.from_pydict(
                ...     {"x": [1.0, 2.0, 3.0], "a": [1.0, 2.0, 3.0], "b": [3.0, 6.0, 9.0]}
                ... )
                >>> model = MultiOutputRegressor(LinearRegression, ["x"], ["a", "b"]).fit(ds)
                >>> sorted(c for c in model.predict(ds).columns if c.startswith("prediction"))
                ['prediction_a', 'prediction_b']

        Args:
            ds: The dataset to score.

        Returns:
            A new lazy `Dataset` with one prediction column per target appended.
        """
        require_fitted(self, self.estimators_)
        scored = ds
        for model in self.estimators_:
            scored = model.predict(scored)
        return scored


class MultiOutputClassifier(MultiOutputRegressor):
    """Fit one classifier per label column, for a multi-label problem.

    The multi-label case, which is different from the multi-class one and often confused with
    it. Multi-class picks exactly one of several classes, and
    {py:class}`OneVsRestClassifier <batcher.ml.multiclass.OneVsRestClassifier>` handles it by
    taking the highest-scoring class. Multi-label lets a row carry any number of independent
    tags at once - a document is both "finance" and "urgent" - so each label is its own
    yes-or-no question with its own model and no argmax over them.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import LogisticRegression, MultiOutputClassifier
            >>> ds = bt.from_pydict(
            ...     {"length": [10.0, 12.0, 90.0, 95.0],
            ...      "urgent": [1, 1, 0, 0],
            ...      "long": [0, 0, 1, 1]}
            ... )
            >>> model = MultiOutputClassifier(
            ...     LogisticRegression, ["length"], ["urgent", "long"]
            ... ).fit(ds)
            >>> scored = model.predict(ds).to_pydict()
            >>> scored["prediction_urgent"][0] > 0.5, scored["prediction_long"][0] > 0.5
            (True, False)

    Args:
        estimator: The single-label classifier class to fit once per label.
        features: The predictor columns, shared by every label.
        targets: The label columns, one model each.
        params: Hyperparameters passed to every sub-model.
        output_prefix: The stem of the appended prediction columns.
        max_targets: The ceiling on how many labels to fit.
    """

    __slots__ = ()
