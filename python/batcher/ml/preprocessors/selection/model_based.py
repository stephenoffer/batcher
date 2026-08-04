"""`SelectFromModel` and `RFE` — feature selection that reads a fitted model.

A univariate filter scores each feature alone, which is cheap and blind: it cannot see that
two columns are the same column, or that a feature only matters in the presence of another.
Both selectors here fix that by asking a model which features it actually used.

They differ in how much they are willing to spend. `SelectFromModel` fits once and reads the
coefficients. `RFE` refits after every elimination, which is the more faithful answer —
dropping a feature changes what the survivors are worth — and costs one fit per round.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from batcher.api.dataset import Dataset

__all__ = ["RFE", "SelectFromModel", "feature_importances"]


def feature_importances(estimator: Any) -> dict[str, float]:
    """Read a fitted estimator's per-feature importance as ``{feature: magnitude}``.

    Understands Batcher's own linear estimators (`features` plus `coef_`), scikit-learn's
    ``feature_importances_`` and ``coef_``, and anything already shaped like the dict this
    returns. A coefficient's *sign* says which way the feature pushes, not how much it
    matters, so magnitudes are returned; a multiclass ``coef_`` is reduced to the largest
    magnitude across classes, matching scikit-learn's default.

    Args:
        estimator: A fitted estimator, or a ``{feature: importance}`` mapping.

    Returns:
        One non-negative importance per feature name.

    Raises:
        PlanError: If the object exposes no importance the caller could have meant.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import LinearRegression
            >>> from batcher.ml.preprocessors import feature_importances
            >>> ds = bt.from_pydict({"a": [1.0, 2.0, 3.0], "b": [0.0, 0.0, 1.0],
            ...                      "y": [2.0, 4.0, 6.0]})
            >>> sorted(feature_importances(LinearRegression(["a", "b"], "y").fit(ds)))
            ['a', 'b']
    """
    if isinstance(estimator, dict):
        return {str(k): abs(float(v)) for k, v in estimator.items()}

    names = getattr(estimator, "features", None) or getattr(estimator, "feature_names_in_", None)
    weights = getattr(estimator, "feature_importances_", None)
    if weights is None:
        weights = getattr(estimator, "coef_", None)
    if names is None or weights is None:
        raise PlanError(
            f"{type(estimator).__name__} exposes no feature importances: expected `features` "
            "with `coef_` (a Batcher estimator), `feature_importances_`, or a "
            "{feature: importance} dict. Pass one of those, or a plain dict you built."
        )

    import numpy as np

    matrix = np.atleast_2d(np.asarray(weights, dtype=float))
    magnitudes = np.abs(matrix).max(axis=0)
    names = list(names)
    if len(names) != magnitudes.size:
        raise PlanError(
            f"{type(estimator).__name__} reports {len(names)} feature name(s) but "
            f"{magnitudes.size} importance(s). The estimator was fitted on a different "
            "feature set than it is reporting."
        )
    return {str(n): float(v) for n, v in zip(names, magnitudes, strict=True)}


class SelectFromModel(Preprocessor):
    """Keep the features a fitted model gave a large enough coefficient.

    One fit, then a threshold on the magnitudes. Paired with a
    {py:class}`Lasso <batcher.ml.Lasso>` this is the standard embedded-selection recipe: the
    L1 penalty drives useless coefficients to exactly zero, and the default ``threshold=0``
    keeps whatever survived.

    The estimator must already be fitted. That is deliberate — refitting it here would hide
    which data the selection saw, and the split it was fitted on is the thing that decides
    whether the selection leaks.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import Lasso
            >>> from batcher.ml.preprocessors import SelectFromModel
            >>> ds = bt.from_pydict({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 0.0, 1.0, 0.0],
            ...                      "y": [2.0, 4.0, 6.0, 8.0]})
            >>> model = Lasso(["a", "b"], "y", alpha=0.1).fit(ds)
            >>> SelectFromModel(model).fit(ds).selected_
            ['a']

    Args:
        estimator: A fitted estimator, or a ``{feature: importance}`` mapping.
        threshold: Keep a feature whose importance is strictly above this. A string
            ``"mean"`` or ``"median"`` uses that statistic of the importances instead.
        max_features: Also cap the number kept, highest importance first.
    """

    __slots__ = ("estimator", "importances_", "max_features", "selected_", "threshold")

    def __init__(
        self,
        estimator: Any,
        *,
        threshold: float | str = 0.0,
        max_features: int | None = None,
    ) -> None:
        if isinstance(threshold, str) and threshold not in ("mean", "median"):
            raise PlanError(
                f"SelectFromModel: threshold must be a number, 'mean', or 'median', "
                f"got {threshold!r}"
            )
        if max_features is not None and max_features < 1:
            raise PlanError(f"SelectFromModel: max_features must be at least 1, got {max_features}")
        self.estimator = estimator
        self.threshold = threshold
        self.max_features = max_features
        self.importances_: dict[str, float] = {}
        self.selected_: list[str] = []

    def fit(self, ds: Dataset) -> SelectFromModel:
        """Read the estimator's importances and apply the threshold.

        `ds` is not scored — the estimator carries the fit — so this runs no query. The
        argument is still taken because it is the `Preprocessor` contract, and because a
        `Chain` calls every step's `fit` with the same dataset.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import SelectFromModel
                >>> ds = bt.from_pydict({"a": [1.0], "b": [2.0]})
                >>> SelectFromModel({"a": 1.0, "b": 0.0}).fit(ds).selected_
                ['a']

        Args:
            ds: Unused; present for the `fit` contract.

        Returns:
            ``self``, fitted.
        """
        _ = ds
        self.importances_ = feature_importances(self.estimator)
        self.selected_ = sorted(
            _above_threshold(self.importances_, self.threshold, self.max_features)
        )
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Drop the features the model gave too small a coefficient, keeping everything else.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import SelectFromModel
                >>> ds = bt.from_pydict({"a": [1.0], "b": [2.0], "y": [3.0]})
                >>> SelectFromModel({"a": 1.0, "b": 0.0}).fit_transform(ds).columns
                ['a', 'y']

        Args:
            ds: The dataset to prune.

        Returns:
            A new lazy `Dataset` without the rejected feature columns.
        """
        self._require_fitted()
        rejected = {f for f in self.importances_ if f not in set(self.selected_)}
        return ds.select(*[c for c in ds.columns if c not in rejected])


def _above_threshold(
    importances: dict[str, float], threshold: float | str, max_features: int | None
) -> list[str]:
    """The feature names clearing `threshold`, capped at `max_features` by importance."""
    values = list(importances.values())
    if threshold == "mean":
        cut = sum(values) / len(values) if values else 0.0
    elif threshold == "median":
        ordered = sorted(values)
        middle = len(ordered) // 2
        if not ordered:
            cut = 0.0
        elif len(ordered) % 2:
            cut = ordered[middle]
        else:
            cut = (ordered[middle - 1] + ordered[middle]) / 2.0
    else:
        cut = float(threshold)
    kept = [name for name, value in importances.items() if value > cut]
    if max_features is not None and len(kept) > max_features:
        kept = sorted(kept, key=lambda n: importances[n], reverse=True)[:max_features]
    return kept


class RFE(Preprocessor):
    """Recursive feature elimination — refit, drop the weakest feature, repeat.

    The multivariate answer to a univariate filter. Dropping a feature changes what the
    remaining ones are worth, so ranking once and cutting is not the same as cutting one at
    a time; `RFE` pays for the difference with one model fit per elimination round.

    `fit_model` is a ``(dataset, features) -> estimator`` callable, which keeps this
    independent of any particular model: it works with a Batcher estimator, a scikit-learn
    one, or a closure that fits a whole pipeline, as long as what comes back exposes
    importances that `feature_importances` can read.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import LinearRegression
            >>> from batcher.ml.preprocessors import RFE
            >>> ds = bt.from_pydict(
            ...     {"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 0.0, 1.0, 0.0],
            ...      "c": [0.0, 0.0, 0.0, 1.0], "y": [2.0, 4.0, 6.0, 8.0]}
            ... )
            >>> rfe = RFE(lambda d, f: LinearRegression(list(f), "y").fit(d),
            ...           features=["a", "b", "c"], n_features=1)
            >>> rfe.fit(ds).selected_
            ['a']

    Args:
        fit_model: A ``(dataset, features) -> estimator`` callable, refit each round.
        features: The candidate feature columns.
        n_features: How many features to keep.
        step: How many features to drop per round; a float is read as a fraction of the
            features still in play.
    """

    __slots__ = ("features", "fit_model", "n_features", "ranking_", "selected_", "step")

    def __init__(
        self,
        fit_model: Callable[[Dataset, Sequence[str]], Any],
        *,
        features: Sequence[str],
        n_features: int = 1,
        step: int | float = 1,
    ) -> None:
        if not callable(fit_model):
            raise PlanError("RFE: fit_model must be a (dataset, features) -> estimator callable")
        self.fit_model = fit_model
        self.features = list(features)
        if not self.features:
            raise PlanError("RFE: features must name at least one column")
        if not 1 <= n_features <= len(self.features):
            raise PlanError(
                f"RFE: n_features must be between 1 and the {len(self.features)} candidate "
                f"feature(s), got {n_features}"
            )
        self.n_features = n_features
        if (isinstance(step, float) and not 0 < step < 1) or (
            not isinstance(step, float) and step < 1
        ):
            raise PlanError(
                f"RFE: step must be a positive int or a fraction in (0, 1), got {step!r}"
            )
        self.step = step
        self.selected_: list[str] = []
        self.ranking_: dict[str, int] = {}

    def _drop_count(self, remaining: int) -> int:
        """How many features this round eliminates, never overshooting `n_features`."""
        import math

        raw = math.floor(remaining * self.step) if isinstance(self.step, float) else self.step
        return max(1, min(int(raw), remaining - self.n_features))

    def fit(self, ds: Dataset) -> RFE:
        """Eliminate the weakest features one round at a time, refitting each round.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import RFE
                >>> ds = bt.from_pydict({"a": [1.0, 2.0], "b": [0.0, 0.0], "y": [1.0, 2.0]})
                >>> fit = lambda d, f: {name: 1.0 if name == "a" else 0.0 for name in f}
                >>> RFE(fit, features=["a", "b"], n_features=1).fit(ds).ranking_["a"]
                1

        Args:
            ds: The training split to refit on each round.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If a round's model reports no importance for a surviving feature.
        """
        remaining = list(self.features)
        # Rank 1 is "survived to the end"; a feature eliminated earlier gets a higher rank,
        # which is scikit-learn's convention and reads the right way round — rank 1 is best.
        ranking = dict.fromkeys(remaining, 1)
        while len(remaining) > self.n_features:
            importances = feature_importances(self.fit_model(ds, tuple(remaining)))
            missing = [f for f in remaining if f not in importances]
            if missing:
                raise PlanError(
                    f"RFE: the model fitted on {len(remaining)} feature(s) reported no "
                    f"importance for {missing}. fit_model must fit on exactly the features "
                    "it is given."
                )
            weakest = sorted(remaining, key=lambda f: (importances[f], f))
            for name in weakest[: self._drop_count(len(remaining))]:
                remaining.remove(name)
                ranking[name] = len(remaining) + 1
        self.selected_ = sorted(remaining)
        self.ranking_ = ranking
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Drop the eliminated features, keeping every other column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import RFE
                >>> ds = bt.from_pydict({"a": [1.0, 2.0], "b": [0.0, 0.0], "y": [1.0, 2.0]})
                >>> fit = lambda d, f: {name: 1.0 if name == "a" else 0.0 for name in f}
                >>> RFE(fit, features=["a", "b"], n_features=1).fit_transform(ds).columns
                ['a', 'y']

        Args:
            ds: The dataset to prune.

        Returns:
            A new lazy `Dataset` without the eliminated feature columns.
        """
        self._require_fitted()
        rejected = set(self.features) - set(self.selected_)
        return ds.select(*[c for c in ds.columns if c not in rejected])
