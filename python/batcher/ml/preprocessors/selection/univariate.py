"""`SelectKBest` and `SelectPercentile` — keep the features that score against the target.

`batcher.ml.feature_scores` already computes the scores; what was missing is the half that
makes them usable in a pipeline. A score dict tells you which columns look promising, and
then the caller has to write the `drop` themselves, remember which columns were chosen, and
apply the *same* choice to the validation split — which is exactly the bookkeeping the
`Preprocessor` contract exists to hold.

Selecting is fitted state, not a property of a dataset. Choose features from the validation
split and the selection has already seen the answer, so the score you measure afterwards is
optimistic by an amount nothing reports. Fitting on train and transforming both splits is
the whole point of these being objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["SCORE_FUNCTIONS", "SelectKBest", "SelectPercentile"]

#: The univariate scorers a selector accepts, under scikit-learn's names. Each is a
#: ``(dataset, target, features) -> {feature: score}`` callable in `ml.feature_scores`, and
#: a caller may pass any callable of that shape instead of one of these names.
SCORE_FUNCTIONS = ("f_classif", "f_regression", "chi2", "mutual_info")


def _scorer(score_func: str | Any, *, what: str) -> Any:
    """Resolve a scorer name to the callable in `ml.feature_scores`, or pass a callable through."""
    if callable(score_func):
        return score_func
    from batcher.ml import feature_scores

    if score_func not in SCORE_FUNCTIONS:
        raise PlanError(
            f"{what}: score_func must be one of {', '.join(SCORE_FUNCTIONS)}, or a callable "
            f"taking (dataset, target, features), got {score_func!r}"
        )
    return getattr(feature_scores, f"{score_func}_scores")


class _UnivariateSelector(Preprocessor):
    """The fit/transform machinery `SelectKBest` and `SelectPercentile` share.

    Only the rule turning ranked scores into a keep-count differs between the two, so that
    is the one method a subclass supplies.
    """

    __slots__ = ("features", "score_func", "scores_", "selected_", "target")

    def __init__(
        self,
        target: str,
        *,
        score_func: str | Any = "f_classif",
        features: Sequence[str] | None = None,
    ) -> None:
        what = type(self).__name__
        if not isinstance(target, str):
            raise PlanError(f"{what} takes a single target column name, got {target!r}")
        self.target = target
        self.score_func = score_func
        self.features = list(features) if features is not None else None
        self.scores_: dict[str, float] = {}
        self.selected_: list[str] = []

    def _keep_count(self, ranked: list[str]) -> int:
        """How many of the ranked features to keep."""
        raise NotImplementedError

    def fit(self, ds: Dataset) -> _UnivariateSelector:
        """Score every candidate feature against the target and record the survivors.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import SelectKBest
                >>> ds = bt.from_pydict(
                ...     {"y": ["a", "a", "b", "b"], "signal": [1.0, 1.1, 9.0, 9.2],
                ...      "noise": [5.0, 1.0, 5.0, 1.0]}
                ... )
                >>> SelectKBest("y", k=1).fit(ds).selected_
                ['signal']

        Args:
            ds: The training split to score against.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If the target or a named feature is not a column of `ds`.
        """
        from batcher.ml.feature_scores import select_k_best

        available = set(ds.columns)
        if self.target not in available:
            raise PlanError(
                f"{type(self).__name__}: target {self.target!r} is not a column of this dataset"
            )
        candidates = self.features if self.features is not None else None
        if candidates is not None:
            missing = [c for c in candidates if c not in available]
            if missing:
                raise PlanError(
                    f"{type(self).__name__}: no such column(s) {missing}. "
                    "Pass feature names that exist, or leave features unset to score them all."
                )
        scorer = _scorer(self.score_func, what=type(self).__name__)
        self.scores_ = dict(scorer(ds, self.target, candidates))
        ranked = select_k_best(self.scores_, len(self.scores_))
        self.selected_ = sorted(ranked[: self._keep_count(ranked)])
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Drop the features that did not survive `fit`, keeping everything else.

        Columns that were never candidates — the target, an id, anything `features`
        excluded — are kept. Only a scored-and-rejected feature is dropped, so a selector
        composes into the middle of a pipeline without taking the label with it.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import SelectKBest
                >>> ds = bt.from_pydict(
                ...     {"y": ["a", "a", "b", "b"], "signal": [1.0, 1.1, 9.0, 9.2],
                ...      "noise": [5.0, 1.0, 5.0, 1.0]}
                ... )
                >>> SelectKBest("y", k=1).fit_transform(ds).columns
                ['y', 'signal']

        Args:
            ds: The dataset to prune.

        Returns:
            A new lazy `Dataset` without the rejected feature columns.
        """
        self._require_fitted()
        rejected = {f for f in self.scores_ if f not in set(self.selected_)}
        keep = [c for c in ds.columns if c not in rejected]
        return ds.select(*keep)


class SelectKBest(_UnivariateSelector):
    """Keep the `k` features scoring highest against the target.

    The filter step of feature selection, and the one worth running first on a wide table:
    it costs one mergeable aggregate per feature and removes the columns that carry no
    signal on their own before anything expensive sees them.

    A univariate score is blind to interaction by construction — a feature that only matters
    alongside another scores as noise — so use this to prune obvious dead weight, not to
    decide a final feature set. {py:class}`RFE <batcher.ml.preprocessors.RFE>` is the
    multivariate answer when you can afford the refits.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import SelectKBest
            >>> ds = bt.from_pydict(
            ...     {"y": [1.0, 2.0, 3.0, 4.0], "linear": [1.0, 2.0, 3.0, 4.0],
            ...      "flat": [1.0, 1.0, 1.0, 2.0]}
            ... )
            >>> SelectKBest("y", k=1, score_func="f_regression").fit(ds).selected_
            ['linear']

    Args:
        target: The target column to score against.
        k: How many features to keep.
        score_func: One of `SCORE_FUNCTIONS`, or a ``(dataset, target, features) ->
            {feature: score}`` callable.
        features: The candidate feature columns; defaults to every column but the target.
    """

    __slots__ = ("k",)

    def __init__(
        self,
        target: str,
        *,
        k: int = 10,
        score_func: str | Any = "f_classif",
        features: Sequence[str] | None = None,
    ) -> None:
        super().__init__(target, score_func=score_func, features=features)
        if k < 1:
            raise PlanError(f"SelectKBest: k must be at least 1, got {k}")
        self.k = k

    def _keep_count(self, ranked: list[str]) -> int:
        """Keep `k` features, or all of them when there are fewer than `k`."""
        return min(self.k, len(ranked))


class SelectPercentile(_UnivariateSelector):
    """Keep the highest-scoring `percentile` percent of the features.

    The same filter as {py:class}`SelectKBest <batcher.ml.preprocessors.SelectKBest>`, sized
    as a fraction rather than a count. Prefer it when the feature count varies between runs
    — a fixed `k` that suited a fifty-column table keeps almost nothing from a five-hundred
    column one.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import SelectPercentile
            >>> ds = bt.from_pydict(
            ...     {"y": ["a", "a", "b", "b"], "signal": [1.0, 1.1, 9.0, 9.2],
            ...      "noise": [5.0, 1.0, 5.0, 1.0]}
            ... )
            >>> SelectPercentile("y", percentile=50).fit(ds).selected_
            ['signal']

    Args:
        target: The target column to score against.
        percentile: The percentage of features to keep, in ``(0, 100]``.
        score_func: One of `SCORE_FUNCTIONS`, or a ``(dataset, target, features) ->
            {feature: score}`` callable.
        features: The candidate feature columns; defaults to every column but the target.
    """

    __slots__ = ("percentile",)

    def __init__(
        self,
        target: str,
        *,
        percentile: float = 10.0,
        score_func: str | Any = "f_classif",
        features: Sequence[str] | None = None,
    ) -> None:
        super().__init__(target, score_func=score_func, features=features)
        if not 0 < percentile <= 100:
            raise PlanError(f"SelectPercentile: percentile must be in (0, 100], got {percentile!r}")
        self.percentile = percentile

    def _keep_count(self, ranked: list[str]) -> int:
        """Keep the requested share, rounded down, but never fewer than one feature."""
        import math

        if not ranked:
            return 0
        return max(1, math.floor(len(ranked) * self.percentile / 100.0))
