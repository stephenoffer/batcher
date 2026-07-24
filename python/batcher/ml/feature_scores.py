"""Univariate feature scoring — rank every feature against the target in one pass each.

Before fitting a model on a wide table, the cheapest useful thing to know is which columns
carry signal about the target on their own. That is a univariate score per feature, and it is
the filter half of feature selection (scikit-learn's ``SelectKBest`` with ``f_classif`` /
``f_regression`` / ``mutual_info``). Each score here reuses a statistic that is already a
mergeable aggregate, so scoring a hundred features is a hundred one-pass reductions, not a
materialized correlation matrix.

The scorers return a plain ``{feature: score}`` dict, and `select_k_best` turns that into the
surviving column names. A score is univariate by construction, so it sees a feature that
matters only in combination with another as noise; use it to prune obvious dead weight, not as
the final word on a feature set.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from batcher.ml.stats.association import anova_f, chi_square, mutual_information
from batcher.plan.expr_ir.constructors import col
from batcher.plan.functions.aggregate import corr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = [
    "chi2_scores",
    "f_classif_scores",
    "f_regression_scores",
    "mutual_info_scores",
    "select_k_best",
]


def _feature_list(ds: Dataset, features: Sequence[str] | None, target: str) -> list[str]:
    """Resolve the feature list, defaulting to every column but the target."""
    if features is not None:
        return list(features)
    return [c for c in ds.columns if c != target]


def f_classif_scores(
    ds: Dataset, target: str, features: Sequence[str] | None = None
) -> dict[str, float]:
    """Score each numeric feature against a categorical target by its ANOVA F value.

    The classification counterpart of scikit-learn's ``f_classif``: a large F means the
    feature's mean differs sharply across the target's classes, so the feature separates them.
    Each score is one `anova_f` aggregate over the feature grouped by the target.

    Args:
        ds: The dataset to score.
        target: The categorical target column.
        features: The numeric feature columns to score; defaults to every column but the target.

    Returns:
        A ``{feature: f_value}`` dict.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.feature_scores import f_classif_scores
            >>> ds = bt.from_pydict(
            ...     {"y": ["a", "a", "b", "b"], "signal": [1.0, 1.1, 9.0, 9.2],
            ...      "noise": [5.0, 1.0, 5.0, 1.0]}
            ... )
            >>> scores = f_classif_scores(ds, "y")
            >>> scores["signal"] > scores["noise"]
            True
    """
    features = _feature_list(ds, features, target)
    return {f: anova_f(ds, f, target) for f in features}


def f_regression_scores(
    ds: Dataset, target: str, features: Sequence[str] | None = None
) -> dict[str, float]:
    """Score each numeric feature against a continuous target by its regression F value.

    The regression counterpart of scikit-learn's ``f_regression``: the F statistic of the
    univariate linear fit, ``F = r^2 / (1 - r^2) * (n - 2)``, monotone in the squared
    correlation. Each score is one `corr` aggregate plus a row count.

    Args:
        ds: The dataset to score.
        target: The continuous target column.
        features: The numeric feature columns to score; defaults to every column but the target.

    Returns:
        A ``{feature: f_value}`` dict.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.feature_scores import f_regression_scores
            >>> ds = bt.from_pydict(
            ...     {"y": [1.0, 2.0, 3.0, 4.0], "linear": [1.0, 2.0, 3.0, 4.0],
            ...      "flat": [1.0, 1.0, 1.0, 2.0]}
            ... )
            >>> scores = f_regression_scores(ds, "y")
            >>> scores["linear"] > scores["flat"]
            True
    """
    features = _feature_list(ds, features, target)
    n = ds.count()
    out: dict[str, float] = {}
    for f in features:
        r = ds.agg(r=corr(col(f), col(target))).collect().column("r")[0].as_py()
        if r is None or abs(r) >= 1.0:
            out[f] = math.inf if r is not None else math.nan
            continue
        r2 = float(r) ** 2
        out[f] = r2 / (1.0 - r2) * (n - 2)
    return out


def chi2_scores(
    ds: Dataset, target: str, features: Sequence[str] | None = None
) -> dict[str, float]:
    """Score each categorical feature against a categorical target by its chi-squared statistic.

    The categorical counterpart of scikit-learn's ``chi2``: a large statistic means the
    feature and target are far from independent. Each score is one `chi_square` over the
    feature-target contingency table.

    Args:
        ds: The dataset to score.
        target: The categorical target column.
        features: The categorical feature columns to score; defaults to every column but the
            target.

    Returns:
        A ``{feature: chi_squared}`` dict.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.feature_scores import chi2_scores
            >>> ds = bt.from_pydict(
            ...     {"y": ["p", "p", "q", "q"], "linked": ["a", "a", "b", "b"],
            ...      "free": ["a", "b", "a", "b"]}
            ... )
            >>> scores = chi2_scores(ds, "y")
            >>> scores["linked"] > scores["free"]
            True
    """
    features = _feature_list(ds, features, target)
    return {f: chi_square(ds, f, target) for f in features}


def mutual_info_scores(
    ds: Dataset, target: str, features: Sequence[str] | None = None, *, base: float = 2.0
) -> dict[str, float]:
    """Score each feature against the target by mutual information.

    The information-theoretic score: how many bits knowing the feature saves about the target.
    Unlike the F scores it catches a non-monotone association a correlation misses. Continuous
    columns are binned by `mutual_information`, so discretize a continuous feature first if the
    default binning is too coarse.

    Args:
        ds: The dataset to score.
        target: The target column.
        features: The feature columns to score; defaults to every column but the target.
        base: The logarithm base, so 2 gives bits and ``math.e`` gives nats.

    Returns:
        A ``{feature: mutual_information}`` dict.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.feature_scores import mutual_info_scores
            >>> ds = bt.from_pydict(
            ...     {"y": ["a", "a", "b", "b"], "copy": ["a", "a", "b", "b"],
            ...      "rand": ["a", "b", "a", "b"]}
            ... )
            >>> scores = mutual_info_scores(ds, "y")
            >>> scores["copy"] > scores["rand"]
            True
    """
    features = _feature_list(ds, features, target)
    return {f: mutual_information(ds, f, target, base=base) for f in features}


def select_k_best(scores: dict[str, float], k: int) -> list[str]:
    """The `k` highest-scoring feature names, best first.

    The selection step after a scorer: turn a ``{feature: score}`` dict into the columns to
    keep. Features scoring NaN (a degenerate fit) sort last and are dropped before any real
    score.

    Args:
        scores: A ``{feature: score}`` mapping from one of the scorers.
        k: How many features to keep.

    Returns:
        The `k` feature names with the largest scores, highest first.

    Examples:
        .. doctest::

            >>> from batcher.ml.feature_scores import select_k_best
            >>> select_k_best({"a": 10.0, "b": 1.0, "c": 5.0}, 2)
            ['a', 'c']
    """
    ranked = sorted(
        scores.items(),
        key=lambda kv: kv[1] if not math.isnan(kv[1]) else -math.inf,
        reverse=True,
    )
    return [name for name, _ in ranked[:k]]
