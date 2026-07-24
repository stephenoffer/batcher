"""Clustering-quality metrics against a reference labeling.

Every one is a closed-form function of the two labelings' contingency table, so the test is an
exact match with scikit-learn on random and structured labelings, plus the boundary cases: a
perfect (permuted) clustering scores 1 and each metric's defining asymmetry holds.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher.ml.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    fowlkes_mallows_score,
    homogeneity_score,
    mutual_info_score,
    normalized_mutual_info_score,
    rand_score,
    v_measure_score,
)

pytestmark = pytest.mark.unit

skm = pytest.importorskip("sklearn.metrics")

_METRICS = [
    (adjusted_rand_score, skm.adjusted_rand_score),
    (adjusted_mutual_info_score, skm.adjusted_mutual_info_score),
    (rand_score, skm.rand_score),
    (mutual_info_score, skm.mutual_info_score),
    (normalized_mutual_info_score, skm.normalized_mutual_info_score),
    (homogeneity_score, skm.homogeneity_score),
    (completeness_score, skm.completeness_score),
    (v_measure_score, skm.v_measure_score),
    (fowlkes_mallows_score, skm.fowlkes_mallows_score),
]


@pytest.fixture(scope="module")
def labelings() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(1)
    true = rng.integers(0, 4, 600)
    pred = (true + rng.integers(0, 2, 600)) % 5
    ds = bt.from_pydict({"t": true.tolist(), "p": pred.tolist()})
    return true, pred, ds


@pytest.mark.parametrize(("ours", "sklearn"), _METRICS)
def test_matches_sklearn(labelings, ours, sklearn) -> None:
    true, pred, ds = labelings
    assert ours(ds, "t", "p") == pytest.approx(sklearn(true, pred), abs=1e-9)


@pytest.mark.parametrize(("ours", "sklearn"), _METRICS)
def test_matches_sklearn_on_random_labelings(ours, sklearn) -> None:
    rng = np.random.default_rng(7)
    true = rng.integers(0, 6, 800)
    pred = rng.integers(0, 8, 800)
    ds = bt.from_pydict({"t": true.tolist(), "p": pred.tolist()})
    assert ours(ds, "t", "p") == pytest.approx(sklearn(true, pred), abs=1e-9)


# The normalized metrics reach 1 for a perfect clustering; the raw `mutual_info_score` reaches
# the labeling's entropy instead, so it is excluded from the score-one check.
_NORMALIZED = [pair for pair in _METRICS if pair[0] is not mutual_info_score]


@pytest.mark.parametrize(("ours", "_sklearn"), _NORMALIZED)
def test_perfect_clustering_scores_one(ours, _sklearn) -> None:
    # Cluster ids permuted from the truth: a perfect clustering, so every normalized metric is 1.
    ds = bt.from_pydict({"t": [0, 0, 1, 1, 2, 2], "p": [7, 7, 3, 3, 9, 9]})
    assert ours(ds, "t", "p") == pytest.approx(1.0)


def test_mutual_info_reaches_the_entropy_for_a_perfect_clustering() -> None:
    import math

    # Three balanced classes recovered perfectly: MI equals the label entropy, log(3).
    ds = bt.from_pydict({"t": [0, 0, 1, 1, 2, 2], "p": [7, 7, 3, 3, 9, 9]})
    assert mutual_info_score(ds, "t", "p") == pytest.approx(math.log(3))


def test_homogeneity_and_completeness_swap_under_transpose(labelings) -> None:
    true, pred, ds = labelings
    swapped = bt.from_pydict({"t": pred.tolist(), "p": true.tolist()})
    assert homogeneity_score(ds, "t", "p") == pytest.approx(completeness_score(swapped, "t", "p"))


def test_splitting_a_class_hurts_completeness_not_homogeneity() -> None:
    # Two true classes, but class 0 is split into two pure clusters: homogeneous, not complete.
    ds = bt.from_pydict({"t": [0, 0, 0, 0, 1, 1], "p": [0, 0, 1, 1, 2, 2]})
    assert homogeneity_score(ds, "t", "p") == pytest.approx(1.0)
    assert completeness_score(ds, "t", "p") < 1.0
