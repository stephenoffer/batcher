"""Clustering contingency and pair-confusion tables.

Both are the objects the clustering scores are built from, so they are checked against
scikit-learn's `contingency_matrix` and `pair_confusion_matrix` cell for cell.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher.ml.metrics import contingency_matrix, pair_confusion_matrix

pytestmark = pytest.mark.unit

skc = pytest.importorskip("sklearn.metrics.cluster")


@pytest.fixture(scope="module")
def labelings() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(1)
    true = rng.integers(0, 4, 300)
    pred = rng.integers(0, 5, 300)
    ds = bt.from_pydict({"t": true.tolist(), "p": pred.tolist()})
    return true, pred, ds


def test_contingency_matrix_matches_sklearn(labelings) -> None:
    true, pred, ds = labelings
    got = contingency_matrix(ds, "t", "p").to_pydict()
    mine = np.array([[got[str(j)][i] for j in sorted(set(pred))] for i in range(len(set(true)))])
    assert np.array_equal(mine, skc.contingency_matrix(true, pred))


def test_contingency_matrix_is_labeled_and_sorted() -> None:
    ds = bt.from_pydict({"t": [0, 0, 1, 1], "p": [0, 0, 0, 1]})
    got = contingency_matrix(ds, "t", "p").to_pydict()
    assert got["t"] == [0, 1]
    assert got["0"] == [2, 1]
    assert got["1"] == [0, 1]


def test_pair_confusion_matches_sklearn(labelings) -> None:
    true, pred, ds = labelings
    got = pair_confusion_matrix(ds, "t", "p")
    ref = skc.pair_confusion_matrix(true, pred)
    assert got["different_different"] == ref[0][0]
    assert got["same_different"] == ref[0][1]
    assert got["different_same"] == ref[1][0]
    assert got["same_same"] == ref[1][1]


def test_pair_confusion_is_pure_for_a_perfect_clustering() -> None:
    # Four points, two pure pairs: all agreeing-together pairs, no disagreements.
    ds = bt.from_pydict({"t": [0, 0, 1, 1], "p": [0, 0, 1, 1]})
    got = pair_confusion_matrix(ds, "t", "p")
    assert got["same_different"] == 0
    assert got["different_same"] == 0
    assert got["same_same"] == 4
