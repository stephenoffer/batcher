"""Internal clustering-quality scores (Calinski-Harabasz and Davies-Bouldin).

Both are built from per-cluster centroids and dispersions, so they are checked against
scikit-learn on a well-separated clustering and a random one, and pinned to the direction each
moves: a good clustering scores high on Calinski-Harabasz and low on Davies-Bouldin.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher.ml.metrics import calinski_harabasz_score, davies_bouldin_score

pytestmark = pytest.mark.unit

skm = pytest.importorskip("sklearn.metrics")


@pytest.fixture(scope="module")
def blobs() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    x = np.vstack(
        [
            rng.normal([0, 0], 1, (100, 2)),
            rng.normal([5, 5], 1, (100, 2)),
            rng.normal([0, 6], 1, (100, 2)),
        ]
    )
    labels = np.repeat([0, 1, 2], 100)
    ds = bt.from_pydict({"a": x[:, 0].tolist(), "b": x[:, 1].tolist(), "c": labels.tolist()})
    return x, labels, ds


def test_calinski_harabasz_matches_sklearn(blobs) -> None:
    x, labels, ds = blobs
    assert calinski_harabasz_score(ds, ["a", "b"], "c") == pytest.approx(
        skm.calinski_harabasz_score(x, labels), rel=1e-6
    )


def test_davies_bouldin_matches_sklearn(blobs) -> None:
    x, labels, ds = blobs
    assert davies_bouldin_score(ds, ["a", "b"], "c") == pytest.approx(
        skm.davies_bouldin_score(x, labels), abs=1e-9
    )


def test_both_match_sklearn_on_a_random_labeling(blobs) -> None:
    x, _, _ = blobs
    rng = np.random.default_rng(9)
    random_labels = rng.integers(0, 3, len(x))
    ds = bt.from_pydict({"a": x[:, 0].tolist(), "b": x[:, 1].tolist(), "c": random_labels.tolist()})
    assert calinski_harabasz_score(ds, ["a", "b"], "c") == pytest.approx(
        skm.calinski_harabasz_score(x, random_labels), rel=1e-6
    )
    assert davies_bouldin_score(ds, ["a", "b"], "c") == pytest.approx(
        skm.davies_bouldin_score(x, random_labels), abs=1e-9
    )


def test_good_clustering_beats_a_random_one(blobs) -> None:
    x, _, ds = blobs
    rng = np.random.default_rng(3)
    random_labels = rng.integers(0, 3, len(x))
    random_ds = bt.from_pydict(
        {"a": x[:, 0].tolist(), "b": x[:, 1].tolist(), "c": random_labels.tolist()}
    )
    assert calinski_harabasz_score(ds, ["a", "b"], "c") > calinski_harabasz_score(
        random_ds, ["a", "b"], "c"
    )
    assert davies_bouldin_score(ds, ["a", "b"], "c") < davies_bouldin_score(
        random_ds, ["a", "b"], "c"
    )
