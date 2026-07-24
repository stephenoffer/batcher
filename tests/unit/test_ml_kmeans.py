"""K-means clustering.

K-means has random initialization and local optima, so the tests do not demand a byte-exact
match with scikit-learn. They pin what must hold: on separable blobs the clustering agrees with
both the ground truth and scikit-learn up to label permutation (adjusted Rand index of 1), the
inertia matches, the fit is reproducible from its seed, and serving reuses the trained centroids.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.cluster import KMeans

pytestmark = pytest.mark.unit

sk_cluster = pytest.importorskip("sklearn.cluster")
sk_metrics = pytest.importorskip("sklearn.metrics")


@pytest.fixture(scope="module")
def blobs() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 10], [0, 10], [10, 0]])
    x = np.vstack([c + rng.normal(0, 0.6, (150, 2)) for c in centers])
    truth = np.repeat(range(4), 150)
    ds = bt.from_pydict({"x": x[:, 0].tolist(), "y": x[:, 1].tolist()})
    return x, truth, ds


def test_recovers_the_true_blobs(blobs) -> None:
    _, truth, ds = blobs
    labels = KMeans(["x", "y"], n_clusters=4, seed=3).fit_predict(ds).to_pydict()["cluster"]
    assert sk_metrics.adjusted_rand_score(truth, labels) == pytest.approx(1.0)


def test_agrees_with_sklearn_up_to_permutation(blobs) -> None:
    x, _, ds = blobs
    labels = KMeans(["x", "y"], n_clusters=4, seed=3).fit(ds).predict(ds).to_pydict()["cluster"]
    sk = sk_cluster.KMeans(n_clusters=4, n_init=10, random_state=0).fit(x)
    assert sk_metrics.adjusted_rand_score(sk.labels_, labels) == pytest.approx(1.0)


def test_inertia_matches_sklearn(blobs) -> None:
    x, _, ds = blobs
    km = KMeans(["x", "y"], n_clusters=4, seed=3).fit(ds)
    sk = sk_cluster.KMeans(n_clusters=4, n_init=10, random_state=0).fit(x)
    assert km.inertia_ == pytest.approx(sk.inertia_, rel=0.02)


def test_is_reproducible_from_the_seed(blobs) -> None:
    _, _, ds = blobs
    a = KMeans(["x", "y"], n_clusters=4, seed=5).fit(ds).centroids_
    b = KMeans(["x", "y"], n_clusters=4, seed=5).fit(ds).centroids_
    assert a == b


def test_predict_reuses_the_trained_centroids(blobs) -> None:
    _, _, ds = blobs
    km = KMeans(["x", "y"], n_clusters=4, seed=3).fit(ds)
    serve = bt.from_pydict({"x": [0.0, 10.0], "y": [0.0, 10.0]})
    labels = km.predict(serve).to_pydict()["cluster"]
    # The two serving points sit on opposite corners, so they land in different clusters.
    assert labels[0] != labels[1]


def test_rejects_a_bad_cluster_count() -> None:
    with pytest.raises(PlanError, match="n_clusters must be"):
        KMeans(["x"], n_clusters=0)


def test_rejects_more_clusters_than_rows() -> None:
    ds = bt.from_pydict({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    with pytest.raises(PlanError, match="at least n_clusters"):
        KMeans(["x", "y"], n_clusters=5).fit(ds)


def test_inertia_falls_as_clusters_rise(blobs) -> None:
    _, _, ds = blobs
    coarse = KMeans(["x", "y"], n_clusters=2, seed=1).fit(ds).inertia_
    fine = KMeans(["x", "y"], n_clusters=4, seed=1).fit(ds).inertia_
    assert fine < coarse
