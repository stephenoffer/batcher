"""Gaussian mixture models by expectation-maximization.

EM finds a local optimum, so there is no exact oracle; the tests pin the properties that must
hold: the mean log-likelihood increases every iteration, on separable data the soft clustering
agrees with both the truth and scikit-learn (adjusted Rand index near 1), and `score_samples`
scores a genuine outlier far below every inlier.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.mixture import GaussianMixture

pytestmark = pytest.mark.unit

sk_mixture = pytest.importorskip("sklearn.mixture")
sk_metrics = pytest.importorskip("sklearn.metrics")


@pytest.fixture(scope="module")
def blobs() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    x = np.vstack(
        [
            rng.normal([0, 0], [1, 2], (200, 2)),
            rng.normal([6, 6], [2, 1], (200, 2)),
            rng.normal([0, 7], [1, 1], (200, 2)),
        ]
    )
    truth = np.repeat([0, 1, 2], 200)
    ds = bt.from_pydict({"a": x[:, 0].tolist(), "b": x[:, 1].tolist()})
    return x, truth, ds


def test_recovers_the_components(blobs) -> None:
    _, truth, ds = blobs
    labels = np.array(
        GaussianMixture(["a", "b"], n_components=3, seed=2)
        .fit(ds)
        .predict(ds)
        .to_pydict()["component"]
    )
    assert sk_metrics.adjusted_rand_score(truth, labels) > 0.9


def test_agrees_with_sklearn_up_to_permutation(blobs) -> None:
    x, _, ds = blobs
    labels = np.array(
        GaussianMixture(["a", "b"], n_components=3, seed=2)
        .fit(ds)
        .predict(ds)
        .to_pydict()["component"]
    )
    sk = sk_mixture.GaussianMixture(
        n_components=3, covariance_type="full", random_state=0, n_init=5
    )
    assert sk_metrics.adjusted_rand_score(sk.fit(x).predict(x), labels) > 0.9


def test_score_samples_flags_an_outlier(blobs) -> None:
    _, _, ds = blobs
    gm = GaussianMixture(["a", "b"], n_components=3, seed=2).fit(ds)
    inliers = gm.score_samples(ds).to_pydict()["log_likelihood"]
    outlier = gm.score_samples(bt.from_pydict({"a": [50.0], "b": [50.0]})).to_pydict()[
        "log_likelihood"
    ][0]
    assert outlier < min(inliers)


def test_log_likelihood_increases_monotonically(blobs) -> None:
    _, _, ds = blobs
    likelihoods = []
    for max_iter in range(1, 8):
        gm = GaussianMixture(["a", "b"], n_components=3, seed=2, max_iter=max_iter, tol=0.0).fit(ds)
        likelihoods.append(gm.log_likelihood_)
    # Each extra EM iteration cannot decrease the log-likelihood.
    import itertools

    for earlier, later in itertools.pairwise(likelihoods):
        assert later >= earlier - 1e-9


def test_is_reproducible_from_the_seed(blobs) -> None:
    _, _, ds = blobs
    a = GaussianMixture(["a", "b"], n_components=3, seed=7).fit(ds).means_
    b = GaussianMixture(["a", "b"], n_components=3, seed=7).fit(ds).means_
    assert a == b


def test_rejects_no_features() -> None:
    with pytest.raises(PlanError, match="at least one feature"):
        GaussianMixture([], n_components=2)


def test_rejects_more_components_than_rows() -> None:
    ds = bt.from_pydict({"a": [1.0, 2.0], "b": [1.0, 2.0]})
    with pytest.raises(PlanError, match="at least"):
        GaussianMixture(["a", "b"], n_components=5).fit(ds)


def test_names_a_missing_column(blobs) -> None:
    _, _, ds = blobs
    with pytest.raises(ColumnNotFoundError):
        GaussianMixture(["a", "nope"], n_components=2).fit(ds)


def test_a_bimodal_column_does_not_collapse_to_the_global_distribution() -> None:
    """Both components starting alike is a fixed point EM can never leave.

    Seeding each mean from a hash-sampled row while every covariance starts at the *global*
    covariance meant two nearby sampled rows made every responsibility ~1/K. Identical
    components then reproduce identical components, so `fit` sat there and set `converged_`.
    Whether it happened depended on which rows the hash picked, which is why
    `test_every_seed_starts_the_components_apart` is the case that reproduces it.
    """
    rng = np.random.default_rng(7)
    values = np.concatenate([rng.normal(-3.0, 0.6, 150), rng.normal(3.0, 0.6, 150)])
    ds = bt.from_pydict({"v": values.tolist()})

    gm = GaussianMixture(["v"], n_components=2, max_iter=300, tol=1e-8).fit(ds)
    means = sorted(float(m[0]) for m in gm.means_)
    spread = float(np.var(values))

    assert means[0] < -2.0 < 2.0 < means[1], (
        f"the two components collapsed onto the global mean: {means}"
    )
    for covariance in gm.covariances_:
        assert float(np.asarray(covariance).ravel()[0]) < spread / 2, (
            "a component kept the global covariance, so it is modelling the whole column"
        )


def test_a_bimodal_column_matches_sklearn() -> None:
    rng = np.random.default_rng(7)
    values = np.concatenate([rng.normal(-3.0, 0.6, 150), rng.normal(3.0, 0.6, 150)])
    ds = bt.from_pydict({"v": values.tolist()})

    gm = GaussianMixture(["v"], n_components=2, max_iter=300, tol=1e-8).fit(ds)
    sk = sk_mixture.GaussianMixture(n_components=2, max_iter=300, tol=1e-8, random_state=0).fit(
        values.reshape(-1, 1)
    )

    got = sorted(float(m[0]) for m in gm.means_)
    want = sorted(float(m) for m in sk.means_.ravel())
    for g, w in zip(got, want, strict=True):
        assert g == pytest.approx(w, abs=1e-3)


@pytest.mark.parametrize("seed", [0, 1, 3, 7, 11])
def test_every_seed_starts_the_components_apart(seed: int) -> None:
    """A seed picks a different start, never a degenerate one.

    This is the case that reproduces the collapse. On this data, hash-sampling the seed rows
    put ``seed=7`` at means ``[-0.1506, -0.0042]`` with both covariances at ``9.391`` against a
    global variance of ``9.398`` -- two copies of the whole column, ``converged_`` after three
    iterations, where the components are at -3 and +3. Seeds 1, 3 and 11 did escape but needed
    49, 71 and 34 iterations to do it, against 6 from a separated start, so the old
    initialization was fragile even when it eventually worked.
    """
    rng = np.random.default_rng(7)
    values = np.concatenate([rng.normal(-3.0, 0.6, 150), rng.normal(3.0, 0.6, 150)])
    ds = bt.from_pydict({"v": values.tolist()})

    gm = GaussianMixture(["v"], n_components=2, max_iter=300, tol=1e-8, seed=seed).fit(ds)
    means = sorted(float(m[0]) for m in gm.means_)
    assert means[0] < -2.0 < 2.0 < means[1], f"seed={seed} collapsed: {means}"


def test_a_constant_column_is_unidentifiable_but_does_not_raise() -> None:
    """Coincident means are the honest answer when there is nothing to separate."""
    ds = bt.from_pydict({"v": [4.0] * 20})
    gm = GaussianMixture(["v"], n_components=2, max_iter=50).fit(ds)
    assert all(np.isfinite(float(m[0])) for m in gm.means_)
    assert sum(gm.weights_) == pytest.approx(1.0, abs=1e-9)
