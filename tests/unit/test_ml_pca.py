"""Principal component analysis.

PCA reproduces scikit-learn up to the per-component sign it is free to flip, so the projection
is checked against `sklearn.decomposition.PCA` after sign-aligning each component, and the
explained-variance ratio directly. The structural contract — output columns, dropping the
inputs, learning the training basis for serving — is pinned separately.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.preprocessors import PCA

pytestmark = pytest.mark.unit

sk_decomp = pytest.importorskip("sklearn.decomposition")


@pytest.fixture(scope="module")
def wide() -> tuple[np.ndarray, list[str], bt.Dataset]:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (300, 4)) @ rng.normal(0, 1, (4, 4))
    names = [f"f{i}" for i in range(4)]
    ds = bt.from_pydict({name: x[:, i].tolist() for i, name in enumerate(names)})
    return x, names, ds


def _sign_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.copy()
    for j in range(a.shape[1]):
        if np.dot(a[:, j], b[:, j]) < 0:
            a[:, j] = -a[:, j]
    return a


def test_projection_matches_sklearn(wide) -> None:
    x, names, ds = wide
    pre = PCA(names, n_components=2).fit(ds)
    out = pre.transform(ds).to_pydict()
    proj = np.column_stack([out["pc1"], out["pc2"]])
    ref = sk_decomp.PCA(n_components=2).fit_transform(x)
    assert np.allclose(_sign_align(proj, ref), ref, atol=1e-6)


def test_explained_variance_ratio_matches_sklearn(wide) -> None:
    x, names, ds = wide
    pre = PCA(names, n_components=3).fit(ds)
    ref = sk_decomp.PCA(n_components=3).fit(x).explained_variance_ratio_
    assert np.allclose(pre.explained_variance_ratio_, ref, atol=1e-6)


def test_components_are_ordered_by_variance(wide) -> None:
    _, names, ds = wide
    pre = PCA(names, n_components=4).fit(ds)
    assert pre.explained_variance_ == sorted(pre.explained_variance_, reverse=True)


def test_transform_replaces_features_and_keeps_others() -> None:
    ds = bt.from_pydict({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0], "id": [0, 1, 2]})
    out = PCA(["a", "b"], n_components=1).fit_transform(ds)
    assert out.columns == ["id", "pc1"]


def test_keep_original_retains_the_inputs() -> None:
    ds = bt.from_pydict({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    out = PCA(["a", "b"], n_components=1, keep_original=True).fit_transform(ds)
    assert set(out.columns) == {"a", "b", "pc1"}


def test_serving_reuses_the_training_basis() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (200, 3))
    names = ["a", "b", "c"]
    train = bt.from_pydict({n: x[:, i].tolist() for i, n in enumerate(names)})
    pre = PCA(names, n_components=2).fit(train)
    sk = sk_decomp.PCA(n_components=2).fit(x)
    serve_x = rng.normal(0, 1, (5, 3))
    serve = bt.from_pydict({n: serve_x[:, i].tolist() for i, n in enumerate(names)})
    got = pre.transform(serve).to_pydict()
    proj = np.column_stack([got["pc1"], got["pc2"]])
    ref = sk.transform(serve_x)
    assert np.allclose(_sign_align(proj, ref), ref, atol=1e-6)


def test_rejects_too_many_components() -> None:
    with pytest.raises(PlanError, match="n_components must be"):
        PCA(["a", "b"], n_components=5)
