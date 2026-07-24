"""Truncated SVD dimensionality reduction.

Reproduces scikit-learn's `TruncatedSVD` up to the per-component sign, so the projection is
checked after sign-alignment and the explained-variance ratio directly. The structural contract
(output columns, dropping inputs, serving with the training basis) is pinned separately.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.preprocessors import TruncatedSVD

pytestmark = pytest.mark.unit

sk_decomp = pytest.importorskip("sklearn.decomposition")


@pytest.fixture(scope="module")
def matrix() -> tuple[np.ndarray, list[str], bt.Dataset]:
    rng = np.random.default_rng(0)
    m = rng.normal(0, 1, (300, 5)) @ rng.normal(0, 1, (5, 5))
    names = [f"c{i}" for i in range(5)]
    ds = bt.from_pydict({name: m[:, i].tolist() for i, name in enumerate(names)})
    return m, names, ds


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.copy()
    for j in range(a.shape[1]):
        if np.dot(a[:, j], b[:, j]) < 0:
            a[:, j] = -a[:, j]
    return a


def test_projection_matches_sklearn(matrix) -> None:
    m, names, ds = matrix
    pre = TruncatedSVD(names, n_components=2).fit(ds)
    out = pre.transform(ds).to_pydict()
    proj = np.column_stack([out["svd1"], out["svd2"]])
    ref = sk_decomp.TruncatedSVD(n_components=2, algorithm="arpack").fit_transform(m)
    assert np.allclose(_align(proj, ref), ref, atol=1e-6)


def test_explained_variance_ratio_matches_sklearn(matrix) -> None:
    m, names, ds = matrix
    pre = TruncatedSVD(names, n_components=3).fit(ds)
    ref = (
        sk_decomp.TruncatedSVD(n_components=3, algorithm="arpack").fit(m).explained_variance_ratio_
    )
    assert np.allclose(pre.explained_variance_ratio_, ref, atol=1e-6)


def test_transform_replaces_features_and_keeps_others() -> None:
    ds = bt.from_pydict({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0], "id": [0, 1, 2]})
    out = TruncatedSVD(["a", "b"], n_components=1).fit_transform(ds)
    assert out.columns == ["id", "svd1"]


def test_serving_reuses_the_training_basis(matrix) -> None:
    m, names, ds = matrix
    pre = TruncatedSVD(names, n_components=2).fit(ds)
    sk = sk_decomp.TruncatedSVD(n_components=2, algorithm="arpack").fit(m)
    rng = np.random.default_rng(5)
    serve = rng.normal(0, 1, (6, 5))
    serve_ds = bt.from_pydict({name: serve[:, i].tolist() for i, name in enumerate(names)})
    out = pre.transform(serve_ds).to_pydict()
    proj = np.column_stack([out["svd1"], out["svd2"]])
    ref = sk.transform(serve)
    assert np.allclose(_align(proj, ref), ref, atol=1e-6)


def test_rejects_too_many_components() -> None:
    with pytest.raises(PlanError, match="n_components must be"):
        TruncatedSVD(["a", "b"], n_components=5)
