"""Embedding-quality metrics on vector columns.

The pairwise metrics are checked against numpy (cosine, L2 distance, dot product), and the
single-column rates against their defining counts.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def pairs() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (50, 8))
    b = rng.normal(0, 1, (50, 8))
    ds = bt.from_pydict({"a": a.tolist(), "b": b.tolist()})
    return a, b, ds


def _agg(ds: bt.Dataset, expr) -> float:
    return ds.agg(m=expr).to_pydict()["m"][0]


def test_mean_cosine_similarity_matches_numpy(pairs) -> None:
    a, b, ds = pairs
    ref = np.mean(
        [float(a[i] @ b[i] / (np.linalg.norm(a[i]) * np.linalg.norm(b[i]))) for i in range(len(a))]
    )
    assert _agg(ds, bt.mean_cosine_similarity("a", "b")) == pytest.approx(ref, abs=1e-6)


def test_mean_euclidean_distance_matches_numpy(pairs) -> None:
    a, b, ds = pairs
    ref = np.mean([float(np.linalg.norm(a[i] - b[i])) for i in range(len(a))])
    assert _agg(ds, bt.mean_euclidean_distance("a", "b")) == pytest.approx(ref, abs=1e-6)


def test_mean_dot_product_matches_numpy(pairs) -> None:
    a, b, ds = pairs
    ref = np.mean([float(a[i] @ b[i]) for i in range(len(a))])
    assert _agg(ds, bt.mean_dot_product("a", "b")) == pytest.approx(ref, abs=1e-6)


def test_mean_embedding_norm_matches_numpy(pairs) -> None:
    a, _, ds = pairs
    ref = np.mean([float(np.linalg.norm(a[i])) for i in range(len(a))])
    assert _agg(ds, bt.mean_embedding_norm("a")) == pytest.approx(ref, abs=1e-6)


def test_unit_norm_rate() -> None:
    ds = bt.from_pydict({"v": [[1.0, 0.0], [0.6, 0.8], [3.0, 4.0]]})
    # Two of three are unit length (norm 1, 1), one is not (norm 5).
    assert _agg(ds, bt.unit_norm_rate("v")) == pytest.approx(2 / 3)


def test_zero_vector_rate() -> None:
    ds = bt.from_pydict({"v": [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]})
    assert _agg(ds, bt.zero_vector_rate("v")) == pytest.approx(2 / 3)


def test_identical_vectors_have_cosine_one() -> None:
    ds = bt.from_pydict({"a": [[1.0, 2.0, 3.0]], "b": [[1.0, 2.0, 3.0]]})
    assert _agg(ds, bt.mean_cosine_similarity("a", "b")) == pytest.approx(1.0)
    assert _agg(ds, bt.mean_euclidean_distance("a", "b")) == pytest.approx(0.0)


def test_metrics_compose_with_group_by() -> None:
    ds = bt.from_pydict(
        {"g": ["x", "x", "y"], "a": [[1.0, 0.0]] * 3, "b": [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]}
    )
    out = ds.group_by("g").agg(c=bt.mean_cosine_similarity("a", "b")).sort("g").to_pydict()
    assert out["c"][0] == pytest.approx(0.5)
    assert out["c"][1] == pytest.approx(1.0)
