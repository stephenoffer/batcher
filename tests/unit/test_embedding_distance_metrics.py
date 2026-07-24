"""Embedding distance metrics — mean pairwise distance across the spaces ANN indexes use.

Each is a per-row vector distance aggregated to a corpus mean, so they are pinned to numpy over the
same paired vectors: cosine distance (1 - cos), Manhattan (L1), angular (normalized angle), and
Hamming (differing positions). A random batch checks the general case; small batches pin the edges.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def pairs() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(64, 8))
    b = rng.normal(size=(64, 8))
    ds = bt.from_pydict({"a": a.tolist(), "b": b.tolist()})
    return a, b, ds


def _mean(ds: bt.Dataset, expr) -> float:
    return ds.agg(m=expr).to_pydict()["m"][0]


def test_cosine_distance_matches_one_minus_cosine(pairs) -> None:
    a, b, ds = pairs
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    expected = float((1.0 - cos).mean())
    assert _mean(ds, bt.mean_cosine_distance("a", "b")) == pytest.approx(expected, rel=1e-6)


def test_manhattan_distance_matches_l1(pairs) -> None:
    a, b, ds = pairs
    expected = float(np.abs(a - b).sum(1).mean())
    assert _mean(ds, bt.mean_manhattan_distance("a", "b")) == pytest.approx(expected, rel=1e-6)


def test_angular_distance_matches_normalized_arccos(pairs) -> None:
    a, b, ds = pairs
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    expected = float((np.arccos(np.clip(cos, -1.0, 1.0)) / np.pi).mean())
    assert _mean(ds, bt.mean_angular_distance("a", "b")) == pytest.approx(expected, rel=1e-6)


def test_hamming_distance_counts_differing_positions() -> None:
    ds = bt.from_pydict({"a": [[1.0, 0.0, 1.0, 0.0]], "b": [[1.0, 1.0, 0.0, 0.0]]})
    assert _mean(ds, bt.mean_hamming_distance("a", "b")) == pytest.approx(2.0)


def test_orthogonal_vectors_have_half_angular_distance() -> None:
    ds = bt.from_pydict({"a": [[1.0, 0.0]], "b": [[0.0, 1.0]]})
    assert _mean(ds, bt.mean_angular_distance("a", "b")) == pytest.approx(0.5, abs=1e-6)


def test_identical_vectors_have_zero_distance() -> None:
    ds = bt.from_pydict({"a": [[1.0, 2.0, 3.0]], "b": [[1.0, 2.0, 3.0]]})
    for metric in (
        bt.mean_cosine_distance,
        bt.mean_manhattan_distance,
        bt.mean_angular_distance,
        bt.mean_hamming_distance,
    ):
        assert _mean(ds, metric("a", "b")) == pytest.approx(0.0, abs=1e-6)


def test_distance_metrics_compose_with_group_by() -> None:
    ds = bt.from_pydict(
        {"m": ["x", "x"], "a": [[1.0, 0.0], [0.0, 1.0]], "b": [[1.0, 0.0], [1.0, 0.0]]}
    )
    out = ds.group_by("m").agg(d=bt.mean_cosine_distance("a", "b")).to_pydict()
    assert out["m"] == ["x"]
    assert out["d"][0] == pytest.approx(0.5)
