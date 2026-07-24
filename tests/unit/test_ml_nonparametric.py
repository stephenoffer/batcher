"""Rank-based non-parametric tests (Mann-Whitney U and Kruskal-Wallis).

Ranks with tie-averaging and the tie correction are exactly where these tests go wrong, so the
bar is an exact match with SciPy's asymptotic `mannwhitneyu` (continuity-corrected) and `kruskal`
on both the statistic and the p-value, on data with deliberate ties.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.stats import (
    cliffs_delta,
    common_language_effect_size,
    friedman_test,
    kruskal_wallis,
    mann_whitney_u,
    wilcoxon_signed_rank,
)

pytestmark = pytest.mark.unit

scipy_stats = pytest.importorskip("scipy.stats")


def test_mann_whitney_matches_scipy_with_ties() -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 10, 60).astype(float)
    b = rng.integers(2, 12, 70).astype(float)
    ds = bt.from_pydict({"g": ["a"] * 60 + ["b"] * 70, "x": a.tolist() + b.tolist()})
    got = mann_whitney_u(ds, "x", "g")
    u, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
    assert got.statistic == pytest.approx(u)
    assert got.pvalue == pytest.approx(p, abs=1e-7)


def test_mann_whitney_rejects_a_shift() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 100)
    b = rng.normal(1.5, 1, 100)
    ds = bt.from_pydict({"g": ["a"] * 100 + ["b"] * 100, "x": a.tolist() + b.tolist()})
    assert mann_whitney_u(ds, "x", "g").pvalue < 0.001


def test_mann_whitney_rejects_a_non_binary_group() -> None:
    ds = bt.from_pydict({"g": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]})
    with pytest.raises(PlanError, match="exactly two groups"):
        mann_whitney_u(ds, "x", "g")


def test_kruskal_matches_scipy_with_ties() -> None:
    rng = np.random.default_rng(2)
    g1 = rng.integers(0, 8, 50).astype(float)
    g2 = rng.integers(2, 10, 60).astype(float)
    g3 = rng.integers(1, 9, 55).astype(float)
    ds = bt.from_pydict(
        {"g": ["a"] * 50 + ["b"] * 60 + ["c"] * 55, "x": g1.tolist() + g2.tolist() + g3.tolist()}
    )
    got = kruskal_wallis(ds, "x", "g")
    h, p = scipy_stats.kruskal(g1, g2, g3)
    assert got.statistic == pytest.approx(h, abs=1e-7)
    assert got.pvalue == pytest.approx(p, abs=1e-7)
    assert got.df == 2.0


def test_kruskal_rejects_shifted_groups() -> None:
    rng = np.random.default_rng(3)
    groups = [rng.normal(m, 1, 80) for m in (0.0, 1.0, 2.0)]
    ds = bt.from_pydict(
        {
            "g": ["a"] * 80 + ["b"] * 80 + ["c"] * 80,
            "x": np.concatenate(groups).tolist(),
        }
    )
    assert kruskal_wallis(ds, "x", "g").pvalue < 0.001


def test_effect_sizes_match_the_pairwise_definition() -> None:
    rng = np.random.default_rng(4)
    a = rng.integers(0, 10, 50).astype(float)
    b = rng.integers(3, 13, 60).astype(float)
    ds = bt.from_pydict({"g": ["a"] * 50 + ["b"] * 60, "x": a.tolist() + b.tolist()})
    greater = sum(1 for x in a for y in b if x > y)
    less = sum(1 for x in a for y in b if x < y)
    equal = sum(1 for x in a for y in b if x == y)
    pairs = len(a) * len(b)
    assert cliffs_delta(ds, "x", "g") == pytest.approx((greater - less) / pairs)
    assert common_language_effect_size(ds, "x", "g") == pytest.approx(
        (greater + 0.5 * equal) / pairs
    )


def test_cliffs_delta_is_the_rescaled_cles() -> None:
    rng = np.random.default_rng(6)
    a = rng.normal(0, 1, 80)
    b = rng.normal(1, 1, 90)
    ds = bt.from_pydict({"g": ["a"] * 80 + ["b"] * 90, "x": a.tolist() + b.tolist()})
    cles = common_language_effect_size(ds, "x", "g")
    assert cliffs_delta(ds, "x", "g") == pytest.approx(2 * cles - 1)


def test_cliffs_delta_is_extreme_for_disjoint_groups() -> None:
    ds = bt.from_pydict({"g": ["a", "a", "b", "b"], "x": [1.0, 2.0, 10.0, 11.0]})
    assert cliffs_delta(ds, "x", "g") == pytest.approx(-1.0)


def test_wilcoxon_matches_scipy() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.3, 1, 60)
    y = rng.normal(0, 1, 60)
    ds = bt.from_pydict({"x": x.tolist(), "y": y.tolist()})
    got = wilcoxon_signed_rank(ds, "x", "y")
    w, p = scipy_stats.wilcoxon(x, y, correction=True, mode="approx")
    assert got.statistic == pytest.approx(w)
    assert got.pvalue == pytest.approx(p, abs=1e-6)


def test_wilcoxon_matches_scipy_with_ties() -> None:
    rng = np.random.default_rng(1)
    x = rng.integers(0, 5, 80).astype(float)
    y = rng.integers(0, 5, 80).astype(float)
    ds = bt.from_pydict({"x": x.tolist(), "y": y.tolist()})
    got = wilcoxon_signed_rank(ds, "x", "y")
    w, p = scipy_stats.wilcoxon(x, y, correction=True, mode="approx")
    assert got.statistic == pytest.approx(w)
    assert got.pvalue == pytest.approx(p, abs=1e-6)


def test_wilcoxon_rejects_a_consistent_shift() -> None:
    rng = np.random.default_rng(2)
    before = rng.normal(0, 1, 100)
    after = before + 0.8  # every pair increases
    ds = bt.from_pydict({"x": before.tolist(), "y": after.tolist()})
    assert wilcoxon_signed_rank(ds, "x", "y").pvalue < 0.001


def _long_frame(data: np.ndarray) -> bt.Dataset:
    rows: dict[str, list] = {"subject": [], "drug": [], "score": []}
    n, k = data.shape
    for b in range(n):
        for t in range(k):
            rows["subject"].append(b)
            rows["drug"].append(f"t{t}")
            rows["score"].append(float(data[b, t]))
    return bt.from_pydict(rows)


def test_friedman_matches_scipy() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, (30, 4)) + np.array([0, 0.5, 1.0, 0.3])
    got = friedman_test(_long_frame(data), "score", "subject", "drug")
    stat, p = scipy_stats.friedmanchisquare(*[data[:, j] for j in range(4)])
    assert got.statistic == pytest.approx(stat, abs=1e-6)
    assert got.pvalue == pytest.approx(p, abs=1e-7)
    assert got.df == 3.0


def test_friedman_matches_scipy_with_ties() -> None:
    rng = np.random.default_rng(1)
    data = rng.integers(0, 4, (25, 3)).astype(float)
    got = friedman_test(_long_frame(data), "score", "subject", "drug")
    stat, p = scipy_stats.friedmanchisquare(*[data[:, j] for j in range(3)])
    assert got.statistic == pytest.approx(stat, abs=1e-6)
    assert got.pvalue == pytest.approx(p, abs=1e-7)


def test_friedman_rejects_a_consistent_treatment_effect() -> None:
    rng = np.random.default_rng(2)
    data = rng.normal(0, 0.3, (40, 3)) + np.array([0.0, 1.0, 2.0])
    assert friedman_test(_long_frame(data), "score", "subject", "drug").pvalue < 0.001
