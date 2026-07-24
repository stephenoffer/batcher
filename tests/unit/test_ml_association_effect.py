"""Directional categorical association and ANOVA effect sizes.

Theil's U and the eta/epsilon effect sizes are the *bounded* companions to the unbounded
`chi_square`/`anova_f`, so they are pinned to their closed forms (mutual information over
entropy; between-group sum of squares over total) rather than to a p-value, and to the
property that distinguishes each from the statistic it complements.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher.ml.stats import cohens_f, epsilon_squared, eta_squared, omega_squared, theils_u

pytestmark = pytest.mark.unit


def _entropy(a: np.ndarray) -> float:
    _, counts = np.unique(a, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _mutual_info(a: np.ndarray, b: np.ndarray) -> float:
    conditional = sum((a == lvl).mean() * _entropy(b[a == lvl]) for lvl in set(a))
    return _entropy(b) - conditional


@pytest.fixture(scope="module")
def grouped() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    g = np.array(["a"] * 40 + ["b"] * 30 + ["c"] * 30)
    v = np.concatenate([rng.normal(0, 1, 40), rng.normal(2, 1, 30), rng.normal(4, 1, 30)])
    return g, v, bt.from_pydict({"g": g.tolist(), "v": v.tolist()})


def test_eta_squared_matches_the_sum_of_squares_ratio(grouped) -> None:
    g, v, ds = grouped
    grand = v.mean()
    ss_between = sum(len(v[g == lvl]) * (v[g == lvl].mean() - grand) ** 2 for lvl in set(g))
    ss_total = ((v - grand) ** 2).sum()
    assert eta_squared(ds, "v", "g") == pytest.approx(ss_between / ss_total)


def test_eta_squared_is_bounded(grouped) -> None:
    _, _, ds = grouped
    assert 0.0 <= eta_squared(ds, "v", "g") <= 1.0


def test_epsilon_squared_is_the_bias_corrected_eta(grouped) -> None:
    _, v, ds = grouped
    eta = eta_squared(ds, "v", "g")
    n, k = len(v), 3
    expected = eta - (1 - eta) * (k - 1) / (n - k)
    assert epsilon_squared(ds, "v", "g") == pytest.approx(expected)


def test_epsilon_can_go_slightly_negative_on_a_null_effect() -> None:
    rng = np.random.default_rng(5)
    v = rng.normal(0, 1, 300)
    g = rng.integers(0, 10, 300).astype(str)
    ds = bt.from_pydict({"v": v.tolist(), "g": g.tolist()})
    # A grouping with no real effect: epsilon^2 is near zero and may dip below it.
    assert epsilon_squared(ds, "v", "g") < 0.1


def test_theils_u_matches_mi_over_entropy() -> None:
    rng = np.random.default_rng(1)
    x = rng.integers(0, 3, 500).astype(str)
    y = (x.astype(int) + rng.integers(0, 2, 500)).astype(str)
    ds = bt.from_pydict({"x": x.tolist(), "y": y.tolist()})
    assert theils_u(ds, "x", "y") == pytest.approx(_mutual_info(x, y) / _entropy(y))


def test_theils_u_is_asymmetric() -> None:
    # x determines y coarsely but y pins down x more, so the two directions differ.
    rng = np.random.default_rng(2)
    x = rng.integers(0, 5, 400).astype(str)
    y = (x.astype(int) // 2).astype(str)
    ds = bt.from_pydict({"x": x.tolist(), "y": y.tolist()})
    assert theils_u(ds, "x", "y") != pytest.approx(theils_u(ds, "y", "x"))


def test_theils_u_is_one_when_x_determines_y() -> None:
    ds = bt.from_pydict({"x": ["a", "a", "b", "b"], "y": ["p", "p", "q", "q"]})
    assert theils_u(ds, "x", "y") == pytest.approx(1.0)


def test_effect_sizes_compose_with_group_by() -> None:
    ds = bt.from_pydict(
        {"batch": ["m", "m", "m", "m"], "g": ["a", "a", "b", "b"], "v": [1.0, 2.0, 8.0, 9.0]}
    )
    out = ds.group_by("batch").agg(n=bt.col("v").count())
    assert out.count() == 1


def test_omega_squared_is_the_least_biased_effect_size(grouped) -> None:
    g, v, ds = grouped
    grand = v.mean()
    ss_between = sum(len(v[g == lvl]) * (v[g == lvl].mean() - grand) ** 2 for lvl in set(g))
    ss_within = sum(np.sum((v[g == lvl] - v[g == lvl].mean()) ** 2) for lvl in set(g))
    ss_total = ss_between + ss_within
    n, k = len(v), 3
    ms_within = ss_within / (n - k)
    expected = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
    assert omega_squared(ds, "v", "g") == pytest.approx(expected)


def test_omega_is_smaller_than_eta(grouped) -> None:
    _, _, ds = grouped
    assert omega_squared(ds, "v", "g") < eta_squared(ds, "v", "g")


def test_cohens_f_matches_the_eta_transform(grouped) -> None:
    _, _, ds = grouped
    eta = eta_squared(ds, "v", "g")
    assert cohens_f(ds, "v", "g") == pytest.approx(np.sqrt(eta / (1 - eta)))
