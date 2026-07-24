"""Variance-homogeneity tests (Bartlett and Levene).

Both reduce to per-group aggregates plus a survival function, so the test is an exact match with
SciPy's `bartlett` and `levene` (median-centered) on both the statistic and the p-value, plus the
behavioural check that unequal spreads are rejected and equal spreads are not.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher.ml.stats import bartlett_test, levene_test

pytestmark = pytest.mark.unit

scipy_stats = pytest.importorskip("scipy.stats")


def _dataset(groups: dict[str, np.ndarray]) -> bt.Dataset:
    rows: dict[str, list] = {"g": [], "x": []}
    for name, values in groups.items():
        rows["g"] += [name] * len(values)
        rows["x"] += values.tolist()
    return bt.from_pydict(rows)


@pytest.fixture(scope="module")
def unequal() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "a": rng.normal(0, 1, 80),
        "b": rng.normal(0, 2, 90),
        "c": rng.normal(0, 1.5, 70),
    }


def test_bartlett_matches_scipy(unequal) -> None:
    ds = _dataset(unequal)
    got = bartlett_test(ds, "x", "g")
    stat, p = scipy_stats.bartlett(*unequal.values())
    assert got.statistic == pytest.approx(stat, abs=1e-7)
    assert got.pvalue == pytest.approx(p, abs=1e-9)
    assert got.df == 2.0


def test_levene_matches_scipy(unequal) -> None:
    ds = _dataset(unequal)
    got = levene_test(ds, "x", "g")
    stat, p = scipy_stats.levene(*unequal.values())
    assert got.statistic == pytest.approx(stat, abs=1e-6)
    assert got.pvalue == pytest.approx(p, abs=1e-7)
    assert got.df == (2.0, 237.0)


def test_both_reject_clearly_unequal_variances(unequal) -> None:
    ds = _dataset(unequal)
    assert bartlett_test(ds, "x", "g").pvalue < 0.01
    assert levene_test(ds, "x", "g").pvalue < 0.01


def test_both_keep_equal_variances() -> None:
    rng = np.random.default_rng(5)
    groups = {name: rng.normal(0, 1, 200) for name in ("a", "b", "c")}
    ds = _dataset(groups)
    assert bartlett_test(ds, "x", "g").pvalue > 0.05
    assert levene_test(ds, "x", "g").pvalue > 0.05
