"""Correlation-significance, proportion, and paired-classifier tests.

Each reuses a survival function already checked against SciPy, so these tests check the whole
composition end to end: `pearson_test` and `spearman_test` against SciPy's own `pearsonr` /
`spearmanr`, the proportion and McNemar tests against their closed forms (SciPy has no direct
McNemar).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import batcher as bt
from batcher.ml.stats import mcnemar_test, pearson_test, proportion_ztest, spearman_test

pytestmark = pytest.mark.unit

scipy_stats = pytest.importorskip("scipy.stats")


def test_pearson_test_matches_scipy() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    y = 0.5 * x + rng.normal(0, 1, 200)
    ds = bt.from_pydict({"x": x.tolist(), "y": y.tolist()})
    got = pearson_test(ds, "x", "y")
    ref = scipy_stats.pearsonr(x, y)
    assert got.statistic == pytest.approx(ref.statistic, abs=1e-9)
    assert got.pvalue == pytest.approx(ref.pvalue, abs=1e-7)


def test_spearman_test_matches_scipy() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 200)
    y = np.exp(x) + rng.normal(0, 0.1, 200)
    ds = bt.from_pydict({"x": x.tolist(), "y": y.tolist()})
    got = spearman_test(ds, "x", "y")
    ref = scipy_stats.spearmanr(x, y)
    assert got.statistic == pytest.approx(ref.correlation, abs=1e-6)
    assert got.pvalue == pytest.approx(ref.pvalue, abs=1e-5)


def test_proportion_ztest_matches_the_closed_form() -> None:
    won = np.array([1] * 70 + [0] * 30)
    ds = bt.from_pydict({"w": won.tolist()})
    got = proportion_ztest(ds, "w", 0.5)
    z = (0.7 - 0.5) / math.sqrt(0.25 / 100)
    assert got.statistic == pytest.approx(z)
    assert got.pvalue == pytest.approx(math.erfc(abs(z) / math.sqrt(2)))


def test_proportion_ztest_keeps_a_fair_coin() -> None:
    rng = np.random.default_rng(4)
    flips = rng.integers(0, 2, 1000)
    ds = bt.from_pydict({"h": flips.tolist()})
    assert proportion_ztest(ds, "h", 0.5).pvalue > 0.05


def test_mcnemar_uses_only_the_discordant_pairs() -> None:
    a = np.array([True] * 60 + [False] * 40)
    b = np.array([True] * 50 + [False] * 10 + [True] * 25 + [False] * 15)
    ds = bt.from_pydict({"a": a.tolist(), "b": b.tolist()})
    got = mcnemar_test(ds, "a", "b")
    b_only = int(((~a) & b).sum())
    c_only = int((a & (~b)).sum())
    ref = (abs(b_only - c_only) - 1) ** 2 / (b_only + c_only)
    assert got.statistic == pytest.approx(ref)
    assert got.df == 1.0


def test_mcnemar_is_null_when_the_models_agree() -> None:
    # No discordant pairs: nothing distinguishes the two models.
    ds = bt.from_pydict({"a": [True, False, True], "b": [True, False, True]})
    got = mcnemar_test(ds, "a", "b")
    assert got.statistic == 0.0
    assert got.pvalue == 1.0


def test_mcnemar_flags_a_one_sided_improvement() -> None:
    # B fixes 30 of A's errors and breaks none: a clear, significant improvement.
    a = np.array([False] * 40 + [True] * 60)
    b = a.copy()
    b[:30] = True
    ds = bt.from_pydict({"a": a.tolist(), "b": b.tolist()})
    assert mcnemar_test(ds, "a", "b").pvalue < 0.001
