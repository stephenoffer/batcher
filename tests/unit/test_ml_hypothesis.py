"""Hypothesis tests — statistic and p-value both checked against SciPy.

These pair a mergeable statistic with a reference-distribution tail probability computed in
dependency-free Python. The point of the module is the p-value, so the tests check it against
SciPy's own tests directly rather than re-deriving it, and separately check the distribution
survival functions across their range.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.stats import (
    anova_test,
    chi_square_test,
    normality_test,
    t_test_1samp,
    t_test_ind,
)
from batcher.ml.stats._special import chi2_sf, f_sf, students_t_two_sided_p

pytestmark = pytest.mark.unit

scipy_stats = pytest.importorskip("scipy.stats")


# --- survival functions ----------------------------------------------------------------


@pytest.mark.parametrize(("t", "df"), [(2.0, 10), (0.5, 3), (3.5, 100), (1.0, 1), (0.0, 5)])
def test_students_t_matches_scipy(t: float, df: int) -> None:
    assert students_t_two_sided_p(t, df) == pytest.approx(
        2 * scipy_stats.t.sf(abs(t), df), abs=1e-10
    )


@pytest.mark.parametrize(("f", "d1", "d2"), [(2.0, 3, 20), (1.0, 5, 5), (5.0, 2, 100), (0.3, 4, 4)])
def test_f_survival_matches_scipy(f: float, d1: int, d2: int) -> None:
    assert f_sf(f, d1, d2) == pytest.approx(scipy_stats.f.sf(f, d1, d2), abs=1e-10)


@pytest.mark.parametrize(("x", "df"), [(3.0, 2), (10.0, 5), (1.0, 1), (50.0, 30), (0.5, 10)])
def test_chi2_survival_matches_scipy(x: float, df: int) -> None:
    assert chi2_sf(x, df) == pytest.approx(scipy_stats.chi2.sf(x, df), abs=1e-10)


# --- tests vs scipy --------------------------------------------------------------------


def test_one_sample_t_matches_scipy() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 2.0, 200)
    ds = bt.from_pydict({"x": x.tolist()})
    got = t_test_1samp(ds, 0.5)
    t, p = scipy_stats.ttest_1samp(x, 0.5)
    assert got.statistic == pytest.approx(t, abs=1e-9)
    assert got.pvalue == pytest.approx(p, abs=1e-9)
    assert got.df == 199.0


def test_welch_two_sample_matches_scipy() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 150), rng.normal(0.4, 1.5, 180)
    ds = bt.from_pydict({"g": ["a"] * 150 + ["b"] * 180, "x": a.tolist() + b.tolist()})
    got = t_test_ind(ds, "x", "g")
    t, p = scipy_stats.ttest_ind(a, b, equal_var=False)
    assert abs(got.statistic) == pytest.approx(abs(t), abs=1e-8)
    assert got.pvalue == pytest.approx(p, abs=1e-8)


def test_welch_rejects_a_non_binary_group() -> None:
    ds = bt.from_pydict({"g": ["a", "b", "c"], "x": [1.0, 2.0, 3.0]})
    with pytest.raises(PlanError, match="exactly two groups"):
        t_test_ind(ds, "x", "g")


def test_anova_matches_scipy() -> None:
    rng = np.random.default_rng(2)
    g1, g2, g3 = rng.normal(0, 1, 80), rng.normal(0.5, 1, 90), rng.normal(1.0, 1, 70)
    ds = bt.from_pydict(
        {"g": ["a"] * 80 + ["b"] * 90 + ["c"] * 70, "x": g1.tolist() + g2.tolist() + g3.tolist()}
    )
    got = anova_test(ds, "x", "g")
    f, p = scipy_stats.f_oneway(g1, g2, g3)
    assert got.statistic == pytest.approx(f, abs=1e-7)
    assert got.pvalue == pytest.approx(p, abs=1e-9)
    assert got.df == (2.0, 237.0)


def test_chi_square_matches_scipy() -> None:
    import pandas as pd

    rng = np.random.default_rng(3)
    a, b = rng.integers(0, 3, 500), rng.integers(0, 4, 500)
    ds = bt.from_pydict({"a": a.tolist(), "b": b.tolist()})
    got = chi_square_test(ds, "a", "b")
    chi2, p, dof, _ = scipy_stats.chi2_contingency(pd.crosstab(a, b).values, correction=False)
    assert got.statistic == pytest.approx(chi2, abs=1e-7)
    assert got.pvalue == pytest.approx(p, abs=1e-9)
    assert got.df == dof


def test_normality_rejects_a_skewed_column() -> None:
    rng = np.random.default_rng(4)
    x = rng.exponential(1.0, 2000)
    ds = bt.from_pydict({"x": x.tolist()})
    assert normality_test(ds, "x").pvalue < 0.01


def test_normality_keeps_a_gaussian_column() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 2000)
    ds = bt.from_pydict({"x": x.tolist()})
    assert normality_test(ds, "x").pvalue > 0.01
