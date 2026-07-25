"""Hypothesis tests — statistic and p-value both checked against SciPy.

These pair a mergeable statistic with a reference-distribution tail probability computed in
dependency-free Python. The point of the module is the p-value, so the tests check it against
SciPy's own tests directly rather than re-deriving it, and separately check the distribution
survival functions across their range.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.stats import (
    anova_test,
    binomial_test,
    chi_square_test,
    mcnemar_test,
    normality_test,
    proportion_ztest,
    t_test_1samp,
    t_test_ind,
)
from batcher.ml.stats._special import (
    chi2_sf,
    f_sf,
    normal_two_sided_p,
    students_t_two_sided_p,
)

pytestmark = pytest.mark.unit

scipy_stats = pytest.importorskip("scipy.stats")


# --- survival functions ----------------------------------------------------------------


# A *relative* tolerance, because these are p-values and the tail is where they are read. An
# absolute 1e-10 is satisfied by any answer at all once the true value drops below it, so it
# cannot see a tail regression: at chi2_sf(200, 100) the reference is 1.2e-08 and at
# normal_two_sided_p(15) it is 7.3e-51. Measured worst relative error across this grid is
# 4.4e-13, so 1e-9 is loose by three orders of magnitude and still catches a real drift.
_TAIL_RTOL = 1e-9


@pytest.mark.parametrize(
    ("t", "df"),
    [(2.0, 10), (0.5, 3), (3.5, 100), (1.0, 1), (0.0, 5), (10.0, 5), (30.0, 30), (100.0, 1000)],
)
def test_students_t_matches_scipy(t: float, df: int) -> None:
    assert students_t_two_sided_p(t, df) == pytest.approx(
        2 * scipy_stats.t.sf(abs(t), df), rel=_TAIL_RTOL, abs=1e-300
    )


@pytest.mark.parametrize(
    ("f", "d1", "d2"),
    [(2.0, 3, 20), (1.0, 5, 5), (5.0, 2, 100), (0.3, 4, 4), (100.0, 5, 20), (1000.0, 20, 1000)],
)
def test_f_survival_matches_scipy(f: float, d1: int, d2: int) -> None:
    assert f_sf(f, d1, d2) == pytest.approx(scipy_stats.f.sf(f, d1, d2), rel=_TAIL_RTOL, abs=1e-300)


@pytest.mark.parametrize(
    ("x", "df"),
    [
        (3.0, 2),
        (10.0, 5),
        (1.0, 1),
        (50.0, 30),
        (0.5, 10),
        (200.0, 100),
        (400.0, 100),
        (1000.0, 500),
    ],
)
def test_chi2_survival_matches_scipy(x: float, df: int) -> None:
    assert chi2_sf(x, df) == pytest.approx(scipy_stats.chi2.sf(x, df), rel=_TAIL_RTOL, abs=1e-300)


@pytest.mark.parametrize("z", [0.0, 0.5, 1.0, 1.96, 3.0, 5.0, 8.0, 15.0])
def test_normal_two_sided_matches_scipy(z: float) -> None:
    """The one survival function that had no test, and the deepest tail of the four."""
    assert normal_two_sided_p(z) == pytest.approx(
        2 * scipy_stats.norm.sf(abs(z)), rel=_TAIL_RTOL, abs=1e-300
    )


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


@pytest.mark.parametrize("n", [4, 10, 50, 200, 2000])
def test_normality_statistic_and_pvalue_match_scipy(n: int) -> None:
    """Jarque-Bera is defined on the *population* moments, not the bias-corrected ones.

    `skewness`/`kurtosis` are the ``G1``/``G2`` estimators pandas reports, so substituting them
    into the JB formula yields a different statistic under the same name — and a p-value drawn
    from a null distribution that does not describe it. The old tests only asserted which side
    of 0.01 the p-value fell on, which both versions satisfy.
    """
    rng = np.random.default_rng(4)
    x = rng.normal(0.0, 1.0, n)
    ds = bt.from_pydict({"x": x.tolist()})
    got = normality_test(ds, "x")
    want = scipy_stats.jarque_bera(x)
    assert got.statistic == pytest.approx(float(want.statistic), rel=1e-9)
    assert got.pvalue == pytest.approx(float(want.pvalue), rel=1e-9)
    assert got.df == 2.0


@pytest.mark.parametrize("dist", ["normal", "exponential", "uniform"])
def test_normality_statistic_matches_scipy_across_shapes(dist: str) -> None:
    rng = np.random.default_rng(9)
    x = {
        "normal": lambda: rng.normal(0.0, 1.0, 300),
        "exponential": lambda: rng.exponential(1.0, 300),
        "uniform": lambda: rng.uniform(0.0, 1.0, 300),
    }[dist]()
    ds = bt.from_pydict({"x": x.tolist()})
    assert normality_test(ds, "x").statistic == pytest.approx(
        float(scipy_stats.jarque_bera(x).statistic), rel=1e-9
    )


@pytest.mark.parametrize("n", [1, 2, 3])
def test_normality_is_undefined_below_four_rows(n: int) -> None:
    """`G2` needs four rows; the statistic must stay NaN rather than inventing a number."""
    ds = bt.from_pydict({"x": [float(i) for i in range(1, n + 1)]})
    assert math.isnan(normality_test(ds, "x").statistic)


# --------------------------------------------------------------------------- #
# A per-row flag is a boolean or a 0/1 number, and the tests must accept both
# --------------------------------------------------------------------------- #
_FLAG_ENCODINGS = {
    "int": ([1, 1, 0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 1, 0]),
    "bool": (
        [True, True, False, False, True, False, True, True],
        [True, False, True, True, False, False, True, False],
    ),
    "float": ([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
}


@pytest.mark.parametrize("encoding", sorted(_FLAG_ENCODINGS))
def test_indicator_tests_accept_every_flag_encoding(encoding: str) -> None:
    """`== 1` rejects a boolean column and `== True` rejects an integer one, each as a raw
    Arrow `RuntimeError` from inside the engine. The three tests took *opposite* conventions,
    so no single habit worked across the module."""
    a, b = _FLAG_ENCODINGS[encoding]
    ds = bt.from_pydict({"a": a, "b": b})
    assert binomial_test(ds, "a", 0.5).pvalue > 0.0
    assert proportion_ztest(ds, "a", 0.5).statistic == pytest.approx(0.70710678, abs=1e-7)
    assert mcnemar_test(ds, "a", "b").df == 1.0


def test_every_flag_encoding_gives_an_identical_result() -> None:
    results = {}
    for encoding, (a, b) in _FLAG_ENCODINGS.items():
        ds = bt.from_pydict({"a": a, "b": b})
        results[encoding] = (
            binomial_test(ds, "a", 0.5).pvalue,
            proportion_ztest(ds, "a", 0.5).statistic,
            proportion_ztest(ds, "a", 0.5).pvalue,
            mcnemar_test(ds, "a", "b").statistic,
        )
    assert len(set(results.values())) == 1, results


def test_a_null_flag_is_not_counted_as_a_failure() -> None:
    """Casting keeps nulls null, so a missing observation stays missing."""
    ds = bt.from_pydict({"a": [True, None, True, None, True]})
    assert proportion_ztest(ds, "a", 0.5).statistic == pytest.approx(
        proportion_ztest(bt.from_pydict({"a": [1, None, 1, None, 1]}), "a", 0.5).statistic
    )
