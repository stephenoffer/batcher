"""Two-sample statistics and interval estimation, against SciPy and the closed forms.

`plan.functions.analysis.inference` had no tests, which is how `mean_ci_half_width` and
`proportion_ci_half_width` came to serve every confidence level below 0.97 with the 95%
multiplier: the docstring said only 0.95 and 0.99 were exact, and nothing checked the rest.

The multiplier is now the normal quantile, evaluated once when the plan is built, so every
level is exact. Both are checked here against SciPy, along with the rest of the module.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit

scipy_stats = pytest.importorskip("scipy.stats")

_N = 200


@pytest.fixture(scope="module")
def rate_data() -> tuple[np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(5)
    hit = rng.random(_N) < 0.4
    return hit, bt.from_pydict({"hit": hit.tolist()})


@pytest.fixture(scope="module")
def mean_data() -> tuple[np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, 1.0, _N)
    return x, bt.from_pydict({"x": x.tolist()})


@pytest.fixture(scope="module")
def two_groups() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(11)
    a = rng.normal(10.0, 2.0, 120)
    b = rng.normal(11.5, 3.0, 90)
    ds = bt.from_pydict(
        {
            "x": np.concatenate([a, b]).tolist(),
            "g": ([True] * 120) + ([False] * 90),
        }
    )
    return a, b, ds


@pytest.mark.parametrize("confidence", [0.5, 0.8, 0.9, 0.95, 0.99, 0.999])
def test_proportion_ci_half_width_is_exact_at_every_level(rate_data, confidence) -> None:
    """0.5, 0.9 and 0.95 all returned the 95% half-width, so a 90% interval was 15% too wide."""
    hit, ds = rate_data
    got = ds.agg(w=bt.proportion_ci_half_width(bt.col("hit"), confidence=confidence)).to_pydict()[
        "w"
    ][0]
    p = float(hit.mean())
    z = float(scipy_stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    assert got == pytest.approx(z * np.sqrt(p * (1.0 - p) / _N), rel=1e-8)


@pytest.mark.parametrize("confidence", [0.5, 0.8, 0.9, 0.95, 0.99, 0.999])
def test_mean_ci_half_width_is_exact_at_every_level(mean_data, confidence) -> None:
    x, ds = mean_data
    got = ds.agg(w=bt.mean_ci_half_width(bt.col("x"), confidence=confidence)).to_pydict()["w"][0]
    z = float(scipy_stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    assert got == pytest.approx(z * x.std(ddof=1) / np.sqrt(_N), rel=1e-7)


def test_a_narrower_confidence_gives_a_narrower_interval(rate_data) -> None:
    """The property the two-level lookup broke: the half-width is monotone in the level."""
    _, ds = rate_data
    widths = [
        ds.agg(w=bt.proportion_ci_half_width(bt.col("hit"), confidence=c)).to_pydict()["w"][0]
        for c in (0.5, 0.8, 0.9, 0.95, 0.99)
    ]
    assert widths == sorted(widths)
    assert len(set(widths)) == len(widths), f"levels collapsed onto one multiplier: {widths}"


@pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5, -0.2])
@pytest.mark.parametrize("func", ["mean_ci_half_width", "proportion_ci_half_width"])
def test_a_confidence_outside_the_unit_interval_is_rejected(func, confidence) -> None:
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "hit": [True, False, True, True]})
    column = "x" if func == "mean_ci_half_width" else "hit"
    with pytest.raises(PlanError, match="strictly between 0 and 1"):
        ds.agg(w=getattr(bt, func)(bt.col(column), confidence=confidence)).to_pydict()


# --------------------------------------------------------------------------- #
# The rest of the module, which had no coverage either
# --------------------------------------------------------------------------- #
def test_group_mean_is_the_conditional_mean(two_groups) -> None:
    a, _, ds = two_groups
    got = ds.agg(m=bt.group_mean("x", bt.col("g"))).to_pydict()["m"][0]
    assert got == pytest.approx(float(a.mean()), rel=1e-12)


def test_welch_t_and_df_match_scipy(two_groups) -> None:
    a, b, ds = two_groups
    row = ds.agg(
        t=bt.welch_t_statistic("x", bt.col("g")), df=bt.welch_df("x", bt.col("g"))
    ).to_pydict()
    want = scipy_stats.ttest_ind(a, b, equal_var=False)
    assert row["t"][0] == pytest.approx(float(want.statistic), rel=1e-8)
    assert row["df"][0] == pytest.approx(float(want.df), rel=1e-8)


def test_cohens_d_and_hedges_g_match_their_definitions(two_groups) -> None:
    a, b, ds = two_groups
    row = ds.agg(d=bt.cohens_d("x", bt.col("g")), h=bt.hedges_g("x", bt.col("g"))).to_pydict()
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / pooled
    assert row["d"][0] == pytest.approx(float(d), rel=1e-8)
    # Hedges' g is Cohen's d with the small-sample correction J.
    correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    assert row["h"][0] == pytest.approx(float(d * correction), rel=1e-8)
    assert abs(row["h"][0]) < abs(row["d"][0]), "the correction shrinks the estimate"


def test_proportion_z_statistic_matches_the_closed_form() -> None:
    ds = bt.from_pydict(
        {
            "won": ([True] * 30 + [False] * 70) + ([True] * 50 + [False] * 50),
            "arm": ["a"] * 100 + ["b"] * 100,
        }
    )
    got = ds.agg(
        z=bt.proportion_z_statistic(bt.col("won"), bt.col("arm") == bt.lit("a"))
    ).to_pydict()["z"][0]
    pooled = 0.40
    want = (0.30 - 0.50) / np.sqrt(pooled * (1 - pooled) * (1 / 100 + 1 / 100))
    assert got == pytest.approx(float(want), rel=1e-8)
