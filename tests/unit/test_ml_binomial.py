"""The exact binomial test.

Checked against SciPy's `binomtest` two-sided p-value, whose convention (sum of every outcome no
more likely than the observed one) this reproduces, across balanced and lopsided samples.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher.ml.stats import binomial_test

pytestmark = pytest.mark.unit

scipy_stats = pytest.importorskip("scipy.stats")


@pytest.mark.parametrize(
    ("successes", "trials", "p0"),
    [(7, 10, 0.5), (3, 20, 0.5), (15, 20, 0.6), (0, 5, 0.3), (50, 100, 0.4)],
)
def test_matches_scipy(successes: int, trials: int, p0: float) -> None:
    ds = bt.from_pydict({"w": [1] * successes + [0] * (trials - successes)})
    got = binomial_test(ds, "w", p0)
    ref = scipy_stats.binomtest(successes, trials, p0).pvalue
    assert got.pvalue == pytest.approx(ref, abs=1e-9)
    assert got.statistic == successes
    assert got.df == trials


def test_rejects_a_lopsided_sample() -> None:
    ds = bt.from_pydict({"w": [1] * 18 + [0] * 2})
    assert binomial_test(ds, "w", 0.5).pvalue < 0.01


def test_keeps_a_fair_sample() -> None:
    rng = np.random.default_rng(0)
    flips = rng.integers(0, 2, 40)
    ds = bt.from_pydict({"w": flips.tolist()})
    assert binomial_test(ds, "w", 0.5).pvalue > 0.05
