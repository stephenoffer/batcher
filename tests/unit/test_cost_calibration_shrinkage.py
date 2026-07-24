"""How a measured cost coefficient blends toward its shipped default.

A cost coefficient is a positive *scale*, so what matters about a measurement is its ratio
to the prior. These tests pin that the blend respects that geometry — the arithmetic blend
is not symmetric under inversion, and its asymmetry biases every coefficient upward whenever
the measurements straddle the prior.
"""

from __future__ import annotations

import pytest

from batcher.kyber.calibration import _shrink

pytestmark = pytest.mark.unit


def test_shrinkage_is_symmetric_under_inversion():
    """A measurement of `r x prior` and one of `prior / r` must blend to reciprocal results.

    The arithmetic blend fails this: at `w = 0.5` it sends 10x to 5.5x and 0.1x to 0.55x,
    and `5.5 x 0.55 = 3.0`, not 1. A coefficient family whose measurements scatter evenly
    around the prior therefore drifts upward with every fit.
    """
    prior, n, strength = 2.0, 10, 10.0  # w = 0.5
    high = _shrink(prior * 10.0, prior, n, strength)
    low = _shrink(prior / 10.0, prior, n, strength)
    assert (high / prior) * (low / prior) == pytest.approx(1.0)


def test_shrinkage_converges_to_the_measurement():
    """With plenty of evidence the prior must stop mattering."""
    prior, measured = 1.0, 10.0
    far = _shrink(measured, prior, 100_000, 10.0)
    assert far == pytest.approx(measured, rel=1e-3)


def test_no_evidence_keeps_the_prior_exactly():
    assert _shrink(10.0, 2.0, 0, 10.0) == 2.0


def test_shrinkage_is_monotone_in_the_evidence():
    """More samples must move the estimate monotonically toward the measurement."""
    prior, measured = 1.0, 8.0
    values = [_shrink(measured, prior, n, 10.0) for n in (1, 2, 5, 20, 100)]
    assert values == sorted(values)
    assert all(prior <= v <= measured for v in values)


def test_a_non_positive_measurement_still_blends():
    """A zero or negative coefficient has no logarithm; the blend must not raise."""
    assert _shrink(0.0, 2.0, 10, 10.0) == pytest.approx(1.0)
