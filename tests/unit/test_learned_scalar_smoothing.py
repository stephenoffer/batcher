"""How a learned scalar folds in each new observation.

Every subsystem that learns one number per key routes through `record_smoothed_scalar`: the
converged shuffle credit window, a source's read throughput, a partition skew factor. The
step it takes toward a new observation is the whole behaviour of that loop.
"""

from __future__ import annotations

import pytest

from batcher.metadata import MetadataHub
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.metadata.smoothed import load_scalar, record_smoothed_scalar

pytestmark = pytest.mark.unit

_NS, _KEY = "test.smoothed", "k"


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


def test_the_first_observation_is_stored_as_is():
    hub = _hub()
    record_smoothed_scalar(hub, _NS, _KEY, 10.0)
    assert load_scalar(hub, _NS, _KEY) == pytest.approx(10.0)


def test_early_observations_average_rather_than_decaying():
    """A fixed `alpha` leaves the *first* run half the weight forever.

    At the shipped alpha of 0.5, a cold anomalous run would still carry 1/8 of the estimate
    after four observations. The `1/(n+1)` floor makes the early steps a running mean, so
    four equal observations average to exactly their mean regardless of order.
    """
    hub = _hub()
    for v in (100.0, 10.0, 10.0, 10.0):
        record_smoothed_scalar(hub, _NS, _KEY, v)
    assert load_scalar(hub, _NS, _KEY) == pytest.approx(32.5)  # the mean of the four

    reversed_hub = _hub()
    for v in (10.0, 10.0, 10.0, 100.0):
        record_smoothed_scalar(reversed_hub, _NS, _KEY, v)
    assert load_scalar(reversed_hub, _NS, _KEY) == pytest.approx(32.5)


def test_it_settles_into_an_exponential_average():
    """Once `n` passes `1/alpha` the step stops shrinking, so old regimes are forgotten."""
    hub = _hub()
    for _ in range(50):
        record_smoothed_scalar(hub, _NS, _KEY, 10.0)
    for _ in range(10):  # a genuine shift in the workload
        record_smoothed_scalar(hub, _NS, _KEY, 100.0)
    got = load_scalar(hub, _NS, _KEY)
    assert got is not None
    # A running mean over all 60 observations would still read ~25 — it can never forget the
    # 50 stale ones. The exponential phase has a `1/floor`-observation memory, so ten runs of
    # the new regime carry it most of the way there.
    assert got > 60.0, f"the estimate is still anchored to the old regime at {got}"


def test_a_legacy_bare_float_is_still_read_and_folded_into():
    """A hub written by an older build stores a bare float, not a record."""
    hub = _hub()
    hub.put_keyed_param(_NS, _KEY, 20.0)
    assert load_scalar(hub, _NS, _KEY) == pytest.approx(20.0)
    record_smoothed_scalar(hub, _NS, _KEY, 40.0)
    got = load_scalar(hub, _NS, _KEY)
    assert got is not None and 20.0 < got <= 40.0


def test_a_missing_key_reads_as_unlearned():
    assert load_scalar(_hub(), _NS, "never-written") is None
    assert load_scalar(None, _NS, _KEY) is None
