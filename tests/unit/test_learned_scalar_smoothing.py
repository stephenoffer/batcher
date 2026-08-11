"""How a learned scalar folds in each new observation.

Every subsystem that learns one number per key routes through `record_smoothed_scalar`: the
converged shuffle credit window, a source's read throughput, a partition skew factor. The
step it takes toward a new observation is the whole behaviour of that loop.
"""

from __future__ import annotations

import pytest

from batcher.metadata import MetadataHub
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.metadata.smoothed import (
    ScalarEstimate,
    load_scalar,
    load_scalar_estimate,
    record_smoothed_scalar,
)

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


# --- dispersion: how firmly the number is held --------------------------------------


def test_a_single_pathological_run_can_no_longer_dominate():
    """Bounded influence, the property plain exponential smoothing does not have.

    `prior + step x (value - prior)` moves the estimate by an amount unbounded in the
    observation, so one GPU that thermally throttled, or one shuffle measured while the node
    was swapping, drags the learned value by however wrong it was. Clamping the deviation into
    `±k sigma` bounds that at `k sigma x step` no matter how extreme the run — the same
    bounded-influence property `ml.HuberRegressor` gives a fit.
    """
    hub = _hub()
    for _ in range(10):
        record_smoothed_scalar(hub, _NS, _KEY, 100.0)
    record_smoothed_scalar(hub, _NS, _KEY, 100_000.0)
    learned = load_scalar(hub, _NS, _KEY)
    assert learned == pytest.approx(101.0, abs=1.0), (
        "a thousand-fold excursion must move a settled estimate by about a per cent"
    )


def test_a_genuine_regime_change_is_still_followed():
    """The counter-property, and the reason the variance is fed the *unclamped* deviation.

    An estimator that both clamped and learned from the clamped value would narrow its band
    toward zero and lock onto whatever the first runs happened to say. Feeding the true
    deviation widens the band after one real excursion, so the second observation of a new
    regime is barely clamped and the estimate follows.
    """
    hub = _hub()
    for _ in range(10):
        record_smoothed_scalar(hub, _NS, _KEY, 100.0)
    for _ in range(6):
        record_smoothed_scalar(hub, _NS, _KEY, 100_000.0)
    assert load_scalar(hub, _NS, _KEY) > 30_000.0, "a sustained shift must not be rejected"


def test_a_perfectly_steady_history_still_leaves_a_band_to_reject_with():
    """The case the sigma-only guard gets exactly backwards.

    Ten identical observations give a variance of zero, which is an artifact of finite
    sampling rather than a claim of infinite precision. Without a relative floor the band
    collapses and the eleventh observation — the one most obviously an outlier — is admitted
    at full weight.
    """
    hub = _hub()
    for _ in range(10):
        record_smoothed_scalar(hub, _NS, _KEY, 50.0)
    estimate = load_scalar_estimate(hub, _NS, _KEY)
    assert estimate is not None and estimate.variance == pytest.approx(0.0)
    record_smoothed_scalar(hub, _NS, _KEY, 5_000.0)
    assert load_scalar(hub, _NS, _KEY) < 60.0


def test_early_observations_are_never_clamped():
    """Otherwise two similar early runs freeze the estimate permanently.

    A near-zero variance from a tiny sample would pin every later observation to the mean,
    which keeps the variance near zero, and no amount of later evidence could move it.
    """
    hub = _hub()
    record_smoothed_scalar(hub, _NS, _KEY, 1.0)
    record_smoothed_scalar(hub, _NS, _KEY, 1.0)
    record_smoothed_scalar(hub, _NS, _KEY, 1000.0)
    assert load_scalar(hub, _NS, _KEY) > 100.0, "the third run must be free to move it"


def test_a_settled_quantity_reports_itself_stable():
    hub = _hub()
    for _ in range(8):
        record_smoothed_scalar(hub, _NS, _KEY, 42.0)
    estimate = load_scalar_estimate(hub, _NS, _KEY)
    assert estimate is not None
    assert estimate.stable
    assert estimate.coefficient_of_variation == pytest.approx(0.0)


def test_a_scattered_quantity_reports_itself_unstable():
    """The mean of a bimodal population is not an estimate of anything.

    A consumer that warm-starts from it *and switches its own search off* ends up worse than
    one that never learned, which is why `stable` exists as a separate question from `value`.
    """
    hub = _hub()
    for value in (1.0, 900.0, 3.0, 700.0, 5.0, 800.0, 2.0, 950.0):
        record_smoothed_scalar(hub, _NS, _KEY, value)
    estimate = load_scalar_estimate(hub, _NS, _KEY)
    assert estimate is not None and not estimate.stable


def test_too_few_observations_is_never_stable():
    hub = _hub()
    record_smoothed_scalar(hub, _NS, _KEY, 7.0)
    record_smoothed_scalar(hub, _NS, _KEY, 7.0)
    estimate = load_scalar_estimate(hub, _NS, _KEY)
    assert estimate is not None and not estimate.stable


def test_a_record_written_before_dispersion_reports_unknown_not_zero():
    """An unmeasured spread is not a small one.

    Reading a legacy record's absent variance as zero would make every pre-existing learned
    value look perfectly steady, which is the one reading that cannot be justified.
    """
    hub = _hub()
    hub.put_keyed_param(_NS, _KEY, {"value": 12.0, "n": 40.0})
    estimate = load_scalar_estimate(hub, _NS, _KEY)
    assert estimate is not None
    assert estimate.variance is None and estimate.stddev is None
    assert not estimate.stable


def test_an_estimate_centred_on_zero_has_no_relative_spread():
    """Undefined rather than infinite, so a consumer is not handed a nonsense ratio."""
    assert ScalarEstimate(0.0, 10.0, 1.0).coefficient_of_variation is None
    assert not ScalarEstimate(0.0, 10.0, 1.0).stable


def test_a_missing_key_has_no_estimate():
    assert load_scalar_estimate(_hub(), _NS, "never-written") is None
