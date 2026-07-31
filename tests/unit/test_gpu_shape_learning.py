"""The crossover keyed by *workload shape*, and the per-device throughput a fan-out divides by.

Two pipelines on one device share a bucket otherwise and average toward each other, which is
worse than not learning: the pooled value still overrides the config default. These pin that a
shaped bucket wins when it has both lines, that it is ignored when it has only one, and that a
first-time shape keeps exactly the threshold it had.
"""

from __future__ import annotations

import pytest

from batcher.kyber.gpu.adaptive import (
    learned_device_throughput,
    learned_gpu_min_rows,
    record_backend_timing,
    record_device_throughput,
)
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend

pytestmark = pytest.mark.unit

# Eight spread sizes, which is what the fit's confidence gate needs.
ROWS = (
    2_000_000,
    6_000_000,
    12_000_000,
    18_000_000,
    26_000_000,
    40_000_000,
    60_000_000,
    80_000_000,
)


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


def _cpu_ms(rows: int) -> float:
    return 200.0 + 4.0e-4 * rows


def _gpu_ms(rows: int) -> float:
    return 6000.0 + 1.1e-4 * rows


def _narrow_gpu_ms(rows: int) -> float:
    """A shape with a lower fixed cost, so its crossover is earlier: about 20M rows."""
    return 6000.0 + 1.1e-4 * rows


def _wide_gpu_ms(rows: int) -> float:
    """A transfer-bound shape whose device overhead is far higher: about 68M rows."""
    return 20000.0 + 1.1e-4 * rows


def test_two_shapes_on_one_device_get_two_crossovers() -> None:
    """Pooled, they average toward each other — and still override the config default."""
    hub = _hub()
    for rows in ROWS:
        record_backend_timing(hub, "cpu", rows, _cpu_ms(rows), None, "narrow")
        record_backend_timing(hub, "gpu", rows, _narrow_gpu_ms(rows), "H100", "narrow")
        record_backend_timing(hub, "cpu", rows, _cpu_ms(rows), None, "wide")
        record_backend_timing(hub, "gpu", rows, _wide_gpu_ms(rows), "H100", "wide")
    narrow = learned_gpu_min_rows(hub, "H100", "narrow")
    wide = learned_gpu_min_rows(hub, "H100", "wide")
    pooled = learned_gpu_min_rows(hub, "H100")
    assert narrow is not None and wide is not None
    assert narrow < wide
    # Each shape recovers its own crossover to within a percent; the pooled bucket, holding two
    # point-clouds at the same x values, recovers neither — it is a different threshold from
    # both when it fits at all.
    assert narrow == pytest.approx(20_000_000, rel=0.01)
    assert wide == pytest.approx(68_275_862, rel=0.01)
    assert pooled not in (narrow, wide)


def test_recording_a_shape_also_feeds_the_pooled_bucket() -> None:
    """A shape seen once still contributes to the fit every other query reads."""
    hub = _hub()
    for rows in ROWS:
        record_backend_timing(hub, "cpu", rows, _cpu_ms(rows), None, "only-shape")
        record_backend_timing(hub, "gpu", rows, _gpu_ms(rows), "H100", "only-shape")
    assert learned_gpu_min_rows(hub, "H100") is not None


def test_an_unseen_shape_falls_back_rather_than_going_cold() -> None:
    hub = _hub()
    for rows in ROWS:
        record_backend_timing(hub, "cpu", rows, _cpu_ms(rows), None, "seen")
        record_backend_timing(hub, "gpu", rows, _gpu_ms(rows), "H100", "seen")
    assert learned_gpu_min_rows(hub, "H100", "never-seen") == learned_gpu_min_rows(hub, "H100")


def test_a_half_measured_shape_is_discarded_rather_than_crossed_with_the_pool() -> None:
    """A shaped GPU line against a pooled CPU line is two workloads crossed against each other."""
    hub = _hub()
    for rows in ROWS:
        record_backend_timing(hub, "cpu", rows, _cpu_ms(rows))  # pooled CPU only
        record_backend_timing(hub, "gpu", rows, _narrow_gpu_ms(rows), "H100", "gpu-only")
    # The shaped GPU bucket exists; the shaped CPU one does not, so the pooled pair is used.
    assert learned_gpu_min_rows(hub, "H100", "gpu-only") == learned_gpu_min_rows(hub, "H100")


def test_no_shape_behaves_exactly_as_before() -> None:
    hub = _hub()
    for rows in ROWS:
        record_backend_timing(hub, "cpu", rows, _cpu_ms(rows))
        record_backend_timing(hub, "gpu", rows, _gpu_ms(rows), "H100")
    assert learned_gpu_min_rows(hub, "H100", None) == learned_gpu_min_rows(hub, "H100")


def test_throughput_is_learned_from_gpu_runs_alone() -> None:
    """The fan-out has to divide work on a fleet that never runs the CPU engine."""
    hub = _hub()
    for _ in range(3):
        record_device_throughput(hub, "H100", 1_000_000, 0.5)
    assert learned_device_throughput(hub, "H100") == pytest.approx(2_000_000.0)


def test_an_under_sampled_device_reports_no_opinion() -> None:
    hub = _hub()
    record_device_throughput(hub, "H100", 1_000_000, 0.5)
    assert learned_device_throughput(hub, "H100") == 0.0


def test_throughput_weighs_a_long_run_more_than_a_short_one() -> None:
    """Totals rather than a mean of means, which is the weighting a long stage needs."""
    hub = _hub()
    record_device_throughput(hub, "L4", 100, 1.0)  # 100 rows/s
    record_device_throughput(hub, "L4", 100, 1.0)
    record_device_throughput(hub, "L4", 8_000, 2.0)  # 4000 rows/s, and most of the work
    assert learned_device_throughput(hub, "L4") == pytest.approx(8200 / 4.0)


def test_an_unlabelled_device_pools_rather_than_being_dropped() -> None:
    hub = _hub()
    for _ in range(3):
        record_device_throughput(hub, None, 1_000, 1.0)
    assert learned_device_throughput(hub, None) == pytest.approx(1000.0)


def test_learning_never_raises_without_a_hub() -> None:
    record_device_throughput(None, "H100", 1_000, 1.0)
    record_backend_timing(None, "gpu", 1_000, 1.0, "H100", "shape")
    assert learned_device_throughput(None, "H100") == 0.0
    assert learned_gpu_min_rows(None, "H100", "shape") is None


def test_a_zero_or_negative_observation_is_ignored() -> None:
    hub = _hub()
    for _ in range(5):
        record_device_throughput(hub, "H100", 0, 1.0)
        record_device_throughput(hub, "H100", 100, 0.0)
    assert learned_device_throughput(hub, "H100") == 0.0
