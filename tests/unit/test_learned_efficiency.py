"""The energy learning loop: Core measures, the conductor folds, Kyber consumes.

The loop is only worth having if each hand-off is honest about what it does not know. An
under-sampled device must read as unmeasured rather than as slow, a modelled figure must never
be learned from (it is the datasheet restated, and folding it teaches the optimizer its own
assumptions back), and two devices must never share a bucket — an average of an H100 and a T4
is right for neither while still overriding the default.
"""

from __future__ import annotations

import pytest

from batcher.core.energy import energy_scope, measure_stage, reset_energy_sampling
from batcher.kyber.gpu import (
    learned_work_per_joule,
    record_measured_efficiency,
    select_device_class,
)
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_sampling():
    """The meter caches a device reading per sampling interval; a faked draw must not leak."""
    reset_energy_sampling()
    yield
    reset_energy_sampling()


_MIN_SAMPLES = 8


def _hub() -> MetadataHub:
    """An in-memory hub, so learning is exercised without touching a user's store."""
    return MetadataHub(InProcessBackend())


def _teach(hub: MetadataHub, device: str, per_joule: float, *, runs: int = _MIN_SAMPLES) -> None:
    for _ in range(runs):
        record_measured_efficiency(hub, device, joules=1000.0, work=int(1000.0 * per_joule))


def test_an_unmeasured_device_reads_as_unknown_not_as_slow() -> None:
    hub = _hub()
    assert learned_work_per_joule(hub, "NVIDIA_H100") is None
    assert learned_work_per_joule(None, "NVIDIA_H100") is None
    assert learned_work_per_joule(hub, None) is None


def test_an_under_sampled_device_is_still_unknown() -> None:
    hub = _hub()
    _teach(hub, "NVIDIA_H100", 50.0, runs=_MIN_SAMPLES - 1)
    assert learned_work_per_joule(hub, "NVIDIA_H100") is None, (
        "efficiency swings with the workload, so a handful of runs is not evidence "
        "about the hardware"
    )
    _teach(hub, "NVIDIA_H100", 50.0, runs=1)
    assert learned_work_per_joule(hub, "NVIDIA_H100") == pytest.approx(50.0)


def test_folding_is_order_independent() -> None:
    a, b = _hub(), _hub()
    for joules, work in ((100.0, 5000), (300.0, 9000), (50.0, 1000)):
        record_measured_efficiency(a, "NVIDIA_H100", joules, work)
    for joules, work in ((50.0, 1000), (300.0, 9000), (100.0, 5000)):
        record_measured_efficiency(b, "NVIDIA_H100", joules, work)
    _teach(a, "NVIDIA_H100", 40.0, runs=_MIN_SAMPLES)
    _teach(b, "NVIDIA_H100", 40.0, runs=_MIN_SAMPLES)
    assert learned_work_per_joule(a, "NVIDIA_H100") == pytest.approx(
        learned_work_per_joule(b, "NVIDIA_H100")
    )


def test_rows_and_tokens_never_share_a_bucket() -> None:
    hub = _hub()
    _teach(hub, "NVIDIA_H100", 10.0)
    for _ in range(_MIN_SAMPLES):
        record_measured_efficiency(hub, "NVIDIA_H100", 1000.0, 900_000, kind="tokens")
    assert learned_work_per_joule(hub, "NVIDIA_H100", kind="rows") == pytest.approx(10.0)
    assert learned_work_per_joule(hub, "NVIDIA_H100", kind="tokens") == pytest.approx(900.0)


def test_devices_never_share_a_bucket() -> None:
    hub = _hub()
    _teach(hub, "NVIDIA_H100", 90.0)
    _teach(hub, "NVIDIA_TESLA_V100", 20.0)
    assert learned_work_per_joule(hub, "NVIDIA_H100") == pytest.approx(90.0)
    assert learned_work_per_joule(hub, "NVIDIA_TESLA_V100") == pytest.approx(20.0)


def test_nonsense_observations_are_dropped_rather_than_stored() -> None:
    hub = _hub()
    for joules, work in ((0.0, 100), (-1.0, 100), (100.0, 0), (100.0, -5)):
        record_measured_efficiency(hub, "NVIDIA_H100", joules, work)
    record_measured_efficiency(hub, "", 100.0, 100)
    record_measured_efficiency(None, "NVIDIA_H100", 100.0, 100)
    assert learned_work_per_joule(hub, "NVIDIA_H100") is None


#: A mixed fleet where a 45 GiB model fits only the two large parts, so there is a choice to
#: make. (When every candidate fits, the answer is "do not pin" whatever the ranking says.)
_MIXED = ["NVIDIA_L40S", "NVIDIA_A100_80G", "NVIDIA_H100"]


def test_the_choice_prefers_what_the_fleet_measured_over_the_datasheet() -> None:
    # The datasheet ranks the H100 well ahead on throughput per watt.
    assert select_device_class(_MIXED, 45.0, prefer_efficiency=True) == "NVIDIA_H100"
    # This fleet measured the opposite for its own workload — a starved H100 does less per
    # joule than a fed A100, and no specification can say that.
    hub = _hub()
    _teach(hub, "NVIDIA_H100", 12.0)
    _teach(hub, "NVIDIA_A100_80G", 31.0)
    assert select_device_class(_MIXED, 45.0, prefer_efficiency=True, hub=hub) == "NVIDIA_A100_80G"


def test_a_partially_measured_fleet_falls_back_to_the_datasheet() -> None:
    # Ranking a measured device against an unmeasured one compares two different things.
    hub = _hub()
    _teach(hub, "NVIDIA_A100_80G", 31.0)
    assert select_device_class(_MIXED, 45.0, prefer_efficiency=True, hub=hub) == "NVIDIA_H100"


def test_the_conductor_folds_only_measured_stages(monkeypatch) -> None:
    import batcher as bt
    from batcher.core import energy as core_energy

    hub = _hub()
    monkeypatch.setattr("batcher.core.runtime.default_hub", lambda: hub)

    # A modelled stage: the datasheet restated, so learning from it teaches the optimizer
    # its own assumptions back.
    with (
        bt.measure_energy(),
        measure_stage("A#1", accelerator_type="NVIDIA_H100", device_count=1) as meter,
    ):
        meter.add_rows(1_000_000)
    assert learned_work_per_joule(hub, "NVIDIA_H100") is None

    monkeypatch.setattr(core_energy, "_draw", lambda: (700.0, 0.9, True))
    reset_energy_sampling()
    for _ in range(_MIN_SAMPLES):
        with (
            bt.measure_energy(),
            measure_stage("A#1", accelerator_type="NVIDIA_H100", device_count=1) as meter,
        ):
            meter.add_rows(1_000_000)
    assert learned_work_per_joule(hub, "NVIDIA_H100") is not None


def test_a_learning_failure_never_breaks_the_block(monkeypatch) -> None:
    import batcher as bt

    def _boom():
        raise RuntimeError("metadata store is down")

    monkeypatch.setattr("batcher.core.runtime.default_hub", _boom)
    with bt.measure_energy() as ledger, measure_stage("A#1", accelerator_type="NVIDIA_H100"):
        pass
    assert len(ledger.stages) == 1, "the run is still reported even when learning fails"


def test_an_empty_run_records_nothing(monkeypatch) -> None:
    import batcher as bt

    def _fail():  # pragma: no cover - the point is that it is never called
        raise AssertionError("an empty ledger must not reach for a hub")

    monkeypatch.setattr("batcher.core.runtime.default_hub", _fail)
    with bt.measure_energy() as ledger:
        pass
    assert ledger.stages == []


def test_the_loop_runs_end_to_end_from_a_measured_scope(monkeypatch) -> None:
    import batcher as bt
    from batcher.core import energy as core_energy

    hub = _hub()
    monkeypatch.setattr("batcher.core.runtime.default_hub", lambda: hub)
    monkeypatch.setattr(core_energy, "_draw", lambda: (700.0, 0.95, True))
    reset_energy_sampling()
    for _ in range(_MIN_SAMPLES):
        with (
            bt.measure_energy(),
            measure_stage("Generate#1", accelerator_type="NVIDIA_H100", device_count=1) as meter,
        ):
            meter.add_tokens(500_000)
    learned = learned_work_per_joule(hub, "NVIDIA_H100", kind="tokens")
    assert learned is not None and learned > 0
    assert learned_work_per_joule(hub, "NVIDIA_H100", kind="rows") is None


def test_a_scope_without_a_meter_is_inert() -> None:
    with energy_scope() as ledger:
        pass
    assert ledger.total_joules == 0.0
