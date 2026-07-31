"""The device readings past the five obvious ones, and the distinctions they exist to preserve.

Two properties are asserted throughout, and both are the same property the fault probes hold to.

**Unreadable is never healthy.** A container without the driver, a consumer part without a
counter, and a MIG instance that refuses a query all produce zeros. Every derived answer here
is written so a zero from an unreadable source cannot be mistaken for a measurement of a
healthy device — a fleet that quarantines on that takes itself offline the day a base image
changes, and one that reports "all clear" on it never finds the device that is failing.

**A number is not a diagnosis.** The bottleneck verdicts exist because 80% SM utilization means
opposite things depending on what else was busy, and the tests below pin the *precedence* between
them: a clamped device is reported as clamped even at full utilization, because "compute bound"
sends a reader to buy hardware that would also be clamped.
"""

from __future__ import annotations

import pytest

from batcher._internal.hardware.telemetry import (
    bottleneck,
    clocks,
    energy,
    engines,
    identity,
    memory,
    processes,
    sampler,
    throughput,
)

pytestmark = pytest.mark.unit


class _Handle:
    """One fake device: attribute lookups stand in for NVML's per-device getters.

    Getters taking arguments are keyed by `(name, args)`, which is not expressible as a keyword,
    so those go in the positional mapping.
    """

    def __init__(self, mapping: dict | None = None, **fields) -> None:
        self.__dict__.update(mapping or {})
        self.__dict__.update(fields)


class _FakeNvml:
    """A minimal NVML stand-in.

    Getters are resolved off each handle by name, so a test declares only the fields it cares
    about and every other query raises — which is exactly how a real driver behaves for a part
    that does not implement a counter, and is the case each `readable` flag exists to describe.
    """

    def __init__(self, handles: list[_Handle]) -> None:
        self._handles = handles

    def nvmlDeviceGetCount(self) -> int:
        return len(self._handles)

    def nvmlDeviceGetHandleByIndex(self, index):
        return self._handles[index]

    def __getattr__(self, name):
        if not name.startswith("nvmlDevice"):
            raise AttributeError(name)

        def getter(handle, *args):
            key = (name, args) if args else name
            if key not in handle.__dict__:
                raise RuntimeError(f"{name} not supported")
            return handle.__dict__[key]

        return getter


# --- link throughput ----------------------------------------------------------------------


def _link(**fields) -> throughput.LinkThroughput:
    return throughput.LinkThroughput(index=0, readable=True, **fields)


def test_a_link_that_trained_below_both_ends_is_reported_as_derated():
    full = _link(pcie_gen=5, pcie_gen_max=5, pcie_width=16, pcie_width_max=16)
    narrow = _link(pcie_gen=5, pcie_gen_max=5, pcie_width=8, pcie_width_max=16)
    slow = _link(pcie_gen=3, pcie_gen_max=5, pcie_width=16, pcie_width_max=16)
    assert full.link_derated is False
    assert narrow.link_derated is True
    assert slow.link_derated is True


def test_an_unreported_link_geometry_is_not_a_derate():
    # The silent-capacity-loss check must not fire on a device that simply did not answer:
    # every deployment that cannot read PCIe geometry would otherwise alert on its whole fleet.
    assert _link(pcie_gen=0, pcie_gen_max=0, pcie_width=0, pcie_width_max=0).link_derated is False


def test_pcie_utilization_is_measured_against_one_direction_not_the_sum():
    # PCIe is full duplex. A link saturated in one direction is at 100%, and a link saturated
    # in both is also at 100% of each — summing them would report 200% for healthy hardware.
    gen4_x16 = 0.985e9 * 2 * 16
    both_ways = _link(
        pcie_gen=4,
        pcie_gen_max=4,
        pcie_width=16,
        pcie_width_max=16,
        pcie_tx_bytes_per_s=gen4_x16,
        pcie_rx_bytes_per_s=gen4_x16,
    )
    assert both_ways.pcie_utilization == pytest.approx(1.0)


def test_transfer_bound_needs_a_reading_not_an_absence():
    saturated = _link(pcie_gen=4, pcie_width=16, pcie_tx_bytes_per_s=0.985e9 * 2 * 16)
    unreadable = throughput.LinkThroughput(index=1, readable=False)
    assert throughput.transfer_bound_devices(readings=(saturated, unreadable)) == (saturated,)


def test_peer_resident_says_whether_the_fabric_is_actually_carrying_the_exchange():
    over_host = _link(pcie_tx_bytes_per_s=1e10, nvlink_tx_bytes_per_s=0.0)
    over_fabric = _link(pcie_tx_bytes_per_s=1e8, nvlink_tx_bytes_per_s=1e11)
    assert over_host.peer_resident is False
    assert over_fabric.peer_resident is True


def test_throughput_degrades_to_empty_without_a_driver(monkeypatch):
    monkeypatch.setattr(throughput, "_nvml", lambda: None)
    assert throughput.device_throughput() == ()


def test_a_device_that_refuses_every_query_still_reports_a_record(monkeypatch):
    monkeypatch.setattr(throughput, "_nvml", lambda: _FakeNvml([_Handle()]))
    records = throughput.device_throughput()
    # One record, flagged unreadable: an idle link and an unreadable one are different
    # findings, and the flag is the only thing that separates them.
    assert len(records) == 1
    assert records[0].readable is False


def test_pcie_throughput_is_converted_from_the_kilobytes_nvml_reports(monkeypatch):
    handle = _Handle(
        {
            ("nvmlDeviceGetPcieThroughput", (0,)): 1000,
            ("nvmlDeviceGetPcieThroughput", (1,)): 2000,
        }
    )
    monkeypatch.setattr(throughput, "_nvml", lambda: _FakeNvml([handle]))
    record = throughput.device_throughput()[0]
    assert record.pcie_tx_bytes_per_s == 1_000_000.0
    assert record.pcie_rx_bytes_per_s == 2_000_000.0
    assert record.readable is True


# --- clocks -------------------------------------------------------------------------------


def test_applications_clock_pinning_is_a_configuration_finding_not_a_clamp():
    pinned = clocks.DeviceClocks(index=0, sm_mhz=1400, sm_max_mhz=1980, sm_applications_mhz=1400)
    free = clocks.DeviceClocks(index=0, sm_mhz=1980, sm_max_mhz=1980, sm_applications_mhz=0)
    assert pinned.applications_clock_pinned is True
    assert free.applications_clock_pinned is False


def test_throttle_fraction_measures_the_interval_rather_than_sampling_it():
    before = clocks.DeviceClocks(index=0, violation_ns=(("thermal", 1_000),), reference_ns=0)
    after = clocks.DeviceClocks(
        index=0, violation_ns=(("thermal", 301_000),), reference_ns=1_000_000
    )
    assert clocks.throttle_fraction(before, after) == {"thermal": pytest.approx(0.3)}


def test_a_driver_reload_between_readings_reports_no_violation_rather_than_a_negative_one():
    before = clocks.DeviceClocks(index=0, violation_ns=(("power", 900_000),), reference_ns=1_000)
    after = clocks.DeviceClocks(index=0, violation_ns=(("power", 10),), reference_ns=2_000)
    assert clocks.throttle_fraction(before, after) == {}


def test_a_policy_absent_from_the_later_reading_is_omitted_not_zeroed():
    before = clocks.DeviceClocks(index=0, violation_ns=(("thermal", 0),), reference_ns=0)
    after = clocks.DeviceClocks(index=0, violation_ns=(), reference_ns=1_000_000)
    assert clocks.throttle_fraction(before, after) == {}


def test_clock_limited_needs_a_reading(monkeypatch):
    clamped = clocks.DeviceClocks(index=0, sm_mhz=900, sm_max_mhz=1980, readable=True)
    blind = clocks.DeviceClocks(index=1, sm_mhz=900, sm_max_mhz=1980, readable=False)
    assert clocks.clock_limited_devices(readings=(clamped, blind)) == (clamped,)


# --- fixed-function engines ---------------------------------------------------------------


def test_an_absent_engine_and_an_idle_one_are_distinguishable(monkeypatch):
    # A part with no JPEG block refuses the query; a part with one that nothing is using
    # answers zero. Reporting both as 0.0 would make "this pipeline never reached NVJPG"
    # indistinguishable from "this part has no NVJPG", which are opposite conclusions.
    handle = _Handle(nvmlDeviceGetDecoderUtilization=(0, 1_000_000))
    monkeypatch.setattr(engines, "_nvml", lambda: _FakeNvml([handle]))
    record = engines.device_engines()[0]
    assert record.supported == ("NVDEC",)
    assert record.decoder == 0.0
    assert record.jpeg == 0.0
    assert record.readable is True


def test_hardware_decode_active_is_false_when_nothing_is_readable(monkeypatch):
    monkeypatch.setattr(engines, "_nvml", lambda: None)
    assert engines.hardware_decode_active() is False


def test_a_device_with_idle_codec_blocks_is_capacity_not_a_fault():
    idle = engines.EngineUtilization(index=0, supported=("NVDEC",), readable=True)
    busy = engines.EngineUtilization(index=1, decoder=0.6, supported=("NVDEC",), readable=True)
    assert engines.engine_idle_devices(readings=(idle, busy)) == (idle,)


# --- energy -------------------------------------------------------------------------------


def test_interval_energy_is_matched_by_uuid_not_by_index():
    # A `CUDA_VISIBLE_DEVICES` change between readings reorders the indices. Subtracting by
    # index would then difference one device's counter from another's and report a plausible
    # wrong figure, which is worse than reporting none.
    before = (
        energy.DeviceEnergy(index=0, uuid="A", total_energy_joules=100.0, readable=True),
        energy.DeviceEnergy(index=1, uuid="B", total_energy_joules=200.0, readable=True),
    )
    after = (
        energy.DeviceEnergy(index=0, uuid="B", total_energy_joules=260.0, readable=True),
        energy.DeviceEnergy(index=1, uuid="A", total_energy_joules=140.0, readable=True),
    )
    assert energy.interval_energy_joules(before, after) == pytest.approx(100.0)


def test_a_counter_that_went_backwards_reports_unmeasurable_rather_than_negative():
    before = (energy.DeviceEnergy(index=0, uuid="A", total_energy_joules=500.0, readable=True),)
    after = (energy.DeviceEnergy(index=0, uuid="A", total_energy_joules=3.0, readable=True),)
    assert energy.interval_energy_joules(before, after) is None


def test_a_part_without_the_counter_reports_unmeasurable():
    before = (energy.DeviceEnergy(index=0, uuid="A", readable=True),)
    after = (energy.DeviceEnergy(index=0, uuid="A", readable=True),)
    assert energy.interval_energy_joules(before, after) is None


def test_a_mixed_fleet_does_not_claim_to_be_exactly_metered():
    # Half a total measured and half estimated, reported as measured, is the failure this
    # guards: the figure would be quoted with a confidence none of it has.
    counted = energy.DeviceEnergy(index=0, uuid="A", total_energy_joules=10.0, readable=True)
    uncounted = energy.DeviceEnergy(index=1, uuid="B", readable=True)
    assert energy.energy_counter_available(readings=(counted, counted)) is True
    assert energy.energy_counter_available(readings=(counted, uncounted)) is False
    assert energy.energy_counter_available(readings=()) is False


def test_a_board_capped_below_its_default_is_surfaced_as_configuration():
    capped = energy.DeviceEnergy(
        index=0, enforced_limit_watts=400.0, default_limit_watts=700.0, readable=True
    )
    normal = energy.DeviceEnergy(
        index=1, enforced_limit_watts=700.0, default_limit_watts=700.0, readable=True
    )
    assert capped.derated_fraction == pytest.approx(1 - 400 / 700)
    assert energy.capped_below_default(readings=(capped, normal)) == (capped,)


# --- memory division ----------------------------------------------------------------------


def test_memory_utilization_excludes_the_driver_reserve():
    # The reserve is available to nobody. Counting it as capacity understates how full the
    # device is by exactly the amount that decides whether the next allocation fits.
    record = memory.DeviceMemory(
        index=0, total_bytes=100, reserved_bytes=20, used_bytes=40, free_bytes=40, v2=True
    )
    assert record.utilization == pytest.approx(0.5)


def test_allocatable_holds_back_a_margin_against_total_not_against_free(monkeypatch):
    record = memory.DeviceMemory(
        index=0, total_bytes=1000, used_bytes=800, free_bytes=200, readable=True
    )
    monkeypatch.setattr(memory, "device_memory", lambda: (record,))
    # 10% of *total* is 100, leaving 100 of the 200 free. A margin against free would leave
    # 180 — a margin that shrinks as the device fills, which is backwards: the margin exists
    # to absorb other tenants, and they grow as the device fills.
    assert memory.allocatable_bytes(0, headroom=0.1) == 100


def test_an_unreadable_device_offers_nothing_rather_than_everything(monkeypatch):
    monkeypatch.setattr(memory, "device_memory", lambda: ())
    assert memory.allocatable_bytes(0) == 0


def test_bar1_pressure_needs_an_aperture_reading():
    pressured = memory.DeviceMemory(index=0, bar1_total_bytes=1000, bar1_used_bytes=900)
    unknown = memory.DeviceMemory(index=1, bar1_total_bytes=0, bar1_used_bytes=0)
    assert pressured.bar1_pressured is True
    assert unknown.bar1_pressured is False
    assert memory.bar1_pressured_devices(readings=(pressured, unknown)) == (pressured,)


# --- identity -----------------------------------------------------------------------------


def _identity(**fields) -> identity.DeviceIdentity:
    return identity.DeviceIdentity(index=0, readable=True, **fields)


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((9, 0), "bfloat16"),
        ((8, 0), "bfloat16"),
        ((7, 5), "float16"),
        ((7, 0), "float16"),
        ((6, 1), None),
        ((0, 0), None),
    ],
)
def test_half_precision_follows_the_capability_the_device_reported(
    monkeypatch, capability, expected
):
    monkeypatch.setattr(
        identity, "device_identity", lambda: (_identity(compute_capability=capability),)
    )
    assert identity.half_precision_dtype(0) == expected


def test_each_device_gets_its_own_dtype_on_a_heterogeneous_node(monkeypatch):
    # The bug this replaces: `torch.cuda.get_device_capability()` answers for device 0 and
    # device 0 only, so a box with an Ada part beside a Turing one picked BF16 for both and
    # emulated it on the half that cannot do it.
    ada = identity.DeviceIdentity(index=0, compute_capability=(8, 9), readable=True)
    turing = identity.DeviceIdentity(index=1, compute_capability=(7, 5), readable=True)
    monkeypatch.setattr(identity, "device_identity", lambda: (ada, turing))
    assert identity.half_precision_dtype(0) == "bfloat16"
    assert identity.half_precision_dtype(1) == "float16"


def test_peak_bandwidth_comes_from_the_device_rather_than_a_name_table(monkeypatch):
    record = _identity(memory_bus_width_bits=5120, memory_clock_max_mhz=1593)
    monkeypatch.setattr(identity, "device_identity", lambda: (record,))
    # 1593 MHz over a 5120-bit bus, double rate: roughly 2.0 TB/s, an A100-class figure.
    assert identity.peak_memory_bandwidth_bytes(0) == pytest.approx(2.038e12, rel=0.01)


def test_a_part_that_reported_no_geometry_reports_no_bandwidth(monkeypatch):
    monkeypatch.setattr(identity, "device_identity", lambda: (_identity(),))
    assert identity.peak_memory_bandwidth_bytes(0) == 0.0


def test_a_mig_slice_is_recognized_from_the_name_the_driver_gives_it():
    assert _identity(name="MIG 1g.10gb").mig_slice is True
    assert _identity(name="NVIDIA A100-SXM4-80GB").mig_slice is False


def test_an_unknown_device_index_reports_nothing_rather_than_the_first(monkeypatch):
    monkeypatch.setattr(identity, "device_identity", lambda: (_identity(),))
    assert identity.half_precision_dtype(7) is None
    assert identity.peak_memory_bandwidth_bytes(7) == 0.0


# --- per-process attribution --------------------------------------------------------------


def test_unattributable_utilization_is_none_and_never_zero(monkeypatch):
    # `None` and `0.0` drive opposite decisions: the first means keep the accounting you had,
    # the second means you are using none of the device and may size up against a neighbour
    # you cannot see.
    monkeypatch.setattr(processes, "device_process_utilization", lambda since_us=0: ())
    assert processes.own_utilization(0) is None
    assert processes.device_shared_with_others(0) is None


def test_the_most_recent_sample_wins_rather_than_the_mean(monkeypatch):
    import os

    mine = os.getpid()
    samples = (
        processes.ProcessUtilization(index=0, pid=mine, sm=0.1, timestamp_us=10),
        processes.ProcessUtilization(index=0, pid=mine, sm=0.9, timestamp_us=20),
    )
    monkeypatch.setattr(processes, "device_process_utilization", lambda since_us=0: samples)
    assert processes.own_utilization(0) == pytest.approx(0.9)


def test_a_neighbour_on_the_device_is_reported(monkeypatch):
    import os

    samples = (
        processes.ProcessUtilization(index=0, pid=os.getpid(), sm=0.3, timestamp_us=10),
        processes.ProcessUtilization(index=0, pid=os.getpid() + 1, sm=0.5, timestamp_us=10),
    )
    monkeypatch.setattr(processes, "device_process_utilization", lambda since_us=0: samples)
    assert processes.device_shared_with_others(0) is True


# --- sampling window ----------------------------------------------------------------------


def test_the_accumulator_keeps_shape_not_just_a_mean():
    window = sampler.TelemetrySampler()
    for value in (0.05, 0.95, 0.05, 0.95):
        window.observe(0, "sm", value)
    summary = window.summary(0, "sm")
    assert summary.samples == 4
    assert summary.mean == pytest.approx(0.5)
    assert summary.peak == pytest.approx(0.95)
    assert summary.trough == pytest.approx(0.05)
    assert summary.bursty is True


def test_a_steady_device_and_a_swinging_one_have_the_same_mean_and_different_shapes():
    steady = sampler.TelemetrySampler()
    swinging = sampler.TelemetrySampler()
    for _ in range(4):
        steady.observe(0, "sm", 0.5)
    for value in (0.02, 0.98, 0.02, 0.98):
        swinging.observe(0, "sm", value)
    assert steady.summary(0, "sm").mean == pytest.approx(swinging.summary(0, "sm").mean)
    assert sampler.saturation_shape(steady.summary(0, "sm")) == "steady"
    assert sampler.saturation_shape(swinging.summary(0, "sm")) == "bursty"


def test_an_unsampled_window_reports_no_shape_rather_than_idle():
    # "" and "idle" are different claims: one says we did not look, the other says we looked
    # and there was nothing there.
    assert sampler.saturation_shape(sampler.MetricSummary()) == ""


def test_a_saturated_window_is_named_as_such():
    window = sampler.TelemetrySampler()
    for _ in range(10):
        window.observe(0, "sm", 0.97)
    assert sampler.saturation_shape(window.summary(0, "sm")) == "saturated"


def test_the_accumulator_is_bounded_by_metrics_not_by_samples():
    window = sampler.TelemetrySampler()
    for i in range(10_000):
        window.observe(0, "sm", (i % 100) / 100.0)
    # One entry per (device, metric) pair regardless of how long the sampler ran, which is
    # what makes it safe to leave running for the life of a job.
    assert len(window._totals) == 1
    assert window.summary(0, "sm").samples == 10_000


# --- verdicts -----------------------------------------------------------------------------


def _summary(mean: float, samples: int = 10, **fields) -> sampler.MetricSummary:
    fields.setdefault("peak", mean)
    fields.setdefault("trough", mean)
    fields.setdefault("last", mean)
    fields.setdefault("above_fraction", 1.0 if mean >= 0.8 else 0.0)
    return sampler.MetricSummary(samples=samples, mean=mean, **fields)


def test_a_clamped_device_is_reported_as_clamped_even_at_full_utilization():
    # Precedence matters more than any individual rule here: "compute bound" on a throttled
    # device sends the reader off to buy hardware that would also be throttled.
    verdict = bottleneck.classify_device(
        0, sm=_summary(0.95), throttled=_summary(0.4, above_fraction=0.4)
    )
    assert verdict.verdict == "throttled"


def test_a_saturated_bus_outranks_busy_sms():
    verdict = bottleneck.classify_device(0, sm=_summary(0.5), pcie=_summary(0.9))
    assert verdict.verdict == "transfer_bound"
    assert "resident" in verdict.advice


def test_busy_sms_holding_few_warps_are_occupancy_limited_not_compute_bound():
    verdict = bottleneck.classify_device(0, sm=_summary(0.9), occupancy=_summary(0.15))
    assert verdict.verdict == "occupancy_limited"
    assert verdict.actionable is True


def test_without_dcgm_the_same_device_is_merely_compute_bound():
    verdict = bottleneck.classify_device(0, sm=_summary(0.9))
    assert verdict.verdict == "compute_bound"
    # Compute bound is the one verdict with nothing to fix, so it is not actionable and a
    # fleet report must not lead with it.
    assert verdict.actionable is False


def test_a_quiet_device_with_nothing_else_busy_is_starved():
    verdict = bottleneck.classify_device(0, sm=_summary(0.1), pcie=_summary(0.05))
    assert verdict.verdict == "starved"


def test_a_bursty_device_is_starved_even_though_its_mean_is_mid_range():
    swinging = sampler.MetricSummary(
        samples=10, mean=0.5, peak=0.98, trough=0.02, last=0.5, above_fraction=0.5
    )
    assert bottleneck.classify_device(0, sm=swinging).verdict == "starved"


def test_a_contended_device_is_not_diagnosed_as_starved():
    verdict = bottleneck.classify_device(0, sm=_summary(0.2), shared=True)
    assert verdict.verdict == "contended"


def test_unknowable_sharing_is_not_treated_as_contention():
    # `None` is the containerized case, which is most of them. Treating it as contention
    # would report every device on every Kubernetes fleet as shared.
    verdict = bottleneck.classify_device(0, sm=_summary(0.1), shared=None)
    assert verdict.verdict == "starved"


def test_no_samples_gives_no_verdict():
    verdict = bottleneck.classify_device(0, sm=sampler.MetricSummary())
    assert verdict.verdict == "unknown"
    assert verdict.actionable is False


def test_a_middling_window_with_no_dominant_unit_refuses_to_guess():
    verdict = bottleneck.classify_device(0, sm=_summary(0.5), memory=_summary(0.45))
    assert verdict.verdict == "unknown"


def test_the_fleet_leads_with_the_actionable_finding_not_the_common_one():
    compute = [
        bottleneck.Bottleneck(index=i, verdict="compute_bound", confidence=0.9) for i in range(7)
    ]
    clamped = bottleneck.Bottleneck(index=7, verdict="throttled", confidence=0.4)
    lead = bottleneck.fleet_verdict((*compute, clamped))
    assert lead is not None
    assert lead.index == 7


def test_a_healthy_fleet_has_no_lead_finding():
    verdicts = (bottleneck.Bottleneck(index=0, verdict="compute_bound", confidence=1.0),)
    assert bottleneck.fleet_verdict(verdicts) is None


def test_every_verdict_carries_advice():
    # A verdict a reader cannot act on is a number with a longer name.
    for name in bottleneck.VERDICT_ADVICE:
        assert bottleneck.Bottleneck(index=0, verdict=name).advice
