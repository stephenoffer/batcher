"""Device fault signals — Xid events, memory row remapping, and what Carbonite does with them.

The property under test throughout is that *unreadable* and *healthy* never collapse into one
another. A container without the host kernel log sees no Xid events, and a driver that refuses
the remapped-row query reports zeros; treating either as evidence of a healthy fleet is
harmless, and treating either as evidence of an unhealthy one takes a cluster offline the day
a base image changes. So the fault path quarantines only on a reading it actually got.
"""

from __future__ import annotations

import os

import pytest

from batcher._internal.hardware.faults import counters, xid
from batcher._internal.hardware.faults.counters import DeviceFaults
from batcher._internal.hardware.nvml import DeviceTelemetry
from batcher.carbonite.accel import health

pytestmark = pytest.mark.unit


# --- Xid ----------------------------------------------------------------------------------


def _kmsg(tmp_path, lines: list[str]) -> str:
    path = tmp_path / "kmsg"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_xid_lines_are_parsed_with_the_device_they_name(tmp_path):
    path = _kmsg(
        tmp_path,
        [
            "6,1,1;NVRM: Xid (PCI:0000:0c:00): 79, pid=1234, GPU has fallen off the bus.",
            "6,2,2;NVRM: Xid (PCI:0000:1a:00): 13, pid=99, Graphics Exception",
        ],
    )
    events = xid.recent_xid_events(path)
    assert [e.code for e in events] == [79, 13]
    assert [e.pci_address for e in events] == ["0000:0c:00.0", "0000:1a:00.0"]
    assert events[0].fatal is True
    assert events[1].fatal is False
    assert "fallen off the bus" in events[0].description


def test_only_fatal_codes_reach_the_quarantine_map(tmp_path):
    path = _kmsg(
        tmp_path,
        [
            "6,1,1;NVRM: Xid (PCI:0000:0c:00): 13, Graphics Exception",
            "6,2,2;NVRM: Xid (PCI:0000:0c:00): 31, MMU fault",
            "6,3,3;NVRM: Xid (PCI:0000:1a:00): 94, contained ECC error",
            "6,4,4;NVRM: Xid (PCI:0000:1a:00): 95, uncontained ECC error",
        ],
    )
    assert xid.xid_fatal(xid.recent_xid_events(path)) == {"0000:1a:00.0": (94, 95)}


def test_an_unattributed_event_is_dropped_not_blamed_on_a_device(tmp_path):
    path = _kmsg(tmp_path, ["6,1,1;NVRM: Xid (PCI:garbage): 79, fell off the bus"])
    events = xid.recent_xid_events(path)
    assert [e.pci_address for e in events] == [""]
    assert xid.xid_fatal(events) == {}


def test_an_unreadable_log_reports_no_events_and_says_so(tmp_path):
    missing = str(tmp_path / "absent")
    assert xid.recent_xid_events(missing) == ()
    assert xid.xid_readable(missing) is False
    # A log that exists is readable, which is what distinguishes "clean" from "blind".
    assert xid.xid_readable(_kmsg(tmp_path, ["6,1,1;nothing to see"])) is True


def test_non_nvrm_kernel_noise_is_ignored(tmp_path):
    path = _kmsg(
        tmp_path,
        [
            "6,1,1;systemd: Started something",
            "6,2,2;audit: type=1400 apparmor",
            "6,3,3;NVRM: Xid (PCI:0000:0c:00): 48, double-bit ECC",
        ],
    )
    assert [e.code for e in xid.recent_xid_events(path)] == [48]


def test_an_unknown_code_is_reported_as_unrecognized_not_guessed():
    assert xid.describe_xid(79) == "GPU has fallen off the bus"
    assert xid.describe_xid(4242) == "unknown Xid 4242"
    assert 4242 not in xid.XID_FATAL


def test_the_documented_set_and_the_fatal_set_agree():
    # A code in one and not the other means an operator reads "fatal" with no explanation, or
    # an explanation for something that never triggers a decision.
    assert set(xid.XID_DESCRIPTIONS) == set(xid.XID_FATAL)


def test_kmsg_history_is_read_without_blocking_on_new_records(tmp_path):
    # The ring buffer replays history then blocks; the probe must return on the first EAGAIN
    # rather than parking the caller. A fifo stands in for that shape.
    path = str(tmp_path / "fifo")
    os.mkfifo(path)
    assert xid.recent_xid_events(path) == ()


# --- NVML fault counters ------------------------------------------------------------------


class _FakeNvml:
    """The NVML surface the counter probe uses, over one scripted device."""

    def __init__(self, *, rows=(0, 0, False, False), replay=0, volatile=0, refuse=False):
        self._rows, self._replay, self._volatile, self._refuse = rows, replay, volatile, refuse

    def nvmlDeviceGetCount(self):
        return 1

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetUUID(self, handle):
        return "GPU-abc"

    def nvmlDeviceGetPciInfo(self, handle):
        return type("Info", (), {"busId": "0000:0C:00.0"})()

    def nvmlDeviceGetRemappedRows(self, handle):
        if self._refuse:
            raise RuntimeError("not supported")
        return self._rows

    def nvmlDeviceGetPcieReplayCounter(self, handle):
        if self._refuse:
            raise RuntimeError("not supported")
        return self._replay

    def nvmlDeviceGetTotalEccErrors(self, handle, error_type, counter_type):
        if self._refuse:
            raise RuntimeError("not supported")
        return self._volatile if counter_type == 0 else self._volatile * 2


def _use(monkeypatch, fake):
    monkeypatch.setattr(counters, "_nvml", lambda: fake)
    monkeypatch.setattr(counters, "_device_count", lambda nv: nv.nvmlDeviceGetCount())


def test_counters_are_read_and_the_pci_address_is_normalized(monkeypatch):
    _use(monkeypatch, _FakeNvml(rows=(3, 1, True, False), replay=17, volatile=2))
    (faults,) = counters.device_faults()
    assert faults.readable is True
    assert faults.pci_address == "0000:0c:00.0"
    assert (faults.remapped_correctable, faults.remapped_uncorrectable) == (3, 1)
    assert faults.remap_pending is True
    assert faults.pcie_replay == 17
    assert (faults.ecc_volatile_uncorrected, faults.ecc_aggregate_uncorrected) == (2, 4)
    assert faults.needs_reset is True
    assert faults.needs_replacement is False
    assert faults.degraded_memory is True


def test_a_driver_refusing_every_query_reports_unreadable_not_healthy(monkeypatch):
    _use(monkeypatch, _FakeNvml(refuse=True))
    (faults,) = counters.device_faults()
    assert faults.readable is False
    assert counters.faulted_devices() == ()


def test_no_driver_reports_nothing(monkeypatch):
    monkeypatch.setattr(counters, "_nvml", lambda: None)
    assert counters.device_faults() == ()
    assert counters.faulted_devices() == ()


def test_a_clean_device_is_not_listed_as_faulted(monkeypatch):
    _use(monkeypatch, _FakeNvml(rows=(0, 0, False, False), replay=0, volatile=0))
    assert counters.device_faults()[0].readable is True
    assert counters.faulted_devices() == ()


# --- Carbonite verdicts -------------------------------------------------------------------


def _healthy(index: int = 0) -> DeviceTelemetry:
    return DeviceTelemetry(
        index=index,
        uuid=f"GPU-{index}",
        temperature_c=45.0,
        memory_used_bytes=1,
        memory_total_bytes=100,
    )


def test_a_remap_failure_quarantines_the_device():
    verdict = health.assess_device(_healthy())
    assert verdict.state == "healthy"
    faults = DeviceFaults(index=0, uuid="GPU-0", remap_failure=True, readable=True)
    out = health.assess_faults(verdict, faults)
    assert out.state == "quarantine"
    assert out.schedulable is False
    assert "row_remap_failed" in out.reasons
    assert out.derate == 0.0


def test_a_pending_reset_degrades_rather_than_quarantines():
    # The device is returning correct results and will until the faulty row is touched; taking
    # it out mid-stage costs more than draining it at the next boundary.
    faults = DeviceFaults(index=0, uuid="GPU-0", remap_pending=True, readable=True)
    out = health.assess_faults(health.assess_device(_healthy()), faults)
    assert out.state == "degraded"
    assert out.schedulable is True
    assert out.derate == pytest.approx(0.75)
    assert health.device_reset_candidates([faults]) == ("GPU-0",)


def test_unreadable_counters_never_change_a_verdict():
    verdict = health.assess_device(_healthy())
    faults = DeviceFaults(index=0, remap_failure=True, readable=False)
    assert health.assess_faults(verdict, faults) == verdict
    assert health.fault_reasons(faults) == ()
    assert health.device_reset_candidates([faults]) == ()


def test_fault_and_thermal_derates_combine_to_the_lowest():
    hot = DeviceTelemetry(
        index=0,
        uuid="GPU-0",
        temperature_c=95.0,
        throttle_reasons=("thermal",),
        memory_used_bytes=1,
        memory_total_bytes=100,
    )
    verdict = health.assess_device(hot)
    assert verdict.derate == pytest.approx(0.5)
    faults = DeviceFaults(index=0, uuid="GPU-0", remapped_uncorrectable=4, readable=True)
    out = health.assess_faults(verdict, faults)
    assert out.derate == pytest.approx(0.5)
    assert set(out.reasons) >= {"thermal_throttle", "row_remap_uncorrectable"}


def test_assess_fleet_joins_faults_by_device_index():
    readings = (_healthy(0), _healthy(1))
    faults = (DeviceFaults(index=1, uuid="GPU-1", remap_failure=True, readable=True),)
    verdicts = health.assess_fleet(readings, faults=faults)
    assert [v.state for v in verdicts] == ["healthy", "quarantine"]
    assert health.schedulable_devices(readings) == (0, 1)


def test_assess_fleet_can_skip_the_second_round_of_nvml_calls():
    # `faults=()` is the hot-path spelling: telemetry only, no counter queries.
    verdicts = health.assess_fleet((_healthy(0),), faults=())
    assert [v.state for v in verdicts] == ["healthy"]


def test_a_remap_failure_can_be_tolerated_by_configuration():
    # An operator running a fleet to exhaustion may prefer a device with no spares left over
    # no device at all. The default is the other way, and both are one flag.
    faults = DeviceFaults(index=0, uuid="GPU-0", remap_failure=True, readable=True)
    lenient = health.HealthThresholds(quarantine_on_remap_failure=False)
    assert health.fault_reasons(faults, lenient) == ()
    verdict = health.assess_faults(health.assess_device(_healthy()), faults, lenient)
    assert verdict.state == "healthy"


def test_a_pending_reset_can_be_made_a_drain():
    faults = DeviceFaults(index=0, uuid="GPU-0", remap_pending=True, readable=True)
    strict = health.HealthThresholds(drain_on_reset_pending=True)
    out = health.assess_faults(health.assess_device(_healthy()), faults, strict)
    assert out.state == "quarantine"
    assert out.schedulable is False


def test_the_configured_thresholds_carry_the_fault_flags():
    th = health.configured_thresholds()
    assert th.quarantine_on_remap_failure is True
    assert th.drain_on_reset_pending is False


def test_a_fatal_xid_quarantines_a_device_everything_else_calls_healthy():
    # The failure this closes: a device that has fallen off the bus still enumerates, still
    # reports a temperature, and still accepts work — so every retry walks onto it in turn.
    readings = (_healthy(0), _healthy(1))
    faults = (
        DeviceFaults(index=0, uuid="GPU-0", pci_address="0000:0c:00.0", readable=True),
        DeviceFaults(index=1, uuid="GPU-1", pci_address="0000:1a:00.0", readable=True),
    )
    verdicts = health.assess_fleet(readings, faults=faults)
    assert [v.state for v in verdicts] == ["healthy", "healthy"]
    quarantined = health.xid_verdicts(verdicts, faults, {"0000:1a:00.0": (79,)})
    assert [v.state for v in quarantined] == ["healthy", "quarantine"]
    assert "xid_79" in quarantined[1].reasons
    assert quarantined[1].derate == 0.0


def test_an_unreadable_kernel_log_never_quarantines(monkeypatch):
    # A container without the host log sees no Xid events, which is indistinguishable from a
    # healthy fleet — so silence must not be a signal.
    monkeypatch.setattr("batcher._internal.hardware.faults.xid_readable", lambda: False)
    monkeypatch.setattr(
        "batcher._internal.hardware.faults.xid_fatal",
        lambda events=None: pytest.fail("read the log after saying it was unreadable"),
    )
    verdicts = (health.HealthVerdict(device_index=0, uuid="GPU-0"),)
    faults = (DeviceFaults(index=0, pci_address="0000:0c:00.0", readable=True),)
    assert health.xid_verdicts(verdicts, faults) == verdicts


def test_an_xid_for_a_device_this_host_does_not_have_changes_nothing():
    verdicts = (health.HealthVerdict(device_index=0, uuid="GPU-0"),)
    faults = (DeviceFaults(index=0, pci_address="0000:0c:00.0", readable=True),)
    assert health.xid_verdicts(verdicts, faults, {"0000:ff:00.0": (79,)}) == verdicts


def test_several_fatal_codes_on_one_device_are_all_recorded():
    verdicts = (health.HealthVerdict(device_index=0, uuid="GPU-0"),)
    faults = (DeviceFaults(index=0, pci_address="0000:0c:00.0", readable=True),)
    out = health.xid_verdicts(verdicts, faults, {"0000:0c:00.0": (48, 95)})
    assert {"xid_48", "xid_95"} <= set(out[0].reasons)


def test_a_corrupting_fault_says_so_beyond_naming_its_code():
    # 48 and 95 are the two codes where the device kept running and returned a *wrong* number
    # rather than none at all, so work already completed on it is suspect. A reason list that
    # said only "xid_95" leaves a caller unable to tell that from an ordinary device fault,
    # and the correct responses differ: one is retried elsewhere, the other fails the run.
    verdicts = (health.HealthVerdict(device_index=0, uuid="GPU-0"),)
    faults = (DeviceFaults(index=0, pci_address="0000:0c:00.0", readable=True),)
    corrupt = health.xid_verdicts(verdicts, faults, {"0000:0c:00.0": (95,)})
    assert "results_untrusted" in corrupt[0].reasons
    # A device that fell off the bus returned nothing, so nothing it produced is suspect.
    lost = health.xid_verdicts(verdicts, faults, {"0000:0c:00.0": (79,)})
    assert "results_untrusted" not in lost[0].reasons


# --- Who an Xid is addressed to --------------------------------------------------------


def test_a_workload_fault_is_not_a_hardware_fault(tmp_path):
    # The expensive mistake this prevents: quarantining a healthy board over a bug in the job
    # that landed on it, then quarantining the next board the retry lands on.
    path = _kmsg(
        tmp_path,
        [
            "6,1,1;NVRM: Xid (PCI:0000:0c:00): 13, Graphics Exception",
            "6,2,2;NVRM: Xid (PCI:0000:0c:00): 31, MMU fault",
            "6,3,3;NVRM: Xid (PCI:0000:1a:00): 79, fell off the bus",
        ],
    )
    events = xid.recent_xid_events(path)
    assert [e.severity for e in events] == ["application", "application", "hardware"]
    assert xid.xid_fatal(events) == {"0000:1a:00.0": (79,)}
    assert xid.xid_application_faults(events) == {"0000:0c:00.0": (13, 31)}


def test_an_unknown_code_is_classified_as_unknown_not_guessed():
    # A future driver release must not be able to quarantine a fleet through a code this
    # build has never seen.
    assert xid.xid_severity(4242) == "unknown"
    assert xid.xid_severity(79) == "hardware"
    assert xid.xid_severity(13) == "application"


def test_an_application_code_still_carries_an_explanation():
    # An operator staring at "Xid 13" on a node that keeps failing needs to know the answer
    # is in their code and not in the rack.
    assert "illegal memory access" in xid.describe_xid(13)
    assert set(xid.XID_APPLICATION) & set(xid.XID_FATAL) == set()


def test_the_two_lists_stay_separate_in_the_fleet_record(monkeypatch):
    from batcher.dist.executors.ray_runtime import fleet_health

    monkeypatch.setattr("batcher.carbonite.accel.assess_fleet", lambda: ())
    monkeypatch.setattr("batcher.carbonite.accel.device_reset_candidates", lambda: ())
    monkeypatch.setattr("batcher.carbonite.accel.device_affinity_summary", dict)
    monkeypatch.setattr("batcher._internal.hardware.fabric.degraded_device_links", lambda: ())
    monkeypatch.setattr(
        "batcher._internal.hardware.faults.xid_application_faults",
        lambda: {"0000:0c:00.0": (13,)},
    )
    record = fleet_health._device_health_on_this_worker()
    assert record["xid_application"] == [13]
    # And it is not a drain reason: the device is fine.
    assert fleet_health.unhealthy_nodes(({"node_id": "a", **record},)) == ()


# --- the sampled window reaching a scheduling decision --------------------------------------
#
# Every other input to a verdict is one NVML reading. Two conditions that cost real throughput
# are invisible to any single reading — a clamp that followed the load and released with it,
# and a neighbour computing on the board — and `telemetry.bottleneck` answers both from a
# window. Those verdicts used to terminate in a report; these pin them to the derate instead.


def _window(verdict: str, *, index: int = 0, confidence: float = 0.9):
    from batcher._internal.hardware.telemetry.bottleneck import Bottleneck

    return Bottleneck(index=index, verdict=verdict, confidence=confidence)


def test_a_contended_device_is_derated_rather_than_quarantined():
    """It is working correctly and delivering less; removing it hands the board to the neighbour."""
    verdict = health.assess_saturation(health.assess_device(_healthy()), _window("contended"))
    assert verdict.state == "degraded"
    assert verdict.derate == 0.5
    assert "contended" in verdict.reasons
    assert verdict.schedulable


def test_a_window_that_says_the_device_is_busy_doing_its_job_changes_nothing():
    """The positive control. `compute_bound` is a device at full output, not a sick one."""
    healthy = health.assess_device(_healthy())
    assert health.assess_saturation(healthy, _window("compute_bound")) == healthy
    assert health.assess_saturation(healthy, _window("memory_bound")) == healthy
    assert health.assess_saturation(healthy, None) == healthy


def test_a_narrow_verdict_does_not_move_the_fleet():
    """Low confidence means the classifier chose between two near-equal limits."""
    healthy = health.assess_device(_healthy())
    assert health.assess_saturation(healthy, _window("contended", confidence=0.1)) == healthy


def test_saturation_never_re_derates_a_device_already_worse_off():
    """The derates fold through `min`, so a clamped *and* contended device is charged once."""
    clamped = DeviceTelemetry(
        index=0,
        uuid="GPU-0",
        temperature_c=45.0,
        memory_used_bytes=1,
        memory_total_bytes=100,
        throttle_reasons=("thermal",),
    )
    before = health.assess_device(clamped)
    assert before.derate == 0.5
    after = health.assess_saturation(before, _window("contended"))
    assert after.derate == 0.5, "0.5 and 0.5 compose to 0.5, not to 0.25"


def test_a_quarantined_device_has_nothing_left_for_a_window_to_say():
    ecc = DeviceTelemetry(
        index=0, uuid="GPU-0", memory_total_bytes=100, memory_used_bytes=1, ecc_uncorrected=99
    )
    condemned = health.assess_device(ecc)
    assert condemned.state == "quarantine"
    assert health.assess_saturation(condemned, _window("contended")) == condemned


def test_assess_fleet_joins_the_window_by_device_index():
    """And a device the window does not cover keeps the verdict its reading earned."""
    verdicts = health.assess_fleet(
        readings=[_healthy(0), _healthy(1)],
        faults=(),
        saturation=(_window("contended", index=1),),
    )
    assert [v.state for v in verdicts] == ["healthy", "degraded"]
    assert [v.derate for v in verdicts] == [1.0, 0.5]


def test_a_fleet_with_no_sampled_window_schedules_exactly_as_before():
    """An operator who never started the dashboard must see the fleet they had."""
    verdicts = health.assess_fleet(readings=[_healthy(0)], faults=(), saturation=())
    assert verdicts[0].state == "healthy"
    assert verdicts[0].derate == 1.0


def test_the_live_path_reaches_the_sampler_rather_than_its_own_except_clause():
    """`_sampled_bottlenecks` is best-effort, which is exactly how a wire dies unnoticed.

    A typo in the lazy import would be caught by its own `except Exception`, logged at debug,
    and return `()` — indistinguishable from "nothing was sampled" at every call site and on
    every dashboard. So assert the call *lands*, not merely that it returns something empty.
    """
    from batcher.observe.accelerators import diagnosis

    reached = []
    real = diagnosis.device_verdicts
    diagnosis.device_verdicts = lambda *a, **k: reached.append(1) or ()
    try:
        assert health._sampled_bottlenecks() == ()
        assert reached, "the lazy import was swallowed instead of resolving"
    finally:
        diagnosis.device_verdicts = real


def test_a_sampler_that_raises_costs_the_window_and_not_the_query():
    """The other half: scheduling must survive a fleet whose telemetry is broken."""
    from batcher.observe.accelerators import diagnosis

    real = diagnosis.device_verdicts

    def _boom(*a, **k):
        raise RuntimeError("no driver")

    diagnosis.device_verdicts = _boom
    try:
        assert health._sampled_bottlenecks() == ()
    finally:
        diagnosis.device_verdicts = real


def test_a_strict_operator_threshold_cannot_turn_contention_into_a_quarantine():
    """`quarantine_below_derate` is the line for a device that is *wrong*, not a busy one.

    An operator tightening it to 0.6 for ECC reasons would otherwise find every contended
    board silently condemned — and a contended board removed from the fleet hands the whole
    device to the neighbour this verdict says is already on it.
    """
    strict = health.HealthThresholds(quarantine_below_derate=0.6)
    healthy = health.assess_device(_healthy(), strict)
    verdict = health.assess_saturation(healthy, _window("contended"))
    assert verdict.state == "degraded"
    assert verdict.schedulable
    assert verdict.derate == 0.5
    # The control: the same threshold still condemns a device that is genuinely degraded.
    hot = DeviceTelemetry(
        index=0,
        uuid="GPU-0",
        temperature_c=45.0,
        memory_used_bytes=99,
        memory_total_bytes=100,
    )
    assert health.assess_device(hot, strict).state == "quarantine"


def test_a_box_with_no_devices_never_reaches_the_observability_layer():
    """The sampler lives in `observe`, whose package import costs ~220 ms the first time.

    Spending that inside a scheduling decision on a machine with no accelerators is 220 ms of
    a sub-second query's budget spent learning that a fleet with no devices has no saturated
    ones. Asserted as "the module was never imported" rather than as a timing, because a
    timing on a shared box measures the neighbours.
    """
    import subprocess
    import sys

    probe = (
        "import sys;"
        "from batcher.carbonite.accel.health import assess_fleet;"
        "assess_fleet(readings=[], faults=());"
        "print('batcher.observe' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", "a device-less fleet pulled in the observability layer"
