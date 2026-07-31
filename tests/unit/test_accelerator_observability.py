"""Watching devices over a run, and annotating the run for a device profiler.

Two surfaces, one shared discipline: neither may cost anything on a host with no accelerator,
and neither may fail a process. A scrape endpoint that raises takes monitoring down at exactly
the moment monitoring is wanted, and a profiler annotation that raises turns a diagnostic into
an outage. So every reader is guarded and every absence reads as an empty answer.

The other property under test is the layer boundary: `observe` exports *facts* through the
scrape endpoint and *verdicts* only through a report a person reads. A verdict on a metrics
endpoint would put observability on the wrong side of the independence contract, because
deciding what a device's numbers mean is a subsystem's job.
"""

from __future__ import annotations

import pytest

from batcher._internal.hardware.telemetry.sampler import TelemetrySampler
from batcher._internal.instrument import nvtx, ranges
from batcher.observe import metrics
from batcher.observe.accelerators import diagnosis, gauges, series

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _detached_collector():
    """Detach the metrics collector afterwards, since it is a process-wide bus subscriber.

    Rendering the exposition text attaches one, and a subscriber left attached tells the engine
    that per-query profiles are being consumed — which silently changes behavior for every later
    test in the session, not just in this file. The same fixture guards the node-condition
    suite next door, for the same reason.

    Only an attachment *this* test caused is undone, and it is undone through `stop_metrics` so
    the module can attach again: calling the raw detach handle leaves it set, and `start_metrics`
    then treats the collector as still attached, which silences every later test.
    """
    attached_before = metrics._detach is not None
    yield
    if not attached_before:
        metrics.stop_metrics()


# --- gauges -------------------------------------------------------------------------------


class _Throughput:
    index = 0
    pcie_tx_bytes_per_s = 4.0e9
    pcie_rx_bytes_per_s = 1.0e9
    pcie_bytes_per_s = 5.0e9
    nvlink_bytes_per_s = 0.0
    pcie_utilization = 0.63
    pcie_gen = 4
    pcie_width = 8
    link_derated = True


def test_the_deep_series_reach_the_prometheus_exposition(monkeypatch):
    monkeypatch.setattr(
        "batcher._internal.hardware.telemetry.throughput.device_throughput",
        lambda: (_Throughput(),),
    )
    lines = "\n".join(gauges.link_gauges())
    assert 'batcher_device_pcie_utilization_ratio{device="0"} 0.63' in lines
    assert 'batcher_device_pcie_link_derated{device="0"} 1' in lines
    # Help text on every series: a bare name in a dashboard is a series nobody knows how to
    # act on, which is the same standard the node conditions already hold to.
    assert lines.count("# HELP") == len(gauges._LINK_GAUGES)
    assert lines.count("# TYPE") == len(gauges._LINK_GAUGES)


def test_a_zero_reading_is_emitted_rather_than_dropped():
    class _Idle(_Throughput):
        pcie_tx_bytes_per_s = 0.0
        pcie_rx_bytes_per_s = 0.0
        pcie_utilization = 0.0
        link_derated = False

    lines = gauges._emit("device", gauges._LINK_GAUGES, (_Idle(),))
    # A series that disappears when the device goes quiet breaks every rate and average built
    # on it. Zero is a real value here and is exported as one.
    assert any("pcie_utilization_ratio" in line and line.endswith("0.0") for line in lines)


def test_a_host_with_no_readable_device_emits_nothing_at_all():
    # Distinct from the case above: no series is different from a series at zero, and it is
    # what a scrape config should see on a CPU-only node.
    assert gauges._emit("device", gauges._LINK_GAUGES, ()) == []


def test_a_source_that_raises_contributes_no_series_rather_than_failing_the_scrape(monkeypatch):
    def _boom():
        raise RuntimeError("driver went away")

    monkeypatch.setattr("batcher._internal.hardware.telemetry.throughput.device_throughput", _boom)
    assert gauges.link_gauges() == []
    # And the whole endpoint still renders, which is the property that actually matters.
    assert "batcher_queries_total" in metrics.prometheus_text()


def test_the_gauges_are_labelled_only_by_device(monkeypatch):
    # Cardinality has to be bounded by hardware rather than by run length, or a long job
    # quietly grows the series count until the scrape times out.
    lines = gauges._emit("device", gauges._LINK_GAUGES, (_Throughput(),))
    for line in lines:
        if line.startswith("batcher_"):
            assert line.count("{") <= 1
            assert "device=" in line


# --- sampling window ----------------------------------------------------------------------


def test_starting_twice_does_not_start_a_second_sampler():
    try:
        assert series.start_device_series() is True
        # Nesting a report inside a running dashboard must not reset the dashboard's window.
        assert series.start_device_series() is False
        assert series.sampling_active() is True
    finally:
        series.reset_device_series()


def test_the_window_survives_a_stop_so_a_reader_does_not_race_the_last_sample():
    try:
        series.start_device_series()
        series.stop_device_series()
        assert series.device_window() is not None
    finally:
        series.reset_device_series()


def test_a_reset_discards_the_window():
    series.start_device_series()
    series.reset_device_series()
    assert series.device_window() is None
    assert series.sampling_active() is False


def test_stopping_a_sampler_that_never_started_is_not_an_error():
    series.reset_device_series()
    series.stop_device_series()


def test_sampling_follows_the_config_in_both_directions():
    import dataclasses

    from batcher.config import Config, config_context
    from batcher.observe import control

    base = Config()
    on = base.replace(accelerator=dataclasses.replace(base.accelerator, telemetry_sampling=True))
    try:
        # `_applied` is the per-process short-circuit `ensure_sinks` uses; clearing it is what
        # a config change would otherwise do, and the test drives the transition directly.
        control._applied = None
        with config_context(on):
            control.ensure_sinks()
            assert series.sampling_active() is True
        control._applied = None
        control.ensure_sinks()
        # A `config_context` that turns sampling off has to actually stop the thread, or a
        # block that enabled it leaves one hitting NVML for the life of the process.
        assert series.sampling_active() is False
    finally:
        control._applied = None
        series.reset_device_series()


def test_stopping_the_dashboard_stops_the_sampler():
    import batcher as bt

    try:
        series.start_device_series()
        bt.stop_ui()
        assert series.sampling_active() is False
        # The window survives, so a report taken after the dashboard closes still has data.
        assert series.device_window() is not None
    finally:
        series.reset_device_series()


def test_a_failing_source_does_not_kill_the_sampling_thread(monkeypatch):
    def _boom():
        raise RuntimeError("NVML went away mid-run")

    monkeypatch.setattr("batcher._internal.hardware.nvml.device_telemetry", _boom)
    window = TelemetrySampler()
    # A device that disappears mid-run is a real condition, reported by the samples stopping
    # rather than by an exception on a thread nobody is waiting on.
    series._sample_once(window)
    assert window.devices() == ()


# --- diagnosis ----------------------------------------------------------------------------


def test_no_window_says_so_rather_than_reporting_a_healthy_fleet():
    series.reset_device_series()
    assert "no sampling window" in diagnosis.format_bottleneck_report()


def test_a_sampled_window_reports_a_verdict_and_a_fix():
    window = TelemetrySampler()
    for _ in range(10):
        window.observe(0, "sm", 0.95)
        window.observe(0, "pcie_utilization", 0.02)
        window.observe(0, "power_watts", 640.0)
    report = diagnosis.format_bottleneck_report(window)
    assert "saturated" in report
    assert "compute_bound" in report
    # Compute bound is the one verdict with nothing to fix, so it must not be promoted to a
    # lead finding — a report that leads with it sends the reader nowhere.
    assert "lead finding" not in report


def test_the_lead_finding_is_the_actionable_one():
    window = TelemetrySampler()
    for _ in range(10):
        window.observe(0, "sm", 0.2)
        window.observe(0, "pcie_utilization", 0.95)
    report = diagnosis.format_bottleneck_report(window)
    assert "transfer_bound" in report
    assert "lead finding" in report


def test_occupancy_is_only_consulted_when_it_was_actually_sampled():
    # Without DCGM every occupancy summary is empty. Passing it through anyway would classify
    # every busy device as occupancy limited, which is the most expensive wrong answer here.
    window = TelemetrySampler()
    for _ in range(10):
        window.observe(0, "sm", 0.9)
    verdicts = diagnosis.device_verdicts(window)
    assert [v.verdict for v in verdicts] == ["compute_bound"]


def test_a_device_with_no_sm_samples_is_left_out_of_the_verdicts():
    window = TelemetrySampler()
    window.observe(3, "power_watts", 70.0)
    assert diagnosis.device_verdicts(window) == ()


def test_the_saturation_line_is_empty_for_an_unsampled_device():
    assert diagnosis.format_saturation_line(0, TelemetrySampler()) == ""


def test_the_snapshot_flags_an_unsampled_window_rather_than_reporting_zeros():
    snapshot = diagnosis.window_snapshot(TelemetrySampler())
    # An all-zeros panel and a fleet nobody watched look identical without this flag, and one
    # of them is a healthy fleet.
    assert snapshot == {"sampled": False, "devices": {}}


def test_the_snapshot_carries_the_shape_a_scrape_cannot():
    window = TelemetrySampler()
    for value in (0.02, 0.98, 0.02, 0.98, 0.02, 0.98):
        window.observe(0, "sm", value)
        window.observe(0, "power_watts", 500.0)
    snapshot = diagnosis.window_snapshot(window)
    assert snapshot["sampled"] is True
    device = snapshot["devices"]["0"]
    # The mean is mid-range and the device was never half-loaded; the shape is the finding,
    # and repeated instantaneous scrapes cannot recover it.
    assert device["sm_mean"] == pytest.approx(0.5)
    assert device["shape"] == "bursty"
    assert device["verdict"] == "starved"
    assert device["advice"]
    assert device["samples"] == 6


def test_the_snapshot_reaches_the_metrics_document():
    snapshot = metrics.metrics_snapshot()
    # A consumer should not have to know whether this process has devices to parse the
    # document, so the key is always present.
    assert "window" in snapshot["gpu"]
    assert set(snapshot["gpu"]["window"]) == {"sampled", "devices"}


# --- findings -----------------------------------------------------------------------------


def test_no_sampling_window_produces_no_device_finding():
    from batcher.observe.insights.devices import device_bottleneck

    series.reset_device_series()
    # Silence is the right answer without a window. A finding derived from one instantaneous
    # reading taken when the query finished would describe an idle fleet, confidently.
    assert device_bottleneck({}, [], 0.0) == []


def test_a_healthy_fleet_produces_no_device_finding(monkeypatch):
    from batcher.observe.accelerators import diagnosis
    from batcher.observe.insights.devices import device_bottleneck

    window = TelemetrySampler()
    for _ in range(10):
        window.observe(0, "sm", 0.95)
    monkeypatch.setattr(series, "device_window", lambda: window)
    monkeypatch.setattr(diagnosis, "device_window", lambda: window, raising=False)
    # Compute bound is not actionable: the device is doing its job and the only remedy is more
    # hardware, so a findings list must not carry it.
    assert device_bottleneck({}, [], 0.0) == []


def test_one_finding_is_reported_for_a_whole_fleet(monkeypatch):
    from batcher.observe.insights.devices import device_bottleneck

    window = TelemetrySampler()
    for index in range(4):
        for _ in range(10):
            window.observe(index, "sm", 0.2)
            window.observe(index, "pcie_utilization", 0.95)
    monkeypatch.setattr(series, "device_window", lambda: window)
    found = device_bottleneck({}, [], 0.0)
    # Four devices with the same limit is one thing to do, not four findings burying it.
    assert len(found) == 1
    assert found[0].rule == "device-bottleneck"
    assert found[0].detail["verdict"] == "transfer_bound"
    assert found[0].detail["devices_affected"] == 4
    assert found[0].action


def test_a_derated_link_is_reported_as_a_node_fault(monkeypatch):
    from batcher._internal.hardware.telemetry.throughput import LinkThroughput
    from batcher.observe.insights.devices import derated_host_link

    narrow = LinkThroughput(
        index=2, pcie_gen=3, pcie_gen_max=5, pcie_width=8, pcie_width_max=16, readable=True
    )
    healthy = LinkThroughput(
        index=0, pcie_gen=5, pcie_gen_max=5, pcie_width=16, pcie_width_max=16, readable=True
    )
    monkeypatch.setattr(
        "batcher._internal.hardware.telemetry.throughput.device_throughput",
        lambda: (healthy, narrow),
    )
    found = derated_host_link({}, [], 0.0)
    assert len(found) == 1
    # Critical, and phrased as a node to drain: no pipeline lever recovers a link at half width.
    assert found[0].severity == "critical"
    assert found[0].detail["devices"] == [2]
    assert "drain" in found[0].action


def test_an_unreadable_link_is_not_reported_as_derated(monkeypatch):
    from batcher._internal.hardware.telemetry.throughput import LinkThroughput
    from batcher.observe.insights.devices import derated_host_link

    monkeypatch.setattr(
        "batcher._internal.hardware.telemetry.throughput.device_throughput",
        lambda: (LinkThroughput(index=0, readable=False),),
    )
    # Every deployment that cannot read PCIe geometry would otherwise raise a critical finding
    # against its whole fleet.
    assert derated_host_link({}, [], 0.0) == []


def test_a_probe_that_raises_produces_no_finding(monkeypatch):
    from batcher.observe.insights.devices import derated_host_link

    def _boom():
        raise RuntimeError("driver went away")

    monkeypatch.setattr("batcher._internal.hardware.telemetry.throughput.device_throughput", _boom)
    assert derated_host_link({}, [], 0.0) == []


# --- profiler annotation ------------------------------------------------------------------


def test_a_range_is_a_no_op_without_a_backend(monkeypatch):
    monkeypatch.setattr(nvtx, "_backend", lambda: None)
    assert nvtx.nvtx_backend() == ""
    with nvtx.device_range("Aggregate#1"):
        pass


def test_a_backend_that_raises_is_dropped_rather_than_retried(monkeypatch):
    calls = []

    def _push(label):
        calls.append(label)
        raise RuntimeError("profiler detached")

    monkeypatch.setattr(nvtx, "_backend", lambda: ("fake", _push, lambda: None))
    nvtx._DISABLED.clear()
    nvtx.push_range("one")
    nvtx.push_range("two")
    try:
        # An unbalanced push and pop under a broken backend corrupts the whole capture, so the
        # first failure disables annotation instead of being retried per range.
        assert calls == ["one"]
        assert nvtx.nvtx_backend() == ""
    finally:
        # Only the disable flag needs clearing here: monkeypatch restores `_backend` itself,
        # and calling the public reset while it is still patched would try to clear a cache
        # the stand-in does not have.
        nvtx._DISABLED.clear()


def test_a_range_closes_on_the_exception_path(monkeypatch):
    events = []
    monkeypatch.setattr(
        nvtx,
        "_backend",
        lambda: (
            "fake",
            lambda label: events.append(("push", label)),
            lambda: events.append("pop"),
        ),
    )
    nvtx._DISABLED.clear()
    with pytest.raises(ValueError), nvtx.device_range("Sort#2"):
        raise ValueError("stage failed")
    # A leaked push does not raise; it produces a range that swallows the rest of the capture.
    assert events == [("push", "Sort#2"), "pop"]


def test_operator_ranges_cost_nothing_when_profiling_is_off(monkeypatch):
    monkeypatch.setattr(ranges, "profiling_enabled", lambda: False)
    with ranges.operator_range("HashJoin", 3):
        pass
    with ranges.time_device_work("HashJoin#3") as timing:
        pass
    # No events recorded means no figure, and `None` is the honest answer rather than zero.
    assert timing.resolve() is None
    assert timing.milliseconds is None


def test_device_timing_reports_nothing_until_it_is_resolved(monkeypatch):
    monkeypatch.setattr(ranges, "profiling_enabled", lambda: True)
    monkeypatch.setattr(ranges, "_cuda_events", lambda: ())
    with ranges.time_device_work("Scan#0") as timing:
        pass
    # Without CUDA events there is no device time to report, and wall-clock is not a
    # substitute across an asynchronous launch.
    assert timing.resolve() is None


def test_the_operator_label_carries_the_node_id(monkeypatch):
    labels = []
    monkeypatch.setattr(ranges, "profiling_enabled", lambda: True)
    monkeypatch.setattr(
        nvtx, "_backend", lambda: ("fake", lambda label: labels.append(label), lambda: None)
    )
    nvtx._DISABLED.clear()
    with ranges.operator_range("HashJoin", 3):
        pass
    with ranges.operator_range("Scan"):
        pass
    # A plan with four joins produces four identically named bands otherwise, and telling
    # them apart is usually the reason the capture was taken.
    assert labels == ["HashJoin#3", "Scan"]
