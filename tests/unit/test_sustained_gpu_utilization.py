"""The GPU-packing loop needs a sustained utilization, not a peak.

`recommend_num_gpus` packs a stage onto a fraction of a device only when utilization is
below `_PACK_BELOW`, and `recommend_inflight_depth` deepens submission on the same signal.
Both were fed a reading taken *right after a forward pass* — i.e. sampled at the instant the
device is busiest. On a four-T4 ResNet-50 stage that reported 86% while the device was in
fact busy 13% of the time, so the two levers that exist to fix a starved GPU could never
fire, and the stage kept a whole GPU each while sitting three-quarters idle.

These tests pin the distinction: a bursty device must read as *sustained-idle*, and the
recommendations that follow from it must be the packing ones.
"""

from __future__ import annotations

import time

import pytest

from batcher.ml.gpu import (
    SustainedUtilization,
    recommend_inflight_depth,
    recommend_num_gpus,
)

pytestmark = pytest.mark.unit


class _Device:
    """A scripted utilization source: a burst of `busy` readings, then idle."""

    def __init__(self, readings: list[float]) -> None:
        self._readings = readings
        self.calls = 0

    def __call__(self) -> float | None:
        value = self._readings[min(self.calls, len(self._readings) - 1)]
        self.calls += 1
        return value


def _monitor(monkeypatch, readings: list[float]) -> SustainedUtilization:
    device = _Device(readings)
    monkeypatch.setattr("batcher.ml.gpu.sample_gpu_utilization", device)
    return SustainedUtilization(interval_s=0.005)


def _drain(monitor: SustainedUtilization, seconds: float) -> None:
    monitor.begin_call()
    time.sleep(seconds)
    monitor.end_call()


def test_a_mostly_idle_device_reads_as_idle_not_as_its_peak(monkeypatch) -> None:
    """One reading in ten is a busy burst; the mean must reflect the nine idle ones."""
    monitor = _monitor(monkeypatch, [1.0] + [0.0] * 9)
    try:
        _drain(monitor, 0.3)
        mean = monitor.mean()
        assert mean is not None
        assert mean < 0.5, f"a device idle 90% of the time read as {mean:.2f}"
        assert monitor.peak() == pytest.approx(1.0)  # the peak is still available, and still 1.0
    finally:
        monitor.close()


def test_a_busy_device_reads_as_busy(monkeypatch) -> None:
    monitor = _monitor(monkeypatch, [0.95])
    try:
        _drain(monitor, 0.1)
        assert monitor.mean() == pytest.approx(0.95, abs=0.02)
    finally:
        monitor.close()


def test_a_device_reporting_nothing_yields_none(monkeypatch) -> None:
    """Apple MPS / Cloud TPU / CPU expose no counter; the loop must stay a no-op."""
    monkeypatch.setattr("batcher.ml.gpu.sample_gpu_utilization", lambda: None)
    monitor = SustainedUtilization(interval_s=0.005)
    try:
        _drain(monitor, 0.05)
        assert monitor.mean() is None
        assert recommend_num_gpus(monitor.mean(), 1.0) == 1.0  # declared request stands
    finally:
        monitor.close()


def test_nothing_is_reported_before_any_work_arrives(monkeypatch) -> None:
    """The window opens at the first call, so a model load measures nothing."""
    monitor = _monitor(monkeypatch, [0.0])
    try:
        assert monitor.mean() is None
    finally:
        monitor.close()


def test_the_idle_tail_after_the_last_partition_is_excluded(monkeypatch) -> None:
    """A stage that finished must not be diluted by the wait before its stats are drained."""
    monitor = _monitor(monkeypatch, [0.9])
    try:
        _drain(monitor, 0.1)
        busy = monitor.mean()
        monkeypatch.setattr("batcher.ml.gpu.sample_gpu_utilization", lambda: 0.0)
        time.sleep(0.15)  # the driver gathering results while the device sits idle
        assert monitor.mean() == pytest.approx(busy)
    finally:
        monitor.close()


def test_the_sustained_reading_is_what_turns_the_packing_levers(monkeypatch) -> None:
    """The end-to-end point: peak 1.0 changes nothing, the 13% mean packs and deepens."""
    monitor = _monitor(monkeypatch, [1.0] + [0.0] * 9)
    try:
        _drain(monitor, 0.3)
        sustained = monitor.mean()
        peak = monitor.peak()
        # What the engine used to feed the loop: a peak, which leaves both levers shut.
        assert recommend_num_gpus(peak, 1.0) == 1.0
        assert recommend_inflight_depth(peak, 2) == 2
        # What it feeds now: the stage packs onto a fraction and submits further ahead.
        assert recommend_num_gpus(sustained, 1.0) < 1.0
        assert recommend_inflight_depth(sustained, 2) > 2
    finally:
        monitor.close()
