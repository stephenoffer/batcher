"""A device actor's VRAM reading is an NVML call, and it used to be one per morsel.

The reading feeds `gpu_vram_stats`, a running maximum the *next* run packs its actors from.
Taken after every forward pass it cost, measured inside the streamed consumer of a GPU
inference pipeline, **12.5 ms against a 71 ms forward** -- 17% of the device stage spent asking
the driver how full the card was, on a quantity that is flat after the model's first batches.

These pin both halves of the fix: the peak still tracks what the device reports, and the
sampler stops calling NVML once it has a recent reading.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors import map as dist_map

pytestmark = pytest.mark.unit


class _Actor:
    """Just the sampling state of `_MapActor`, constructed without Ray or a device."""

    def __init__(self) -> None:
        self._gpu_vram_max = None
        self._gpu_vram_at = float("-inf")

    _sample_gpu_vram = dist_map._MapActor._sample_gpu_vram
    _observe_gpu = dist_map._MapActor._observe_gpu


def _counting(values):
    calls = []

    def sample():
        calls.append(1)
        return values[min(len(calls) - 1, len(values) - 1)]

    return sample, calls


def test_the_first_reading_is_always_taken():
    actor = _Actor()
    sample, calls = _counting([0.25])
    actor._sample_gpu_vram(sample)
    assert len(calls) == 1
    assert actor._gpu_vram_max == 0.25


def test_a_burst_of_morsels_costs_one_nvml_call(monkeypatch):
    """The regression: a thousand forward passes in a second used to be a thousand samples."""
    clock = [1000.0]
    monkeypatch.setattr(dist_map.time, "monotonic", lambda: clock[0])
    actor = _Actor()
    sample, calls = _counting([0.25])
    for _ in range(1000):
        actor._sample_gpu_vram(sample)
    assert len(calls) == 1, f"NVML was called {len(calls)} times inside one instant"


def test_the_peak_still_rises_once_the_interval_has_passed(monkeypatch):
    """The control for the test above: rate-limiting must not freeze the measurement."""
    clock = [1000.0]
    monkeypatch.setattr(dist_map.time, "monotonic", lambda: clock[0])
    actor = _Actor()
    sample, calls = _counting([0.25, 0.80, 0.40])
    actor._sample_gpu_vram(sample)
    clock[0] += dist_map._VRAM_SAMPLE_S * 2
    actor._sample_gpu_vram(sample)
    clock[0] += dist_map._VRAM_SAMPLE_S * 2
    actor._sample_gpu_vram(sample)
    assert len(calls) == 3
    assert actor._gpu_vram_max == 0.80, "a later, smaller reading must not lower the peak"


def test_a_device_that_reports_nothing_leaves_the_peak_unset(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(dist_map.time, "monotonic", lambda: clock[0])
    actor = _Actor()
    actor._sample_gpu_vram(lambda: None)
    assert actor._gpu_vram_max is None
