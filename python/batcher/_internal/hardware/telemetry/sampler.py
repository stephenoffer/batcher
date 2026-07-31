"""Turning instantaneous device readings into a figure that describes a stage.

Every probe in this package returns a snapshot, and a snapshot is the wrong shape for almost
every question anyone asks of it. "Was the GPU busy during this stage" is not answered by the
utilization at the instant the stage ended — that instant is, systematically, the moment the
work drained and the device went idle. Sampling at the ends of a stage measures the ends.

The distribution is also the interesting part, not the mean. A device averaging 50% SM
utilization is either steadily half-fed, which wants a bigger batch, or alternating between
saturated and idle, which wants a deeper prefetch. Those want opposite fixes and have the same
mean, and the peak and the trough are what separate them.

This module is the accumulator: a caller samples it on whatever cadence it likes, and it holds
a bounded, per-device, per-metric summary. It deliberately does **not** own a thread or a clock.
A `_internal` utility that started a background sampler would be starting one in every process
that imports the package — including the short-lived Ray workers where the sampling would cost
more than the stage it measured — so the cadence belongs to the caller that knows the workload,
and `observe` is the one that drives it for a live run.

**Bounded by construction.** The accumulator keeps running aggregates rather than the samples,
so a sampler left running for a day costs the same as one running for a second. That rules out
a median, which needs the samples; the peak, trough, mean, and the fraction of samples above a
threshold are all computable in constant space, and between them they answer the shape question
the median was wanted for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "MetricSummary",
    "TelemetrySampler",
    "saturation_shape",
]

#: SM utilization above which a sample counts as the device being fed. Below saturation on
#: purpose: a device at 80% for a whole stage is not the problem, and a threshold at 95% would
#: classify healthy pipelines as starved.
_FED = 0.8

#: SM utilization below which a sample counts as the device being idle *within* a stage. Not
#: zero, because a device between kernels reports a percent or two of residual activity.
_IDLE = 0.05


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """The shape of one metric on one device over a sampling window.

    Attributes:
        samples: How many readings went in. Zero means every other field is a default.
        mean: Arithmetic mean across the window.
        peak: Highest reading.
        trough: Lowest reading.
        last: Most recent reading, which is what an instantaneous probe would have returned.
        above_fraction: Fraction of readings at or above the sampler's threshold, in [0, 1].
    """

    samples: int = 0
    mean: float = 0.0
    peak: float = 0.0
    trough: float = 0.0
    last: float = 0.0
    above_fraction: float = 0.0

    @property
    def bursty(self) -> bool:
        """Whether the metric swung across most of its range within the window.

        The distinction a mean cannot make. A bursty device is alternating between saturated
        and starved, which is a *feeding* problem — deeper prefetch, more in-flight batches —
        while a steady mid-range device is a *sizing* problem and wants larger work per call.
        """
        return self.samples > 2 and (self.peak - self.trough) > 0.5


@dataclass
class TelemetrySampler:
    """A bounded, running summary of device metrics across a sampling window.

    Not thread-safe, and deliberately not made so: the intended owner is one sampling loop, and
    a lock here would be paid on every sample to protect against a caller that does not exist.
    A caller sampling from several threads should keep one sampler each and merge the summaries.

    Attributes:
        threshold: Value at or above which a sample counts toward `above_fraction`.
    """

    threshold: float = _FED
    _totals: dict[tuple[int, str], list[float]] = field(default_factory=dict)

    def observe(self, index: int, metric: str, value: float) -> None:
        """Fold one reading into the running summary.

        Args:
            index: Device index the reading is for.
            metric: Metric name, free-form; `"sm"`, `"memory"`, `"power_watts"`, and
                `"pcie_utilization"` are the ones the reports use.
            value: The reading.
        """
        key = (index, metric)
        state = self._totals.get(key)
        if state is None:
            # Positional running aggregate, in the order count, sum, peak, trough, last, and
            # the count of samples at or above the threshold. A list rather than a dataclass
            # because this is the one function in the package on a per-sample path.
            self._totals[key] = [1.0, value, value, value, value, float(value >= self.threshold)]
            return
        state[0] += 1.0
        state[1] += value
        state[2] = max(state[2], value)
        state[3] = min(state[3], value)
        state[4] = value
        state[5] += float(value >= self.threshold)

    def observe_telemetry(self, readings) -> None:
        """Fold a whole `nvml.device_telemetry` result in, one metric per useful field.

        Args:
            readings: An iterable of `DeviceTelemetry`, or anything with the same field names.
        """
        for reading in readings:
            self.observe(reading.index, "sm", reading.sm_utilization)
            self.observe(reading.index, "memory", reading.memory_utilization)
            self.observe(reading.index, "power_watts", reading.power_watts)
            self.observe(reading.index, "temperature_c", reading.temperature_c)
            self.observe(reading.index, "throttled", float(bool(reading.throttle_reasons)))

    def observe_throughput(self, readings) -> None:
        """Fold a `telemetry.throughput.device_throughput` result in.

        Args:
            readings: An iterable of `LinkThroughput`.
        """
        for reading in readings:
            self.observe(reading.index, "pcie_utilization", reading.pcie_utilization)
            self.observe(reading.index, "pcie_bytes_per_s", reading.pcie_bytes_per_s)
            self.observe(reading.index, "nvlink_bytes_per_s", reading.nvlink_bytes_per_s)

    def summary(self, index: int, metric: str) -> MetricSummary:
        """The shape of one metric on one device.

        Args:
            index: Device index.
            metric: Metric name as passed to `observe`.

        Returns:
            The summary, all-default when nothing was sampled for that pair.
        """
        state = self._totals.get((index, metric))
        if state is None or state[0] <= 0:
            return MetricSummary()
        count = state[0]
        return MetricSummary(
            samples=int(count),
            mean=state[1] / count,
            peak=state[2],
            trough=state[3],
            last=state[4],
            above_fraction=state[5] / count,
        )

    def devices(self) -> tuple[int, ...]:
        """Device indices this sampler has seen at least one reading for, in order."""
        return tuple(sorted({index for index, _metric in self._totals}))

    def metrics(self, index: int) -> tuple[str, ...]:
        """Metric names sampled for one device, in alphabetical order."""
        return tuple(sorted(metric for i, metric in self._totals if i == index))

    def reset(self) -> None:
        """Discard every accumulated summary, so the next sample starts a new window."""
        self._totals.clear()


def saturation_shape(sm: MetricSummary) -> str:
    """A one-word account of how a device was loaded across a window.

    The classification a report needs, in the vocabulary the fix is written in. `""` when
    nothing was sampled, which is not the same as `"idle"` and must not be rendered as it.

    Args:
        sm: The SM-utilization summary for one device.

    Returns:
        One of `"saturated"` (fed for effectively the whole window; the device is the
        constraint), `"bursty"` (alternating between fed and starved; the pipeline feeding it is
        the constraint), `"steady"` (mid-range throughout; the work per call is too small),
        `"idle"` (never meaningfully busy), or `""` when there were no samples.
    """
    if sm.samples == 0:
        return ""
    if sm.above_fraction >= 0.9:
        return "saturated"
    if sm.peak <= _IDLE:
        return "idle"
    if sm.bursty:
        return "bursty"
    return "steady"
