"""Clock rates against their ceilings, and how long the driver has been holding them down.

`nvml.DeviceTelemetry.throttle_reasons` answers "is this device clamped *right now*", which a
sampled probe reads correctly only if it happens to sample during the clamp. Intermittent
thermal throttling is by construction the case that evades that: a device that spends 30% of a
stage clamped looks unclamped on 70% of samples, and the stage is a third slower than the same
plan on the identical part in the next slot with nothing anywhere to say why.

NVML publishes the cumulative answer, and it is the one worth reading. The *violation status*
counters are monotonic nanosecond totals of time spent below the requested clock, per reason.
Two readings and a subtraction give the fraction of an interval a device spent clamped, which
is a measurement rather than a sample.

The other half of the picture is the ceiling each clock is being held against:

* **Current against maximum** — a device sitting at 60% of its maximum SM clock while drawing
  well under its power limit is not thermally or power constrained; it is *idle inside the
  sample*, which points at the pipeline feeding it rather than at the device.
* **Applications clock against maximum** — an operator can pin application clocks below the
  part's capability, and a locked-down fleet frequently does. That is invisible to every other
  reading here, and it caps throughput permanently rather than transiently.
* **Memory clock** — an HBM device whose memory clock is clamped while its SM clock is not is
  the specific signature of a memory-bandwidth-bound stage hitting a power limit, where the
  useful move is a smaller working set rather than more parallelism.

Every field degrades to `0` when the driver is absent or the query is refused, which is the
normal case for violation counters on consumer parts and inside MIG instances.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher._internal.hardware.nvml import _device_count, _nvml, _read

__all__ = [
    "PERFORMANCE_POLICIES",
    "DeviceClocks",
    "clock_limited_devices",
    "device_clocks",
    "throttle_fraction",
]

#: `nvmlClockType` values: graphics, SM, memory, video. Read positionally because the enum has
#: been stable since NVML shipped and the names on the binding vary by release.
_CLOCK_GRAPHICS, _CLOCK_SM, _CLOCK_MEM, _CLOCK_VIDEO = 0, 1, 2, 3

#: `nvmlPerfPolicyType` values worth accounting, as `(value, label)`. The reliability and
#: board-limit policies are included because they are the two that a *healthy-looking* device
#: fails on: neither raises, neither shows in the instantaneous throttle reasons on most parts,
#: and both cap sustained throughput.
PERFORMANCE_POLICIES: tuple[tuple[int, str], ...] = (
    (0, "power"),
    (1, "thermal"),
    (2, "sync_boost"),
    (3, "board_limit"),
    (4, "low_utilization"),
    (5, "reliability"),
)

#: NVML reports an unknown performance state as 32 rather than as an error.
_PSTATE_UNKNOWN = 32


@dataclass(frozen=True, slots=True)
class DeviceClocks:
    """One device's clock rates, their ceilings, and its cumulative clamp totals.

    Attributes:
        index: NVML device index on this host.
        sm_mhz: Current SM clock.
        sm_max_mhz: Highest SM clock this part supports.
        sm_applications_mhz: SM clock the operator has pinned applications to, `0` when unpinned.
        memory_mhz: Current memory clock.
        memory_max_mhz: Highest memory clock this part supports.
        video_mhz: Current video-engine clock, which paces hardware decode.
        performance_state: NVML P-state, `0` fastest through `15` slowest, `-1` when unknown.
        violation_ns: Cumulative nanoseconds spent below the requested clock, as
            `(label, nanoseconds)` pairs drawn from `PERFORMANCE_POLICIES`. Monotonic within a
            driver load, so a caller subtracts two readings rather than reading one. A policy
            the part does not account for is *absent* rather than zero — those are opposite
            findings, and conflating them reports a device that never throttles.
        reference_ns: The driver's own reference clock at the moment the violations were read,
            in nanoseconds. Two readings of *this* are the denominator for the violation
            deltas; wall-clock is not, because the counters advance on the driver's timebase.
        readable: Whether NVML answered any query.
    """

    index: int
    sm_mhz: int = 0
    sm_max_mhz: int = 0
    sm_applications_mhz: int = 0
    memory_mhz: int = 0
    memory_max_mhz: int = 0
    video_mhz: int = 0
    performance_state: int = -1
    violation_ns: tuple[tuple[str, int], ...] = ()
    reference_ns: int = 0
    readable: bool = False

    @property
    def sm_headroom(self) -> float:
        """Fraction of the part's maximum SM clock left unused, in [0, 1].

        `0.0` when the device is at its ceiling or when either figure is unreported. A device
        with large headroom and high SM utilization is being clamped; a device with large
        headroom and low SM utilization is simply not being fed.
        """
        if self.sm_mhz <= 0 or self.sm_max_mhz <= 0:
            return 0.0
        return max(0.0, 1.0 - self.sm_mhz / self.sm_max_mhz)

    @property
    def applications_clock_pinned(self) -> bool:
        """Whether an operator has pinned applications below the part's maximum SM clock.

        A permanent cap rather than a transient one, and one no instantaneous throttle reading
        reports. It is a fleet-configuration finding, not a device fault.
        """
        return 0 < self.sm_applications_mhz < self.sm_max_mhz

    @property
    def memory_headroom(self) -> float:
        """Fraction of the part's maximum memory clock left unused, in [0, 1]."""
        if self.memory_mhz <= 0 or self.memory_max_mhz <= 0:
            return 0.0
        return max(0.0, 1.0 - self.memory_mhz / self.memory_max_mhz)

    @property
    def throttling_now(self) -> bool:
        """Whether the device is running materially below its clock ceiling.

        A 5% band, because boost clocks fluctuate under any real load and treating every dip as
        a clamp turns this into noise. Something below its ceiling by more than that is being
        held there.
        """
        return self.sm_headroom > 0.05


def _clock(nv, handle, kind: int, getter: str) -> int:
    """One clock reading in MHz, `0` when the query is refused or the getter is absent."""
    fn = getattr(nv, getter, None)
    if fn is None:
        return 0
    return int(_read(lambda: fn(handle, kind), 0) or 0)


def _violations(nv, handle) -> tuple[tuple[tuple[str, int], ...], int]:
    """`(per-policy nanoseconds, reference nanoseconds)` for one device.

    A policy the part does not implement is omitted rather than recorded as zero: zero
    nanoseconds of thermal violation and *no thermal accounting at all* are opposite findings,
    and a caller comparing two readings would see the second as a device that never throttles.
    """
    fn = getattr(nv, "nvmlDeviceGetViolationStatus", None)
    if fn is None:
        return ((), 0)
    totals: list[tuple[str, int]] = []
    reference = 0
    for policy, label in PERFORMANCE_POLICIES:
        status = _read(lambda p=policy: fn(handle, p), None)
        if status is None:
            continue
        violation = getattr(status, "violationTime", None)
        if violation is None:
            continue
        totals.append((label, int(violation)))
        reference = max(reference, int(getattr(status, "referenceTime", 0) or 0))
    return (tuple(totals), reference)


def device_clocks() -> tuple[DeviceClocks, ...]:
    """Clock rates, ceilings, and cumulative clamp totals for every local device.

    Not memoized: the clocks move continuously and the violation counters are the entire point.
    Costs roughly a dozen NVML calls per device, which is a per-stage cadence rather than a
    per-batch one.

    Returns:
        One record per device in NVML index order, empty when NVML is unavailable.
    """
    nv = _nvml()
    if nv is None:
        return ()
    out: list[DeviceClocks] = []
    for index in range(_device_count(nv)):
        handle = _read(lambda i=index: nv.nvmlDeviceGetHandleByIndex(i), None)
        if handle is None:
            continue
        sentinel = object()
        pstate = _read(lambda h=handle: nv.nvmlDeviceGetPerformanceState(h), sentinel)
        violations, reference = _violations(nv, handle)
        sm = _clock(nv, handle, _CLOCK_SM, "nvmlDeviceGetClockInfo")
        out.append(
            DeviceClocks(
                index=index,
                # An SM clock of zero on a part that reports a graphics clock means NVML
                # declined the SM-specific query, not that the SMs are stopped; the graphics
                # clock is the same domain on every part that makes the distinction.
                sm_mhz=sm or _clock(nv, handle, _CLOCK_GRAPHICS, "nvmlDeviceGetClockInfo"),
                sm_max_mhz=_clock(nv, handle, _CLOCK_SM, "nvmlDeviceGetMaxClockInfo")
                or _clock(nv, handle, _CLOCK_GRAPHICS, "nvmlDeviceGetMaxClockInfo"),
                sm_applications_mhz=_clock(
                    nv, handle, _CLOCK_GRAPHICS, "nvmlDeviceGetApplicationsClock"
                ),
                memory_mhz=_clock(nv, handle, _CLOCK_MEM, "nvmlDeviceGetClockInfo"),
                memory_max_mhz=_clock(nv, handle, _CLOCK_MEM, "nvmlDeviceGetMaxClockInfo"),
                video_mhz=_clock(nv, handle, _CLOCK_VIDEO, "nvmlDeviceGetClockInfo"),
                performance_state=(
                    -1 if pstate is sentinel or int(pstate or 0) >= _PSTATE_UNKNOWN else int(pstate)
                ),
                violation_ns=violations,
                reference_ns=reference,
                readable=pstate is not sentinel or bool(violations) or sm > 0,
            )
        )
    return tuple(out)


def throttle_fraction(before: DeviceClocks, after: DeviceClocks) -> dict[str, float]:
    """Fraction of an interval one device spent clamped, per reason.

    The measurement the instantaneous throttle reasons cannot give. Both arguments must be
    readings of the *same* device taken at the ends of the interval of interest; a driver
    reload between them resets the counters, which shows up as a negative delta and is reported
    as no violation rather than as a nonsensical one.

    Args:
        before: Reading taken at the start of the interval.
        after: Reading taken at the end.

    Returns:
        Reason label to fraction of the interval in [0, 1], omitting reasons the device does
        not account for. Empty when the reference clock did not advance, which is what a driver
        that refused the query looks like.
    """
    elapsed = after.reference_ns - before.reference_ns
    if elapsed <= 0:
        return {}
    baseline = dict(before.violation_ns)
    out: dict[str, float] = {}
    for label, total in after.violation_ns:
        delta = total - baseline.get(label, total)
        if delta > 0:
            out[label] = min(1.0, delta / elapsed)
    return out


def clock_limited_devices(
    readings: tuple[DeviceClocks, ...] | None = None,
) -> tuple[DeviceClocks, ...]:
    """Devices running below their clock ceiling, in index order.

    Args:
        readings: Records to inspect, or `None` to read them live.

    Returns:
        The subset that is clamped now or pinned below its maximum by configuration. Empty when
        the fleet is unclamped *or* unreadable — `DeviceClocks.readable` is what separates
        those, and a fleet where it is False must not be reported as healthy.
    """
    records = device_clocks() if readings is None else readings
    return tuple(
        r for r in records if r.readable and (r.throttling_now or r.applications_clock_pinned)
    )
