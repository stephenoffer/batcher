"""Who on a shared device is using it — per-process SM, memory, and codec utilization.

Every utilization figure elsewhere in this package is the *device's*. On a device this process
has to itself that is the same thing as this process's utilization, and on a shared device it is
not remotely the same thing, which is the normal case on rented capacity: a fractional GPU
request, an MPS deployment, a colocated inference server, and another tenant's job all produce a
device reading that has nothing to do with the caller's own work.

That breaks the two decisions that read utilization hardest, in opposite directions:

* **Autobatching reads high and backs off.** A device at 90% from a neighbour looks saturated,
  so the batch size shrinks, so this process gets *less* of the device, so the reading stays
  high. The feedback loop is stable at the wrong answer and nothing about it looks like a bug.
* **Scheduling reads low and packs.** A device at 15% because this process is starved looks
  like spare capacity, so another worker lands on it, so both are starved.

`nvmlDeviceGetProcessUtilization` answers per PID and is the only source that can. It is a
*sampling* interface rather than a counter: it returns the samples the driver has buffered since
a timestamp the caller supplies, and the buffer is short. A caller asking rarely gets a window
much narrower than the gap between its calls, which is fine for a rate and wrong for a total —
so this module reports rates and refuses to present them as totals.

**Everything here is refused far more often than it is answered.** Per-process attribution
requires the driver to see the process, which it does not across a PID namespace boundary, and
that is the common containerized case. An empty result is "we cannot see", never "nobody is
using it", and every caller here is written so those cannot be confused.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher._internal.hardware.nvml import _device_count, _nvml, _read

__all__ = [
    "ProcessUtilization",
    "device_process_utilization",
    "device_shared_with_others",
    "own_utilization",
]


@dataclass(frozen=True, slots=True)
class ProcessUtilization:
    """One process's share of one device over the driver's recent sample window.

    Attributes:
        index: NVML device index the process is running on.
        pid: Operating-system process id, as the *driver* sees it. Inside a PID namespace that
            is not the id this process knows itself by, which is why `own_utilization` matches
            on memory attribution rather than on the number alone.
        sm: Fraction of the window this process had SMs busy, in [0, 1].
        memory: Fraction of the window this process had device memory busy, in [0, 1].
        encoder: Fraction of the window this process used NVENC, in [0, 1].
        decoder: Fraction of the window this process used NVDEC, in [0, 1].
        timestamp_us: The driver's timestamp for the sample, in microseconds. Passing the
            highest one back on the next call is what advances the window instead of re-reading
            the same samples.
    """

    index: int
    pid: int
    sm: float = 0.0
    memory: float = 0.0
    encoder: float = 0.0
    decoder: float = 0.0
    timestamp_us: int = 0

    @property
    def active(self) -> bool:
        """Whether the process did any measurable work on the device in the window."""
        return max(self.sm, self.memory, self.encoder, self.decoder) > 0.0


def _samples(nv, handle, index: int, since_us: int) -> tuple[ProcessUtilization, ...]:
    """Per-process samples for one device since a driver timestamp, empty when refused.

    NVML raises `NotFound` rather than returning an empty list when no samples exist in the
    window, which is a routine condition on an idle device and not an error; `_read` maps it to
    the same empty answer as a genuine refusal. That conflation is deliberate — both mean the
    caller learned nothing, and neither is evidence the device is unused.
    """
    fn = getattr(nv, "nvmlDeviceGetProcessUtilization", None)
    if fn is None:
        return ()
    entries = _read(lambda: fn(handle, since_us), None)
    if not entries:
        return ()
    out: list[ProcessUtilization] = []
    for entry in entries:
        pid = int(getattr(entry, "pid", 0) or 0)
        if pid <= 0:
            continue
        out.append(
            ProcessUtilization(
                index=index,
                pid=pid,
                sm=min(1.0, float(getattr(entry, "smUtil", 0) or 0) / 100.0),
                memory=min(1.0, float(getattr(entry, "memUtil", 0) or 0) / 100.0),
                encoder=min(1.0, float(getattr(entry, "encUtil", 0) or 0) / 100.0),
                decoder=min(1.0, float(getattr(entry, "decUtil", 0) or 0) / 100.0),
                timestamp_us=int(getattr(entry, "timeStamp", 0) or 0),
            )
        )
    return tuple(out)


def device_process_utilization(since_us: int = 0) -> tuple[ProcessUtilization, ...]:
    """Per-process utilization samples across every local device.

    Args:
        since_us: Driver timestamp in microseconds to read forward from. `0` asks for
            everything still buffered, which is the right call the first time and the wrong one
            afterwards: it re-reads samples already seen, so a caller tracking a series should
            pass back the highest `timestamp_us` it got.

    Returns:
        One record per process per device, in device order. Empty when NVML is unavailable,
        when the query is refused, and genuinely when nothing ran — those are not
        distinguishable here, and a caller that needs to tell them apart should compare against
        `nvml.device_processes`, which reports residency rather than activity.
    """
    nv = _nvml()
    if nv is None:
        return ()
    out: list[ProcessUtilization] = []
    for index in range(_device_count(nv)):
        handle = _read(lambda i=index: nv.nvmlDeviceGetHandleByIndex(i), None)
        if handle is None:
            continue
        out.extend(_samples(nv, handle, index, since_us))
    return tuple(out)


def own_utilization(index: int, since_us: int = 0) -> float | None:
    """This process's own share of one device's SMs, or `None` when unattributable.

    The correction autobatching and placement both need on a shared device. `None` is a
    first-class answer and must not be treated as zero: it means the driver would not attribute
    the work, which is the containerized case, and a caller that reads it as "we are using
    nothing" will size itself to compete with a neighbour it cannot see.

    Args:
        index: NVML device index.
        since_us: Driver timestamp to read forward from, as in `device_process_utilization`.

    Returns:
        Fraction in [0, 1], or `None` when no sample was attributed to this process.
    """
    import os

    mine = os.getpid()
    samples = [
        s for s in device_process_utilization(since_us) if s.index == index and s.pid == mine
    ]
    if not samples:
        return None
    # The window can hold several samples for one process; the most recent is the one that
    # describes the device now, and averaging them smooths across a batch boundary the caller
    # is specifically trying to react to.
    latest = max(samples, key=lambda s: s.timestamp_us)
    return latest.sm


def device_shared_with_others(index: int, since_us: int = 0) -> bool | None:
    """Whether another process is doing work on one device, or `None` when unknowable.

    The precondition for trusting a device-level utilization reading as this process's own.
    Where it returns False, `nvml.DeviceTelemetry.sm_utilization` is this process's utilization
    and every decision built on it is sound; where it returns True, that reading is a sum and
    must not be fed to a per-process controller.

    Args:
        index: NVML device index.
        since_us: Driver timestamp to read forward from.

    Returns:
        True when a process other than this one was active, False when only this one was, and
        `None` when nothing was attributed at all.
    """
    import os

    mine = os.getpid()
    samples = [s for s in device_process_utilization(since_us) if s.index == index and s.active]
    if not samples:
        return None
    return any(s.pid != mine for s in samples)
