"""Timing device work correctly, and labelling it so an external profiler agrees with us.

Two problems, one module, because the fix for both is a pair of markers around the same block.

**Wall-clock does not measure a GPU.** A CUDA launch is asynchronous: the host queues the kernel
and returns, usually in single-digit microseconds, whatever the kernel then does for the next
half second. A `perf_counter` bracket around it therefore measures the *launch*, and a stage
built out of such brackets reports device work as effectively free while the run takes minutes.
The classic symptom is a profile in which every operator is fast and the total is not, with the
missing time appearing at whatever call happens to synchronize first — usually the copy back to
the host, which then gets blamed for work it did not do.

CUDA events are the fix and they are the only fix: they are recorded *into the stream*, so the
interval between two of them is the time the device spent, measured by the device. This module
wraps that, including the part callers get wrong — an event's elapsed time is not readable until
it has been synchronized, and synchronizing it inside the hot path serializes the pipeline the
asynchrony existed to build.

**A profiler capture needs the same brackets.** The NVTX range and the CUDA events want to go
around exactly the same block, so they are pushed together and a caller opts into one bracket
rather than remembering two.

**Off by default and free when off.** Annotation is one attribute lookup and a config read when
disabled, and the config read is the expensive part — so the gate is resolved once per range
rather than per call inside it. Turning it on costs a CUDA event pair per range, which is
sub-microsecond on the host and does not synchronize.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field

from batcher._internal.instrument.nvtx import device_range

__all__ = [
    "DeviceTiming",
    "operator_range",
    "profiling_enabled",
    "time_device_work",
]


def profiling_enabled() -> bool:
    """Whether device-profiler annotation is switched on for this process.

    Read from `accelerator.profiling` rather than inferred from whether a profiler is attached,
    because there is no way to ask the driver that and a heuristic here would either annotate
    always or never.

    Returns:
        True when ranges should be emitted. False on any configuration failure, so a broken
        config cannot turn a diagnostic into an outage.
    """
    try:
        from batcher.config import active_config

        return bool(active_config().accelerator.profiling)
    except Exception:
        return False


@dataclass
class DeviceTiming:
    """The device time one block actually spent, filled in after the fact.

    Read `milliseconds` only once the work has completed — the value is `None` until the
    recorded events have been synchronized, and `resolve` is what does that. Reading it early is
    not an error and does not block; it reports `None`, which is the honest answer while the
    kernel is still running.

    Attributes:
        label: The range name, matching what the profiler capture shows.
    """

    label: str = ""
    _events: tuple = field(default=(), repr=False)
    _resolved: float | None = field(default=None, repr=False)

    def resolve(self) -> float | None:
        """Synchronize the recorded events and return the device milliseconds between them.

        **This blocks** until the device reaches the closing event, which is the entire point
        and also why it must not be called inside the loop being measured: doing so drains the
        pipeline every iteration and turns an asynchronous stage into a synchronous one, which
        is a real slowdown introduced by the act of measuring.

        Returns:
            Milliseconds the device spent, or `None` when events were unavailable — no CUDA, no
            torch, or profiling switched off — in which case the caller keeps whatever
            wall-clock figure it had.
        """
        if self._resolved is not None:
            return self._resolved
        if len(self._events) != 2:
            return None
        start, end = self._events
        try:
            end.synchronize()
            self._resolved = float(start.elapsed_time(end))
        except Exception:
            return None
        return self._resolved

    @property
    def milliseconds(self) -> float | None:
        """Device milliseconds if already resolved, `None` otherwise. Never blocks."""
        return self._resolved


def _cuda_events() -> tuple:
    """A recorded `(start, end)` CUDA event pair, or `()` when unavailable.

    Only the start is recorded here; the caller records the end. Returning the pair rather than
    a handle keeps the whole interaction with torch inside this module, so nothing above it has
    to know whether torch is present.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return ()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return (start, end)
    except Exception:
        return ()


@contextlib.contextmanager
def time_device_work(label: str) -> Iterator[DeviceTiming]:
    """Measure the device time a block's work takes, and label it for a profiler capture.

    The measurement is asynchronous: this returns as soon as the closing event is *queued*, not
    when the device reaches it, so bracketing a stage costs nothing and does not serialize the
    pipeline. Call `DeviceTiming.resolve` later — after the stage has drained — to read the
    figure.

    Args:
        label: Range name, conventionally `"Kind#id"` so a capture joins to the stage
            identifiers the energy ledger already uses.

    Yields:
        A `DeviceTiming` that is empty until resolved.
    """
    if not profiling_enabled():
        yield DeviceTiming(label=label)
        return
    events = _cuda_events()
    timing = DeviceTiming(label=label, _events=events)
    with device_range(label):
        try:
            yield timing
        finally:
            if len(events) == 2:
                try:
                    events[1].record()
                except Exception:
                    # An event that will not record leaves the pair unusable; clearing it is
                    # what stops `resolve` from blocking forever on a closing event the device
                    # was never told to reach.
                    timing._events = ()


@contextlib.contextmanager
def operator_range(kind: str, node_id: object = "") -> Iterator[None]:
    """Label one operator's execution on the device profiler's timeline.

    The bracket to put at the point an operator is dispatched, so a Nsight capture reads as the
    plan instead of as an undifferentiated kernel list. Free when profiling is off.

    Args:
        kind: Operator kind, such as `"HashJoin"` or `"Aggregate"`.
        node_id: Plan node identifier, appended after a `#` when non-empty. Included because a
            plan with four joins in it produces four identically named bands otherwise, and
            telling them apart is usually the reason the capture was taken.

    Yields:
        Nothing; the block runs inside the range.
    """
    if not profiling_enabled():
        yield
        return
    label = f"{kind}#{node_id}" if node_id != "" else kind
    with device_range(label):
        yield
