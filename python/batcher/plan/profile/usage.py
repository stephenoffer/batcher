"""The operating system's own account of what this process consumed.

The engine measures this for itself: `bc-interp`'s `rusage` module samples `getrusage` and
`/proc/self/io` around every metered execution, and the reading rides back to the control
plane as the `query` block of an `ExecMetrics` document. That covers every query the engine
runs as one call.

It does not cover the **out-of-core path**, and that is the gap this module fills. A query
too large for memory is not executed once; it is streamed through thousands of unmetered
`execute_plan` dispatches with a partition and a reduce phase around them, deliberately, so
that metering does not cost more than the work. The result is that the queries most worth
observing — the ones that went to disk because they did not fit — reported no CPU, no
memory, and no disk consumption at all, which is exactly backwards.

Measuring the same counters once around that whole phase costs two syscalls and is sound for
the same reason the engine's own measurement is: these are process-wide totals, and the delta
across a phase belongs to that phase. The reading is shaped to match the engine's `query`
block key for key, so `QueryUsage.from_metrics` reads either without knowing which
produced it — which is why this lives beside that type rather than in the hardware package:
the shape of the reading is the contract, and there is one definition of it.

Every field is `0` where the platform cannot report it, and `0` means **unmeasured**.
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

__all__ = ["UsageStopwatch"]

#: `ru_maxrss` is KiB on Linux and bytes on the BSDs and macOS, the same split the Rust
#: side normalizes. Getting it wrong is a 1024x error in a memory figure, so it is named
#: rather than inlined.
_MAXRSS_SCALE = 1024 if sys.platform.startswith("linux") else 1


class UsageStopwatch:
    """A whole-phase resource measurement in progress. Start it before the work runs.

    The counterpart of `bc_interp::QueryStopwatch` for work the engine runs as many small
    unmetered calls rather than one metered one.
    """

    __slots__ = ("_started", "_wall")

    def __init__(self) -> None:
        self._wall = time.perf_counter()
        self._started = _sample()

    def finish(self) -> dict[str, int]:
        """The deltas since construction, keyed like the engine's `ExecMetrics.query` block.

        Returns:
            A dict with ``wall_ns``, ``cpu_ns``, ``peak_rss_bytes``, ``minor_faults``,
            ``major_faults``, ``vol_ctx_switches``, ``invol_ctx_switches``,
            ``io_read_bytes``, and ``io_write_bytes``. Saturating, so a counter that did not
            advance reports `0` rather than a negative number.
        """
        wall_ns = int((time.perf_counter() - self._wall) * 1e9)
        now = _sample()
        out = {name: max(0, now[name] - self._started[name]) for name in now}
        out["wall_ns"] = wall_ns
        return out


def _sample() -> dict[str, int]:
    """Every counter this platform can report, right now. Unreadable ones stay `0`."""
    out = dict.fromkeys(
        (
            "cpu_ns",
            "peak_rss_bytes",
            "minor_faults",
            "major_faults",
            "vol_ctx_switches",
            "invol_ctx_switches",
            "io_read_bytes",
            "io_write_bytes",
        ),
        0,
    )
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    except (OSError, ValueError):  # pragma: no cover - getrusage does not fail in practice
        return out
    out["cpu_ns"] = int((usage.ru_utime + usage.ru_stime) * 1e9)
    out["peak_rss_bytes"] = int(usage.ru_maxrss) * _MAXRSS_SCALE
    out["minor_faults"] = int(usage.ru_minflt)
    out["major_faults"] = int(usage.ru_majflt)
    out["vol_ctx_switches"] = int(usage.ru_nvcsw)
    out["invol_ctx_switches"] = int(usage.ru_nivcsw)
    read_bytes, write_bytes = _proc_io_bytes()
    out["io_read_bytes"] = read_bytes
    out["io_write_bytes"] = write_bytes
    return out


def _proc_io_bytes() -> tuple[int, int]:
    """Block-device bytes from ``/proc/self/io``, or ``(0, 0)`` where it does not exist.

    These are the only two counters `getrusage` cannot supply, and the pair that separates a
    warm scan from a cold one: `ru_inblock` counts 512-byte blocks the *kernel* charged, not
    what reached the device, and is `0` on Linux for exactly this workload.
    """
    try:
        text = Path("/proc/self/io").read_text()
    except OSError:
        return 0, 0
    read_bytes = write_bytes = 0
    for line in text.splitlines():
        name, _, value = line.partition(":")
        if name == "read_bytes":
            read_bytes = _as_int(value)
        elif name == "write_bytes":
            write_bytes = _as_int(value)
    return read_bytes, write_bytes


def _as_int(value: str) -> int:
    """`value` as an int, or `0` when it is not one."""
    try:
        return int(value.strip())
    except ValueError:
        return 0
