"""How a GPU is failing, as distinct from how busy it is.

`hardware.nvml` reads what a device is *doing*. This package reads what has gone *wrong* with
it, which is a different question with different sources and a different cadence. A device that
has fallen off the bus, double-bit-faulted its memory, or exhausted its spare memory rows is
still enumerated, still reports a temperature, and still accepts work. It is the single most
expensive thing a scheduler can keep feeding.

Three sources, because no one source has it all:

* `xid` — the driver's own error events, scraped from the kernel log. Xid is the only place a
  double-bit ECC fault, a fallen-off-the-bus device, or a GPU that stopped responding is
  reported at all; NVML has no counter for any of them.
* `counters` — NVML's remapped-row and retired-page accounting, plus the PCIe replay counter.
  These are the *predictive* signals: a device with pending row remaps needs a reset, and one
  with a remap failure has run out of spares and needs replacing.
* `modes` — the settings a device arrived configured with. ECC off, persistence off, an
  exclusive compute mode, or a power limit at the part's floor each cost throughput or
  correctness without raising anything, and on a rented node they are whatever the last
  tenant left behind.

Both degrade to "nothing reported" without the driver, without permission to read the kernel
log, and off Linux. A caller cannot distinguish "healthy" from "unreadable" by the values
alone, which is deliberate — `readable()` is the question to ask, and a fleet where the answer
is False must never be quarantined for it.

A neutral utility: any layer may import `_internal`.
"""

from __future__ import annotations

from batcher._internal.hardware.faults.counters import (
    DeviceFaults,
    device_faults,
    faulted_devices,
)
from batcher._internal.hardware.faults.modes import (
    DeviceModes,
    device_modes,
    misconfigured_devices,
)
from batcher._internal.hardware.faults.xid import (
    XID_APPLICATION,
    XID_DESCRIPTIONS,
    XID_FATAL,
    XidEvent,
    describe_xid,
    recent_xid_events,
    xid_application_faults,
    xid_fatal,
    xid_readable,
    xid_severity,
)

__all__ = [
    "XID_APPLICATION",
    "XID_DESCRIPTIONS",
    "XID_FATAL",
    "DeviceFaults",
    "DeviceModes",
    "XidEvent",
    "describe_xid",
    "device_faults",
    "device_modes",
    "faulted_devices",
    "misconfigured_devices",
    "recent_xid_events",
    "xid_application_faults",
    "xid_fatal",
    "xid_readable",
    "xid_severity",
]
