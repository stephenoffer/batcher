"""How a GPU is failing, as distinct from how busy it is.

`hardware.nvml` reads what a device is *doing*. This package reads what has gone *wrong* with
it, which is a different question with different sources and a different cadence. A device that
has fallen off the bus, double-bit-faulted its memory, or exhausted its spare memory rows is
still enumerated, still reports a temperature, and still accepts work. It is the single most
expensive thing a scheduler can keep feeding.

Five sources, because no one source has it all:

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
* `kernel` — the node faults that are not about the device at all. A worker killed by the OOM
  killer never raises, and a filesystem remounted read-only takes down every task that spills,
  while the node stays up and stays scheduled.
* `actions` — what to *do* about a code, and whether results already computed on the device
  can still be trusted. A device that fell off the bus corrupts nothing; one that took a
  double-bit ECC error returned a wrong number and kept going.

`kmsg` underlies the two that read the kernel log, so a ring-buffer overrun and a missing
timestamp are handled once rather than in each of them.

Both degrade to "nothing reported" without the driver, without permission to read the kernel
log, and off Linux. A caller cannot distinguish "healthy" from "unreadable" by the values
alone, which is deliberate — `readable()` is the question to ask, and a fleet where the answer
is False must never be quarantined for it.

A neutral utility: any layer may import `_internal`.
"""

from __future__ import annotations

from batcher._internal.hardware.faults.actions import (
    XID_UNTRUSTED,
    device_remedy,
    explain_codes,
    xid_remedy,
    xid_untrusted,
)
from batcher._internal.hardware.faults.counters import (
    DeviceFaults,
    device_faults,
    faulted_devices,
)
from batcher._internal.hardware.faults.kernel import (
    NODE_WINDOW_S,
    NodeFault,
    node_fault_counts,
    node_faults,
    node_faults_readable,
    worst_severity,
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
    XID_WINDOW_S,
    XidEvent,
    describe_xid,
    recent_xid_events,
    xid_application_faults,
    xid_counts,
    xid_fatal,
    xid_readable,
    xid_severity,
    xid_unclassified,
)

__all__ = [
    "NODE_WINDOW_S",
    "XID_APPLICATION",
    "XID_DESCRIPTIONS",
    "XID_FATAL",
    "XID_UNTRUSTED",
    "XID_WINDOW_S",
    "DeviceFaults",
    "DeviceModes",
    "NodeFault",
    "XidEvent",
    "describe_xid",
    "device_faults",
    "device_modes",
    "device_remedy",
    "explain_codes",
    "faulted_devices",
    "misconfigured_devices",
    "node_fault_counts",
    "node_faults",
    "node_faults_readable",
    "recent_xid_events",
    "worst_severity",
    "xid_application_faults",
    "xid_counts",
    "xid_fatal",
    "xid_readable",
    "xid_remedy",
    "xid_severity",
    "xid_unclassified",
    "xid_untrusted",
]
