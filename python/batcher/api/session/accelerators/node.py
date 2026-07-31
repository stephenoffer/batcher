"""What is wrong with the *node*, as distinct from what is wrong with its accelerators.

`rows` and `report` answer for the devices. This answers for the machine around them, and it
exists because a GPU node most often goes bad in a way that leaves every GPU on it reading
perfectly healthy:

* the kernel out-of-memory killer takes a worker, which never raises, never unwinds, and
  never logs — from the driver it is indistinguishable from a preemption;
* the spill filesystem is remounted read-only, and every stateful operator placed there fails
  its first write while the node stays up and stays scheduled;
* the driver never brought a device up at all, so the node is one GPU short and a collective
  sized for the fleet waits forever for a rank that will never join.

None of those appears in a device row, an Xid the scheduler acts on, or a Python traceback.
They are in the kernel log, which is the one place nothing else reads.

Alongside them, the driver error codes this build does *not* classify. Nothing acts on those —
inventing a severity for an unseen code is how a driver release quarantines a fleet — but
dropping them silently is how the same release becomes months of "those nodes are just flaky",
with nothing anywhere naming a code the vendor documents and Batcher does not.
"""

from __future__ import annotations

__all__ = ["node_problems"]

#: What each kernel fault category means for the job, in the words an operator acts on. Spelled
#: out rather than printed as a reason code, for the same reason the device findings are: the
#: reader is deciding whether to drain a node, and `storage_io` on its own does not say why.
_NODE_ADVICE = {
    "host_oom": "the kernel has run out of memory and killed a process here",
    "filesystem_readonly": "a filesystem was remounted read-only; spilling will fail",
    "memory_ecc_uncorrected": "host memory returned an uncorrectable error",
    "pcie_fatal": "an uncorrectable PCIe error was reported on a device link",
    "driver_init": "the driver failed to bring a device up; this node is a GPU short",
    "storage_io": "the storage under the scratch directory reported I/O errors or reset",
    "lockup": "a CPU lockup was reported",
    "hung_task": "a task was blocked in the kernel for minutes, usually storage or a driver",
    "rcu_stall": "CPUs stalled for tens of seconds, usually a driver holding a lock",
    "nvlink": "an NVLink or NVSwitch error was reported",
    "network_down": "a network link went down",
    "cpu_thermal": "a CPU passed its thermal threshold and was clamped",
    "process_limit": "this container hit its process limit; new workers cannot be spawned",
}


def node_problems() -> list[str]:
    """This host's non-device faults and unclassified driver errors, as sentences.

    Returns:
        One sentence per unrecognized Xid and per fault kind seen recently, empty on a clean
        host *and* on one whose kernel log cannot be read. Silence is never treated as health,
        which is why `bt.accelerators()` reports whether the log was readable at all.
    """
    from batcher._internal.hardware.faults import (
        node_fault_counts,
        node_faults,
        xid_unclassified,
    )

    out: list[str] = []
    for address, codes in xid_unclassified().items():
        listed = ", ".join(str(c) for c in codes)
        out.append(
            f"device {address}: unrecognized driver error Xid {listed} — check the vendor's "
            "Xid table; Batcher does not classify it and is not acting on it"
        )
    counts = node_fault_counts(node_faults())
    out.extend(
        f"node: {_NODE_ADVICE[kind]} ({count}x)"
        for kind, count in sorted(counts.items())
        if kind in _NODE_ADVICE
    )
    return out
