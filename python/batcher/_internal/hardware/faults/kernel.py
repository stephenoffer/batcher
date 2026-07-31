"""Node faults the kernel reports that have nothing to do with the GPU.

A GPU job on an unstable node fails for reasons no device probe can see, and the two most
common ones leave no trace anywhere else:

* **The OOM killer.** A worker that is killed by the kernel does not raise. It does not unwind,
  it does not log, and it does not get a chance to say anything. From the orchestrator's side
  it is indistinguishable from a segfault, a preemption, or a network partition — the actor is
  simply gone. The kernel wrote down exactly what happened and to which process, in the one
  place nothing else reads.
* **A filesystem that went read-only.** Spill, checkpoints, and the object store all write to
  disk. When `ext4` hits an error it remounts read-only, and every subsequent write fails with
  `EROFS` while the node stays up, stays scheduled, and accepts more work. Every task placed on
  it from that moment fails, so the retries walk the whole queue onto it — the same shape as a
  bad GPU, from an entirely different cause.

The rest are early warnings for the same class of trouble: a PCIe link retraining under a
device, host DIMM errors, a task blocked in the kernel for minutes, a NIC bouncing, a soft
lockup. None of them is proof a node is bad. All of them are things an operator staring at
"the job keeps failing on node 7" needs, and none of which surface anywhere in a Python
traceback.

**Every pattern here is a match against a kernel message, so it says what the kernel said and
nothing more.** A category is not a verdict — a corrected PCIe error is the link working as
designed — and the severity attached to each is about how much attention it deserves, not about
whether to drain a node. That decision belongs to a policy with thresholds, which is a
Carbonite concern and not this module's.

Degrades to "nothing reported" without a readable kernel log, which is the common case inside a
container. `node_faults_readable()` is the question to ask, because silence here is
indistinguishable from health.

A neutral utility: any layer may import `_internal`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from batcher._internal.hardware.faults.kmsg import kmsg_readable, monotonic_now_s, read_kmsg

__all__ = [
    "NODE_WINDOW_S",
    "SEVERITY_BY_KIND",
    "NodeFault",
    "node_fault_counts",
    "node_faults",
    "node_faults_readable",
    "worst_severity",
]

#: How far back a node-health read looks by default, in seconds. One hour: shorter than the Xid
#: window because these are host conditions that either recur or resolve, where a device fault
#: persists until someone repairs it. An OOM kill from this morning says nothing about whether
#: the node has memory now; a device that fell off the bus this morning is still off the bus.
NODE_WINDOW_S = 60 * 60.0

#: How much attention each category deserves, on its own, before any threshold is applied.
#:
#: `"fatal"` means the node is actively unable to do the work — it cannot write, or it has
#: already killed a process. `"degraded"` means something is wrong that costs throughput or
#: predicts a failure. `"note"` means it is worth having in an incident report and worth
#: nothing on its own.
SEVERITY_BY_KIND: dict[str, str] = {
    "host_oom": "fatal",
    "filesystem_readonly": "fatal",
    "pcie_fatal": "fatal",
    "memory_ecc_uncorrected": "fatal",
    "driver_init": "fatal",
    "storage_io": "degraded",
    "lockup": "degraded",
    "hung_task": "degraded",
    "rcu_stall": "degraded",
    "nvlink": "degraded",
    "network_down": "degraded",
    "cpu_thermal": "degraded",
    "process_limit": "degraded",
    "memory_ecc_corrected": "note",
    "pcie_corrected": "note",
}

#: `(kind, pattern)` in the order they are tried; the first match wins, so the more specific
#: pattern of a pair comes first.
#:
#: Written against the kernel's own message text, which is stable across releases in a way the
#: surrounding format is not — hence the loose anchoring. A pattern that stops matching reports
#: nothing rather than reporting wrongly, which is the correct direction for a signal that is
#: allowed to be absent on every node that does not share its host log.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # "Out of memory: Killed process 1234 (python)" and the cgroup-v2 spelling
    # "oom-kill:constraint=CONSTRAINT_MEMCG,...". Both name the victim, which is what makes an
    # unexplained dead worker explainable.
    ("host_oom", re.compile(r"Out of memory:\s*Kill|oom-kill:|Memory cgroup out of memory")),
    # ext4/xfs/btrfs each say it differently; all three end in a filesystem that cannot be
    # written, which for a spill directory means every task on the node fails from here on.
    (
        "filesystem_readonly",
        re.compile(r"Remounting filesystem read-only|Detected aborted journal|forced readonly"),
    ),
    # AER, fatal first: an uncorrectable error on the link under a device is the link failing,
    # where a corrected one is the link's error correction doing its job.
    ("pcie_fatal", re.compile(r"AER:.*(Uncorrected|Fatal)", re.IGNORECASE)),
    ("pcie_corrected", re.compile(r"AER:.*Corrected", re.IGNORECASE)),
    # Host DIMM errors, via EDAC or the machine-check subsystem. Uncorrected means the host
    # already returned wrong data — the CPU-side equivalent of a double-bit ECC on a device.
    (
        "memory_ecc_uncorrected",
        re.compile(r"(EDAC.*\bUE\b)|Uncorrected error|mce:.*Hardware Error", re.IGNORECASE),
    ),
    ("memory_ecc_corrected", re.compile(r"EDAC.*\bCE\b|Corrected error", re.IGNORECASE)),
    # The driver failing to bring a device up at all. Distinct from every Xid, which is what
    # a *working* driver reports about a device: this is the driver saying it never got one.
    # The node then comes up with fewer GPUs than the fleet expects, and the symptom is a
    # collective that waits forever for a rank that will never join.
    ("driver_init", re.compile(r"NVRM: (RmInitAdapter failed|GPU \S+ Failed to initialize)")),
    # The spill device itself. A controller reset or an I/O error under the scratch filesystem
    # is what precedes the read-only remount above, so it is the earlier warning for the same
    # eventual outcome — every stateful operator on the node failing its first write.
    (
        "storage_io",
        re.compile(
            r"resetting controller|blk_update_request: I/O error|Buffer I/O error|"
            r"nvme\S*:.*(timeout|aborting)",
            re.IGNORECASE,
        ),
    ),
    ("lockup", re.compile(r"watchdog: BUG: (soft|hard) lockup")),
    # An RCU stall means CPUs stopped reporting a quiescent state for tens of seconds, which
    # on a GPU node is usually a driver holding a lock across a device operation that hung.
    ("rcu_stall", re.compile(r"rcu:?\s*INFO: rcu_\w+ detected|self-detected stall on CPU")),
    # A container that has hit its process or thread ceiling. Every worker spawn from here on
    # fails, and it fails as an `OSError` from `fork` that no accelerator probe can explain.
    ("process_limit", re.compile(r"fork rejected by pids controller|cgroup: fork rejected")),
    # A task blocked in uninterruptible sleep for two minutes is usually storage or a driver
    # that has stopped answering, and it is the signature of a hang rather than a crash.
    ("hung_task", re.compile(r"task .* blocked for more than \d+ seconds")),
    ("nvlink", re.compile(r"nvidia-nvswitch|NVLink.*(error|down|training)", re.IGNORECASE)),
    ("network_down", re.compile(r"NETDEV WATCHDOG|Link (is )?[Dd]own|link is not ready")),
    ("cpu_thermal", re.compile(r"CPU\d+: (Core|Package) temperature above threshold")),
)


@dataclass(frozen=True, slots=True)
class NodeFault:
    """One node-level fault the kernel reported.

    Attributes:
        kind: The category, a key of `SEVERITY_BY_KIND`.
        text: The kernel's own message, for an operator to read. Never paraphrased: the
            original line is what a vendor or a kernel mailing list will recognize.
        timestamp_s: Seconds since boot, or `-1.0` when the log carried no usable timestamp.
    """

    kind: str
    text: str
    timestamp_s: float = -1.0

    @property
    def severity(self) -> str:
        """`"fatal"`, `"degraded"`, or `"note"` for this category."""
        return SEVERITY_BY_KIND.get(self.kind, "note")


def node_faults_readable(path: str | None = None) -> bool:
    """Whether the kernel log can be read at all on this host.

    Args:
        path: Kernel log to check, defaulting to the ring buffer.

    Returns:
        True when the log opened. False in a container without the host log, without
        `CAP_SYSLOG`, and off Linux — where "no faults" is not evidence of a healthy node.
    """
    return kmsg_readable(path)


def node_faults(
    path: str | None = None, within_s: float | None = NODE_WINDOW_S
) -> tuple[NodeFault, ...]:
    """Node faults in the kernel ring buffer, oldest first.

    Args:
        path: Kernel log to read, defaulting to the ring buffer.
        within_s: Keep only faults written within this many seconds, or `None` for the whole
            buffer. An undated record is kept either way — unknown age is not old age.

    Returns:
        The faults, empty when the log holds none *or* cannot be read. Call
        `node_faults_readable()` to tell those apart.
    """
    now = monotonic_now_s()
    out: list[NodeFault] = []
    for record in read_kmsg(path):
        if within_s is not None and record.timestamp_s >= 0.0 and record.age_s(now) > within_s:
            continue
        for kind, pattern in _PATTERNS:
            if pattern.search(record.text):
                out.append(NodeFault(kind=kind, text=record.text, timestamp_s=record.timestamp_s))
                break
    return tuple(out)


def node_fault_counts(faults: tuple[NodeFault, ...] | None = None) -> dict[str, int]:
    """How many faults of each kind were seen.

    The shape a policy wants, because almost every one of these categories is a *rate*
    question. One corrected PCIe error is the link working; ten thousand is a cable. One
    corrected DIMM error is a cosmic ray; a rising count on one node is a DIMM on its way out.

    Args:
        faults: Faults to count, or `None` to read them live.

    Returns:
        Kind to occurrence count, only for kinds that occurred.
    """
    records = node_faults() if faults is None else faults
    out: dict[str, int] = {}
    for fault in records:
        out[fault.kind] = out.get(fault.kind, 0) + 1
    return out


def worst_severity(faults: tuple[NodeFault, ...]) -> str:
    """The highest severity present, or `"none"` for no faults.

    Args:
        faults: Faults to reduce.

    Returns:
        `"fatal"`, `"degraded"`, `"note"`, or `"none"`.
    """
    order = ("none", "note", "degraded", "fatal")
    worst = "none"
    for fault in faults:
        if order.index(fault.severity) > order.index(worst):
            worst = fault.severity
    return worst
