"""Xid errors — the driver's own account of what went wrong with a device.

An Xid is the NVIDIA driver's error code, written to the kernel log and nowhere else. It is
the only report of the failures that matter most on a GPU fleet: a double-bit ECC fault
(Xid 48), a device that has fallen off the bus (79), a GPU that stopped responding (62, 119,
120), a channel error that killed a running kernel (13, 31), or memory that has run out of
spare rows (94, 95). NVML has no counter for any of these. `nvidia-smi` shows none of them.
A job that hits one usually dies with a CUDA error whose text names none of it.

The consequence for a scheduler is specific. After a fatal Xid the device is not merely slow,
it is *unusable until reset*, and every task placed on it fails the same way — so a fleet
without this signal turns one bad device into a retry storm that walks the whole queue onto it.
Reading the code and quarantining the device by UUID converts that into one lost task.

**Severity is a published property of the code, not a judgment.** The fatal set below is the
subset of NVIDIA's own Xid table whose documented remedy is a device reset or an RMA. A code
outside the table reports as unknown severity, which callers treat as "log it, keep scheduling"
— inventing a severity for an unseen code is how a future driver release quarantines a fleet.

Reads the kernel ring buffer, which needs `CAP_SYSLOG` or a readable `/dev/kmsg`. Without
permission, off Linux, or inside a container that did not share the host's log, this reports
nothing and `xid_readable()` is False. Nothing is inferred from silence.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

__all__ = [
    "KMSG_PATH",
    "XID_APPLICATION",
    "XID_DESCRIPTIONS",
    "XID_FATAL",
    "XidEvent",
    "describe_xid",
    "recent_xid_events",
    "xid_application_faults",
    "xid_fatal",
    "xid_readable",
    "xid_severity",
]

#: The kernel ring buffer. A constant so a test can point it at a fixture and so a deployment
#: that exposes the log elsewhere (a mounted `kmsg`, a journal export) can redirect it.
KMSG_PATH = "/dev/kmsg"

#: Xid codes whose documented remedy is a device reset or a replacement — the set that makes a
#: device unschedulable rather than merely noteworthy. Taken from NVIDIA's published Xid table;
#: a code not listed here is *not* assumed benign, it is assumed unknown.
XID_FATAL = frozenset(
    {
        48,  # double-bit ECC error: data has already been returned wrong
        62,  # internal micro-controller halt
        63,  # ECC page retirement or row remapping recording event
        64,  # ECC page retirement or row remapper recording failure
        68,  # video processor exception
        74,  # NVLink error: the fabric between devices has faulted
        79,  # GPU has fallen off the bus
        92,  # high single-bit ECC error rate
        94,  # contained ECC error: the faulting process is dead, the device needs a reset
        95,  # uncontained ECC error: any process on the device may hold corrupt data
        119,  # GSP RPC timeout
        120,  # GSP error
        121,  # C2C link corrected error rate high
    }
)

#: Xid codes caused by the *workload* rather than the hardware, with what each one means.
#:
#: The distinction is the whole point of having it. These fire because a kernel did something
#: illegal — an out-of-bounds access, an MMU fault, a kernel that ran past its watchdog — and
#: the device is fine afterwards. Quarantining on one would take a healthy board out of a
#: fleet over a bug in the job that happened to land on it, and then take out the next board
#: the retry lands on, and so on: an application fault that walks the fleet is strictly worse
#: than the crash it came from.
#:
#: They are worth *naming* rather than ignoring, because an operator staring at "Xid 13" on a
#: node that keeps failing needs to know the answer is in their code and not in the rack.
XID_APPLICATION: dict[int, str] = {
    13: "graphics engine exception, usually an illegal memory access in a kernel",
    31: "GPU memory page fault, usually an out-of-bounds access in a kernel",
    43: "a kernel was stopped by the driver after a fault elsewhere in the process",
    45: "preemptive cleanup, usually a process killed while its kernels were running",
}

#: What each fatal code means, in the words an operator needs to act. Only the fatal set is
#: described here: a table of every Xid would be a copy of the vendor's documentation that
#: goes stale, while these are the ones a scheduler makes a decision about.
XID_DESCRIPTIONS: dict[int, str] = {
    48: "double-bit ECC error",
    62: "internal micro-controller halt",
    63: "ECC row remapping recorded",
    64: "ECC row remapping failed",
    68: "video processor exception",
    74: "NVLink fabric error",
    79: "GPU has fallen off the bus",
    92: "high single-bit ECC error rate",
    94: "contained ECC error, device needs reset",
    95: "uncontained ECC error, device state is untrusted",
    119: "GSP RPC timeout",
    120: "GSP error",
    121: "C2C link corrected error rate high",
}

#: `NVRM: Xid (PCI:0000:0c:00): 79, pid=..., GPU has fallen off the bus.` The PCI fragment the
#: driver prints omits the function digit, so the address it yields is normalized on the way out.
#: The device fragment is captured loosely rather than validated in the pattern: a line whose
#: address the driver printed in an unexpected shape still carries the *code*, which is the
#: half worth logging, and it is `_normalize_pci` that decides the event names no device.
_XID_RE = re.compile(r"NVRM:\s*Xid\s*\(PCI:([^)]*)\)\s*:\s*(\d+)")

#: How much of the ring buffer to read. The buffer is a few hundred kilobytes at most and a
#: node's Xid history is what matters, so this is a bound against a pathological log rather
#: than a sampling window.
_MAX_RECORDS = 4096


@dataclass(frozen=True, slots=True)
class XidEvent:
    """One Xid error the driver reported.

    Attributes:
        code: The Xid number.
        pci_address: The device's PCI address as the driver printed it, normalized to the
            `0000:0c:00.0` form so it joins against `hardware.fabric.pcie`. `""` when the
            driver did not name a device, which happens for a few system-scope codes.
        message: The remainder of the driver's line, for a log an operator will read.
    """

    code: int
    pci_address: str = ""
    message: str = ""

    @property
    def fatal(self) -> bool:
        """Whether this code makes the device unschedulable until it is reset."""
        return self.code in XID_FATAL

    @property
    def severity(self) -> str:
        """`"hardware"`, `"application"`, or `"unknown"` — who to send this to."""
        return xid_severity(self.code)

    @property
    def description(self) -> str:
        """A short description of the code, or `""` for one outside the documented set."""
        return XID_DESCRIPTIONS.get(self.code) or XID_APPLICATION.get(self.code, "")


def _normalize_pci(raw: str) -> str:
    """The driver's `0000:0c:00` fragment as a full PCI address, or `""`.

    The driver omits the function digit, which is always `.0` for a GPU's own function. A
    fragment in any other shape is passed through unchanged when it already looks like an
    address, and dropped otherwise rather than being repaired into something invented.
    """
    text = raw.strip().lower()
    if re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}", text):
        return f"{text}.0"
    if re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", text):
        return text
    return ""


def _kmsg_lines(path: str) -> list[str]:
    """Lines currently in the kernel ring buffer, or `[]` when it cannot be read.

    Opened non-blocking so a caller is never parked waiting for the next kernel message: the
    ring buffer replays its history from the start of the file and then would block for new
    records, and this only wants the history.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return []
    lines: list[str] = []
    try:
        for _ in range(_MAX_RECORDS):
            try:
                chunk = os.read(fd, 8192)
            except OSError:
                break  # EAGAIN: the history is exhausted and the next record has not arrived
            if not chunk:
                break
            lines.extend(chunk.decode("utf-8", "replace").splitlines())
    finally:
        os.close(fd)
    return lines


def xid_readable(path: str | None = None) -> bool:
    """Whether the kernel log can be read at all on this host.

    The question a caller must ask before treating "no Xid events" as "no faults". Inside a
    container without the host log, the two are indistinguishable by the events alone, and
    conflating them means a quarantine policy that silently stops working.

    Args:
        path: Kernel log to check, defaulting to `KMSG_PATH`.

    Returns:
        True when the log opened. False without permission, off Linux, or in a container that
        did not share it.
    """
    try:
        fd = os.open(path or KMSG_PATH, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return False
    os.close(fd)
    return True


def recent_xid_events(path: str | None = None) -> tuple[XidEvent, ...]:
    """Every Xid error currently in the kernel ring buffer, oldest first.

    Not memoized: the whole value is that a device that faulted a minute ago is seen now. The
    read costs a few hundred kilobytes of kernel buffer, so it belongs on a per-stage or
    health-check cadence, not a per-batch one.

    Args:
        path: Kernel log to read, defaulting to `KMSG_PATH`.

    Returns:
        The events, empty when the log holds none *or* cannot be read — call `xid_readable()`
        to tell those apart.
    """
    events: list[XidEvent] = []
    for line in _kmsg_lines(path or KMSG_PATH):
        match = _XID_RE.search(line)
        if match is None:
            continue
        try:
            code = int(match.group(2))
        except ValueError:
            continue
        tail = line[match.end() :].lstrip(" ,:")
        events.append(
            XidEvent(code=code, pci_address=_normalize_pci(match.group(1)), message=tail.strip())
        )
    return tuple(events)


def xid_fatal(events: tuple[XidEvent, ...] | None = None) -> dict[str, tuple[int, ...]]:
    """Fatal Xid codes seen per device, keyed by PCI address.

    The map a scheduler acts on: a device present here should not be given work until it has
    been reset, because every task placed on it fails the same way and the retries walk the
    whole queue onto the same device.

    Args:
        events: Events to inspect, or `None` to read them live.

    Returns:
        PCI address to the fatal codes seen for it, ascending and deduplicated. Events the
        driver did not attribute to a device are dropped rather than being attributed to an
        arbitrary one. Empty when nothing fatal was seen or the log is unreadable.
    """
    records = recent_xid_events() if events is None else events
    out: dict[str, set[int]] = {}
    for event in records:
        if event.fatal and event.pci_address:
            out.setdefault(event.pci_address, set()).add(event.code)
    return {address: tuple(sorted(codes)) for address, codes in sorted(out.items())}


def describe_xid(code: int) -> str:
    """A short description of an Xid code.

    Args:
        code: The Xid number.

    Returns:
        The documented meaning for a code this module classifies, or `"unknown Xid <code>"`
        for one it does not. Never a guess: a code these tables have not seen is reported as
        unrecognized so a reader goes to the vendor's documentation instead of trusting an
        invented gloss.
    """
    return XID_DESCRIPTIONS.get(code) or XID_APPLICATION.get(code) or f"unknown Xid {code}"


def xid_severity(code: int) -> str:
    """Whether an Xid blames the hardware, the workload, or neither.

    The classification a scheduler needs before it reacts. A hardware Xid means the device is
    unusable until it is reset; an application Xid means a kernel did something illegal and
    the device is fine. Treating the second as the first is the expensive mistake: it takes a
    healthy board out of the fleet over a bug in the job, then takes out the next board the
    retry lands on.

    Args:
        code: The Xid number.

    Returns:
        `"hardware"`, `"application"`, or `"unknown"`. An unrecognized code is never guessed
        into a class — a future driver release must not be able to quarantine a fleet through
        a code this build has never seen.
    """
    if code in XID_FATAL:
        return "hardware"
    if code in XID_APPLICATION:
        return "application"
    return "unknown"


def xid_application_faults(
    events: tuple[XidEvent, ...] | None = None,
) -> dict[str, tuple[int, ...]]:
    """Workload-caused Xid codes seen per device, keyed by PCI address.

    The counterpart of `xid_fatal`, and deliberately a separate call rather than a flag on it:
    the two have opposite remedies. A device here needs no operator action at all — the job
    that produced the fault does — and mixing the two lists is how a healthy board ends up
    drained over someone's out-of-bounds write.

    Args:
        events: Events to inspect, or `None` to read them live.

    Returns:
        PCI address to the application codes seen for it, ascending and deduplicated. Empty
        when none were seen or the log is unreadable.
    """
    records = recent_xid_events() if events is None else events
    out: dict[str, set[int]] = {}
    for event in records:
        if event.pci_address and xid_severity(event.code) == "application":
            out.setdefault(event.pci_address, set()).add(event.code)
    return {address: tuple(sorted(codes)) for address, codes in sorted(out.items())}
