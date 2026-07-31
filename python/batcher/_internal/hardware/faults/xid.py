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

**An Xid has an age, and the age decides whether it still means anything.** The ring buffer
holds a node's history, not its present: an entry from before the last device reset, or from
the tenant who had the node yesterday, is still sitting there. Quarantining on it takes a
repaired device out of the fleet and never puts it back, because the evidence never expires.
So every read here can be windowed, `recent_xid_events(within_s=...)`, and the scheduler-facing
maps default to a window rather than to all of history.

Reads the kernel ring buffer, which needs `CAP_SYSLOG` or a readable `/dev/kmsg`. Without
permission, off Linux, or inside a container that did not share the host's log, this reports
nothing and `xid_readable()` is False. Nothing is inferred from silence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from batcher._internal.hardware.faults.kmsg import (
    KMSG_PATH,
    kmsg_readable,
    monotonic_now_s,
    read_kmsg,
)

__all__ = [
    "KMSG_PATH",
    "XID_APPLICATION",
    "XID_DESCRIPTIONS",
    "XID_FATAL",
    "XID_WINDOW_S",
    "XidEvent",
    "describe_xid",
    "recent_xid_events",
    "xid_application_faults",
    "xid_counts",
    "xid_fatal",
    "xid_readable",
    "xid_severity",
    "xid_unclassified",
]

#: How far back a scheduling decision looks by default, in seconds. Six hours is long enough
#: to cover a job that started this morning and short enough that a device reset yesterday is
#: not still being punished for what it did before it.
#:
#: The number matters in one direction only. Too *long* is the dangerous end: a device that was
#: reset, repaired, or reprovisioned stays quarantined on evidence that has no expiry, and the
#: fleet shrinks monotonically over a node's lifetime with nothing in any log to say why. Too
#: short merely re-learns a still-broken device the next time it faults, which it will.
XID_WINDOW_S = 6 * 60 * 60.0

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


@dataclass(frozen=True, slots=True)
class XidEvent:
    """One Xid error the driver reported.

    Attributes:
        code: The Xid number.
        pci_address: The device's PCI address as the driver printed it, normalized to the
            `0000:0c:00.0` form so it joins against `hardware.fabric.pcie`. `""` when the
            driver did not name a device, which happens for a few system-scope codes.
        message: The remainder of the driver's line, for a log an operator will read.
        timestamp_s: Seconds since boot, from the kernel record's own header. `-1.0` when the
            log carried no usable timestamp, which a caller must read as "unknown age" — an
            unknown-age event is kept by every window rather than aged out, because dropping
            a fault you cannot date is how a live one goes unseen.
    """

    code: int
    pci_address: str = ""
    message: str = ""
    timestamp_s: float = -1.0

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
    return kmsg_readable(path)


def recent_xid_events(
    path: str | None = None, within_s: float | None = None
) -> tuple[XidEvent, ...]:
    """Xid errors in the kernel ring buffer, oldest first.

    Not memoized: the whole value is that a device that faulted a minute ago is seen now. The
    read costs a few hundred kilobytes of kernel buffer, so it belongs on a per-stage or
    health-check cadence, not a per-batch one.

    Args:
        path: Kernel log to read, defaulting to `KMSG_PATH`.
        within_s: Keep only events written within this many seconds, or `None` for the whole
            buffer. An event the log did not date is kept either way — an undated fault is
            unknown, not old, and aging it out is how a live one disappears.

    Returns:
        The events, empty when the log holds none *or* cannot be read — call `xid_readable()`
        to tell those apart.
    """
    now = monotonic_now_s()
    events: list[XidEvent] = []
    for record in read_kmsg(path):
        match = _XID_RE.search(record.text)
        if match is None:
            continue
        if within_s is not None and record.timestamp_s >= 0.0 and record.age_s(now) > within_s:
            continue
        tail = record.text[match.end() :].lstrip(" ,:")
        events.append(
            XidEvent(
                code=int(match.group(2)),
                pci_address=_normalize_pci(match.group(1)),
                message=tail.strip(),
                timestamp_s=record.timestamp_s,
            )
        )
    return tuple(events)


def _by_address(
    events: tuple[XidEvent, ...] | None,
    within_s: float | None,
    keep,
) -> dict[str, tuple[int, ...]]:
    """`{pci address: codes}` for the events `keep` accepts, read live when none are given."""
    records = recent_xid_events(within_s=within_s) if events is None else events
    out: dict[str, set[int]] = {}
    for event in records:
        if event.pci_address and keep(event):
            out.setdefault(event.pci_address, set()).add(event.code)
    return {address: tuple(sorted(codes)) for address, codes in sorted(out.items())}


def xid_fatal(
    events: tuple[XidEvent, ...] | None = None,
    within_s: float | None = XID_WINDOW_S,
) -> dict[str, tuple[int, ...]]:
    """Fatal Xid codes seen per device, keyed by PCI address.

    The map a scheduler acts on: a device present here should not be given work until it has
    been reset, because every task placed on it fails the same way and the retries walk the
    whole queue onto the same device.

    Args:
        events: Events to inspect, or `None` to read them live.
        within_s: Ignore events older than this many seconds. Defaults to `XID_WINDOW_S`
            rather than to the whole buffer, because a fatal Xid with no expiry quarantines a
            device that has since been reset and never releases it. Pass `None` for a
            forensic read of everything the buffer still holds. Ignored when `events` is
            given — window those at the read.

    Returns:
        PCI address to the fatal codes seen for it, ascending and deduplicated. Events the
        driver did not attribute to a device are dropped rather than being attributed to an
        arbitrary one. Empty when nothing fatal was seen or the log is unreadable.
    """
    return _by_address(events, within_s, lambda e: e.fatal)


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
    within_s: float | None = XID_WINDOW_S,
) -> dict[str, tuple[int, ...]]:
    """Workload-caused Xid codes seen per device, keyed by PCI address.

    The counterpart of `xid_fatal`, and deliberately a separate call rather than a flag on it:
    the two have opposite remedies. A device here needs no operator action at all — the job
    that produced the fault does — and mixing the two lists is how a healthy board ends up
    drained over someone's out-of-bounds write.

    Args:
        events: Events to inspect, or `None` to read them live.
        within_s: Ignore events older than this many seconds; `None` reads the whole buffer.
            Ignored when `events` is given.

    Returns:
        PCI address to the application codes seen for it, ascending and deduplicated. Empty
        when none were seen or the log is unreadable.
    """
    return _by_address(events, within_s, lambda e: xid_severity(e.code) == "application")


def xid_unclassified(
    events: tuple[XidEvent, ...] | None = None,
    within_s: float | None = XID_WINDOW_S,
) -> dict[str, tuple[int, ...]]:
    """Xid codes this build recognizes as neither hardware nor workload, per device.

    The counterpart of the module's refusal to guess. A code outside both tables is reported
    as unknown severity and deliberately does not quarantine anything — but it is also the
    single most interesting line in the log for an operator on a node that keeps failing,
    because it is the one the vendor's documentation has an entry for and this build does not.
    Dropping it silently is how a driver release introduces a fault mode that a fleet then
    experiences for months as "nodes are just flaky".

    Nothing acts on this. It exists to be *reported*, which is the correct treatment for
    evidence that has not been classified.

    Args:
        events: Events to inspect, or `None` to read them live.
        within_s: Ignore events older than this many seconds; `None` reads the whole buffer.
            Ignored when `events` is given.

    Returns:
        PCI address to the unrecognized codes seen for it, ascending and deduplicated.
    """
    return _by_address(events, within_s, lambda e: xid_severity(e.code) == "unknown")


def xid_counts(
    events: tuple[XidEvent, ...] | None = None,
    within_s: float | None = XID_WINDOW_S,
) -> dict[tuple[str, int], int]:
    """How many times each `(pci address, code)` pair was reported.

    A repeat count is a different signal from the codes themselves, and it separates two
    situations the set-valued maps above cannot. One Xid 31 is a job that indexed past the end
    of a buffer. Four hundred Xid 31s in an hour is a device whose MMU is mistranslating, and
    the fact that the code classifies as `"application"` stops being the right read — no
    workload produces that rate by being buggy. The same holds in reverse for a single Xid 63:
    the code is fatal, and one occurrence is a row remap being *recorded*, which is the device
    repairing itself successfully.

    Callers use it as a rate gate. Nothing here turns a count into a verdict, because how many
    is too many depends on the fleet and belongs with the policy that owns the thresholds.

    Args:
        events: Events to inspect, or `None` to read them live.
        within_s: Ignore events older than this many seconds; `None` reads the whole buffer.
            Ignored when `events` is given.

    Returns:
        `(pci address, code)` to occurrence count. Events with no attributed device are keyed
        on `""`, unlike the maps above that drop them: a rate is still meaningful without
        knowing which device produced it, where a quarantine decision is not.
    """
    records = recent_xid_events(within_s=within_s) if events is None else events
    out: dict[tuple[str, int], int] = {}
    for event in records:
        key = (event.pci_address, event.code)
        out[key] = out.get(key, 0) + 1
    return out
