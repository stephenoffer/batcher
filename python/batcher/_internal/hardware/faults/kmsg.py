"""The kernel ring buffer, read as timestamped records.

Two different fault readers need the same log. `xid` wants the driver's own error events;
`kernel` wants the ones the kernel raises about the node around the device — an OOM kill, a
PCIe link that retrained, a filesystem remounted read-only. Both need the same three things
from `/dev/kmsg`, and each of the three is a place a naive read goes wrong:

* **A timestamp.** Every record carries one, in microseconds since boot, and a reader that
  drops it cannot tell a fault from a minute ago from one that happened before the last
  device reset. That distinction is the difference between quarantining a bad device and
  quarantining a good one forever over an entry that has simply not aged out of a buffer.
* **Tolerance of overrun.** The ring buffer is a ring: when a reader falls behind, the kernel
  returns `EPIPE` and moves the offset to the oldest surviving record. Treating that as
  end-of-log — which is what a plain `except OSError: break` does — silently truncates the
  read at the first wrap, so a busy node reports the *fewest* faults exactly when it has the
  most.
* **An honest "unreadable".** Inside a container without the host log, and without
  `CAP_SYSLOG`, the read yields nothing and looks identical to a clean node. Callers must be
  able to tell those apart, so the reader reports which happened rather than returning an
  empty list for both.

Records are returned newest-last in the kernel's own order. Nothing here interprets a
message; that is each caller's job.

A neutral utility: any layer may import `_internal`.
"""

from __future__ import annotations

import errno
import os
import re
import time
from dataclasses import dataclass

__all__ = [
    "KMSG_PATH",
    "KmsgRecord",
    "kmsg_readable",
    "monotonic_now_s",
    "read_kmsg",
]

#: The kernel ring buffer. A constant so a test can point it at a fixture and so a deployment
#: that exposes the log elsewhere (a mounted `kmsg`, a journal export) can redirect it.
KMSG_PATH = "/dev/kmsg"

#: How many records to read before giving up. The buffer is a few hundred kilobytes, so this
#: is a bound against a pathological log rather than a sampling window — a node that really
#: holds more than this many records has a log storm, which is itself the finding.
_MAX_RECORDS = 8192

#: `<prio>,<seq>,<microseconds since boot>,<flag>[,...];<message>`. Documented in the kernel's
#: own `Documentation/ABI/testing/dev-kmsg`. Only the timestamp is parsed out of the header:
#: priority and sequence carry nothing a fault reader acts on, and a header shaped unexpectedly
#: still has a message worth reading, so the pattern is permissive about what follows. The
#: flag field is optional because a `dmesg --kernel --raw` capture and several journal exports
#: omit it, and a header one field short still carries the timestamp this exists to recover.
_HEADER_RE = re.compile(r"^(\d+),(\d+),(\d+)(?:,([^;]*))?;(.*)$")


@dataclass(frozen=True, slots=True)
class KmsgRecord:
    """One record from the kernel ring buffer.

    Attributes:
        text: The message, with the machine-readable header stripped and any continuation
            lines dropped. Continuation lines carry structured `KEY=value` metadata that no
            fault reader here uses, and keeping them would make every pattern match twice.
        timestamp_s: Seconds since boot, from the record's own header. `-1.0` when the header
            was not in the documented shape, which a caller must treat as "unknown age"
            rather than as "just now" — ageing an unknown record out of a window is how a
            live fault gets dropped.
    """

    text: str
    timestamp_s: float = -1.0

    def age_s(self, now_s: float | None = None) -> float:
        """How long ago this record was written, in seconds.

        Args:
            now_s: Current seconds since boot, or `None` to read the clock. Passing it lets a
                caller age a whole batch of records against one reading, which is both cheaper
                and self-consistent.

        Returns:
            The age, or `-1.0` when the record carried no usable timestamp.
        """
        if self.timestamp_s < 0.0:
            return -1.0
        return max(0.0, (monotonic_now_s() if now_s is None else now_s) - self.timestamp_s)


def monotonic_now_s() -> float:
    """Seconds since boot, on the same clock the kernel stamps its records with.

    The kernel writes `local_clock()`, which on Linux is the same base as
    `CLOCK_MONOTONIC` — so this is what a record's timestamp is comparable against. Falls
    back to a process-relative monotonic reading off Linux, where nothing is reading `kmsg`
    anyway and the value is only ever used to age records that do not exist.

    Returns:
        Seconds since boot.
    """
    clock = getattr(time, "CLOCK_MONOTONIC", None)
    if clock is None:  # pragma: no cover - every supported platform has it
        return time.monotonic()
    return time.clock_gettime(clock)


def kmsg_readable(path: str | None = None) -> bool:
    """Whether the kernel log can be opened at all on this host.

    The question a caller must ask before treating "no fault records" as "no faults". Inside
    a container without the host log the two are indistinguishable by the records alone, and
    conflating them means a health policy that silently stops working.

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


def _parse(line: str) -> KmsgRecord | None:
    """One log line as a `KmsgRecord`, or `None` when it carries nothing.

    A line whose header is not in the documented shape still yields its text, with an unknown
    timestamp: dropping it would lose the fault, and inventing a timestamp would age it
    wrongly in either direction. That is also what makes a plain `dmesg` dump — no headers at
    all — a usable fixture and a usable redirect target.

    A record's *continuation* lines are indented `KEY=value` metadata belonging to the record
    above, and are dropped: no fault pattern here matches against them, and keeping them would
    let one record match a pattern twice.
    """
    if not line.strip() or line[:1] in (" ", "\t"):
        return None
    match = _HEADER_RE.match(line)
    if match is None:
        return KmsgRecord(text=line.strip())
    return KmsgRecord(text=match.group(5).strip(), timestamp_s=int(match.group(3)) / 1_000_000.0)


def read_kmsg(
    path: str | None = None, *, max_records: int = _MAX_RECORDS
) -> tuple[KmsgRecord, ...]:
    """Every record currently in the kernel ring buffer, oldest first.

    Opened non-blocking so a caller is never parked waiting for the next kernel message: the
    buffer replays its history from the start of the file and would then block for new
    records, and this only wants the history.

    An overrun mid-read is *not* the end of the log. When a reader falls behind the writer the
    kernel fails that one read with `EPIPE` and moves the offset to the oldest surviving
    record, so the correct response is to read again. Stopping there instead — which is what
    catching every `OSError` alike does — truncates the history at the first wrap, and does it
    hardest on the noisiest node, which is the one whose faults matter most.

    Args:
        path: Kernel log to read, defaulting to `KMSG_PATH`.
        max_records: Ceiling on records returned.

    Returns:
        The records, empty when the log holds none *or* cannot be read — call `kmsg_readable`
        to tell those apart.
    """
    try:
        fd = os.open(path or KMSG_PATH, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return ()
    out: list[KmsgRecord] = []
    try:
        for _ in range(max_records):
            if len(out) >= max_records:
                break
            try:
                chunk = os.read(fd, 8192)
            except OSError as exc:
                if exc.errno == errno.EPIPE:
                    continue  # records were overwritten; the offset has moved to the survivors
                break  # EAGAIN: the history is exhausted and the next record has not arrived
            if not chunk:
                break
            # One `read` of `/dev/kmsg` returns exactly one record plus its continuation
            # lines; one `read` of a plain file fixture returns many. Splitting into lines
            # and letting `_parse` reject the continuations covers both without the caller
            # having to say which it pointed at.
            for line in chunk.decode("utf-8", "replace").splitlines():
                record = _parse(line)
                if record is not None:
                    out.append(record)
    finally:
        os.close(fd)
    # Sliced rather than only loop-bounded: one read of `/dev/kmsg` yields one record, but one
    # read of a file fixture yields as many as fit in the buffer, so the loop count alone does
    # not bound the result on the path a test takes.
    return tuple(out[:max_records])
