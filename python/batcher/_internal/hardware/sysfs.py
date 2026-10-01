"""Reading a kernel pseudo-file, where "absent" means "unknown" rather than "error".

Almost every probe in this package answers its question by reading one attribute out of
`/sys` or `/proc`. Those reads fail in four ordinary ways that are all *unknown* rather than
a fault: the attribute does not exist on this kernel or driver version, the container never
mounted the tree, the driver returns `EINVAL` for a figure the part does not support, and the
file exists but holds something unparseable (`"N/A"`, an empty string, a hex value where the
caller wanted decimal). A probe that let any of those raise would turn "this machine does not
report its NUMA distance" into a failed query.

So each module grew its own four-line `try: open(...) except OSError: return ""`. There were
**eight** of them — `cache._read_int`/`_read_str`, `memory._read_int`, `storage._read_int`,
`amd.devices._read_text`/`_read_int`, `fabric.ethernet._read`, `fabric.pcie._read_text`,
`fabric.rdma._read_text`, `fabric.counters._read_counter` — spread over the package root and
two subpackages, and they had quietly drifted apart on the one decision that matters:

**what an unreadable file means.** Three conventions were live at once. `""`/`0` says "absent
and zero are the same thing", which is right for a *capacity* (a cache level that is not
there has no size). `None` says they are opposite, which is right for a *counter* — a fabric
that publishes no error counter and a fabric reporting zero errors must not both look
flawless, and `fabric.counters` carries a comment saying exactly that. Nothing named the
distinction, so which one a new probe got depended on which neighbour it was copied from.

This module makes the choice explicit in the function name: `read_text`/`read_int` fold the
absent case into a caller-supplied default, and `read_optional_int` keeps it distinct. That
is the whole reason to have one home for five lines of `open`.

A neutral utility inside a neutral package: it imports nothing of Batcher's, so any module
here may use it without regard to the package's internal import order.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import os
import threading

__all__ = ["read_float", "read_int", "read_live_text", "read_optional_int", "read_text"]

#: Prefixes of the kernel-generated trees whose attributes `read_live_text` holds open. Only
#: these: a kernel attribute is regenerated on every read of the *same* descriptor, so a held
#: descriptor reads the live value. A regular file is not -- a test fixture or config file
#: replaced by rename would keep serving its old inode -- so anything else is opened per read.
_LIVE_PREFIXES = ("/sys/fs/cgroup/", "/proc/")

#: Read size for one `pread`. Every attribute this is used for is a few dozen bytes; a full
#: buffer means the file is larger than that and the read continues at the next offset.
_LIVE_CHUNK = 65536

#: Held descriptors, by path. Process-local: cleared in a forked child (see below).
_LIVE_FDS: dict[str, int] = {}
_LIVE_LOCK = threading.Lock()

#: The interpreter's own `open`, to tell when something has replaced it.
_REAL_OPEN = io.open


def _forget_live_fds() -> None:
    """Close and drop every held descriptor (the post-fork hook, and the test reset)."""
    with _LIVE_LOCK:
        fds = list(_LIVE_FDS.values())
        _LIVE_FDS.clear()
    for fd in fds:
        with contextlib.suppress(OSError):
            os.close(fd)


if hasattr(os, "register_at_fork"):
    # A forked child inherits the descriptors, which still name the parent's cgroup files. A
    # child moved into its own cgroup (a Ray worker) must read *its* charge, so it re-opens.
    os.register_at_fork(after_in_child=_forget_live_fds)


def _pread_all(fd: int) -> bytes:
    """The whole attribute behind `fd`, read from offset 0 (a fresh kernel snapshot)."""
    chunk = os.pread(fd, _LIVE_CHUNK, 0)
    if len(chunk) < _LIVE_CHUNK:
        return chunk
    parts = [chunk]
    offset = len(chunk)
    while chunk:
        chunk = os.pread(fd, _LIVE_CHUNK, offset)
        parts.append(chunk)
        offset += len(chunk)
    return b"".join(parts)


def read_live_text(path: str) -> str | None:
    """A kernel attribute's current contents, or `None` when it cannot be read.

    For the attributes the control plane re-reads on every query -- the cgroup's memory
    charge above all, read several times per terminal op by the pressure ladder. Opening the
    file is most of what a read costs: measured in a cgroup v2 container, `open().read()` on
    `memory.current` took **49 us** and a `pread` at offset 0 on a held descriptor **5.6 us**,
    because the open walks the overlay and cgroupfs path lookup the held descriptor has
    already paid for.

    The reading is not cached. The kernel regenerates a pseudo-file's contents on every read
    from offset 0, so a held descriptor returns exactly what a fresh `open` would, at the
    moment of the call -- the values move between consecutive reads. Only the descriptor is
    kept, and only for paths under `/sys/fs/cgroup/` and `/proc/`; any other path is opened
    per call, because a regular file replaced by rename would keep serving its old inode
    through a held descriptor. A read that fails on a held descriptor (the cgroup was
    removed) drops it and retries with a fresh open, so the failure modes are those of a
    plain open.

    Args:
        path: Absolute path to the attribute.

    Returns:
        The decoded contents, or `None` when the file is absent or unreadable.
    """
    if not path.startswith(_LIVE_PREFIXES) or builtins.open is not _REAL_OPEN:
        # A replaced `open` (a test faking `/sys`, an embedding that audits file access) is
        # honored: the held-descriptor path goes to the OS directly and would bypass it.
        try:
            with open(path) as f:
                return f.read()
        except OSError:
            return None
    fd = _LIVE_FDS.get(path)
    if fd is not None:
        try:
            return _pread_all(fd).decode()
        except OSError:
            # Dropped but deliberately not closed: another thread may be mid-`pread` on the
            # same number, and closing it here would let the next `open` anywhere in the
            # process reuse it -- that reader would then parse an unrelated file. One leaked
            # descriptor on a removed cgroup is the cheap side of that trade.
            with _LIVE_LOCK:
                if _LIVE_FDS.get(path) == fd:
                    del _LIVE_FDS[path]
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    try:
        raw = _pread_all(fd).decode()
    except OSError:
        os.close(fd)
        return None
    with _LIVE_LOCK:
        held = _LIVE_FDS.setdefault(path, fd)
    if held != fd:  # another thread opened it first; keep theirs
        os.close(fd)
    return raw


def read_text(path: str, default: str = "") -> str:
    """The stripped contents of a kernel pseudo-file, or `default` when unreadable.

    Args:
        path: Absolute path to the attribute, such as ``/sys/block/sda/queue/rotational``.
        default: What to return when the file is missing or cannot be read.

    Returns:
        The file's contents with surrounding whitespace removed, or `default`.
    """
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def read_int(path: str, default: int = 0, *, base: int = 10) -> int:
    """One integer attribute, or `default` when absent, unreadable, or unparseable.

    Use this when "the file is not there" and "the file says zero" mean the same thing to the
    caller — a cache level that does not exist has no line size. When they mean opposite
    things, use `read_optional_int` instead.

    Args:
        path: Absolute path to the attribute.
        default: What to return when the file is missing or does not hold an integer.
        base: Radix to parse in. Kernel PCI identity attributes are hexadecimal.

    Returns:
        The parsed integer, or `default`.
    """
    raw = read_text(path)
    if not raw:
        return default
    try:
        return int(raw, base)
    except ValueError:
        return default


def read_optional_int(path: str, *, base: int = 10) -> int | None:
    """One integer attribute, or `None` when absent, unreadable, or unparseable.

    The counter-shaped counterpart to `read_int`: `None` rather than `0`, because a counter
    the driver does not publish and a counter that reads zero mean opposite things, and
    collapsing them would report an unreadable fabric as a flawless one.

    Args:
        path: Absolute path to the attribute.
        base: Radix to parse in.

    Returns:
        The parsed integer, or `None` when the figure is unavailable.
    """
    raw = read_text(path)
    if not raw:
        return None
    try:
        return int(raw, base)
    except ValueError:
        return None


def read_float(path: str, default: float = 0.0, *, scale: float = 1.0) -> float:
    """One integer attribute divided by `scale`, or `default` when unavailable.

    Kernel attributes publish fixed-point figures as integers in a driver-specific unit --
    millidegrees, microwatts, mebibytes -- so the read and the unit conversion belong
    together rather than leaving each caller to remember the divisor.

    Args:
        path: Absolute path to the attribute.
        default: What to return when the file is missing or does not hold an integer.
        scale: Divisor taking the kernel's unit to the caller's, e.g. ``1000.0`` for
            millidegrees to degrees.

    Returns:
        The scaled reading, or `default`.
    """
    raw = read_text(path)
    if not raw:
        return default
    try:
        return int(raw) / scale
    except ValueError:
        return default
