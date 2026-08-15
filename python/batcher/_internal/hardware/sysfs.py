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

A neutral utility inside a neutral package: it imports nothing, so any module here may use it
without regard to the package's internal import order.
"""

from __future__ import annotations

__all__ = ["read_float", "read_int", "read_optional_int", "read_text"]


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
