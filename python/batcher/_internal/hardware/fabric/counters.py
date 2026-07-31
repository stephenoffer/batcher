"""What an RDMA port has actually carried, and what it got wrong doing it.

`rdma` reads a port's *configuration* — its rate, its state, its partition. This reads its
history: bytes moved, and the errors the link recorded moving them. The two answer different
questions and only the second one predicts a failure.

A fabric cable does not fail cleanly. It degrades: symbol errors climb, the link recovers
itself more often, and throughput sags long before the port ever leaves `ACTIVE`. Every
scheduler upstream of that sees a healthy 400 Gb/s port and keeps placing shuffles on a node
whose cross-node stage now runs at a third of the rate. The counters are the only warning, and
they are free to read — the kernel has them open in `/sys`.

Two shapes of number, and they are used differently:

* **Data counters** (`port_xmit_data`, `port_rcv_data`) are monotonic 64-bit totals in units
  of four octets. A single reading says nothing; a *difference* over a window is throughput,
  which is what says whether a fabric a plan was priced against is actually carrying traffic.
* **Error counters** (symbol errors, link recoveries, link downs) are monotonic counts of
  faults. Here a single reading is meaningful, because a healthy cable's lifetime count is
  zero or near it — but the *rate* is what separates "this happened once during bring-up" from
  "this cable is failing now".

Every reading degrades to absence. A container without `/sys/class/infiniband` reports no
ports, a driver that does not publish a counter omits it, and a caller must treat a missing
counter as unknown rather than as zero — the distinction between a clean link and an
unreadable one is the whole point of reading them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

__all__ = [
    "ERROR_COUNTERS",
    "PortCounters",
    "fabric_error_total",
    "port_counters",
    "throughput_delta",
]

#: Error counters worth reading, with the name each is reported under. Deliberately the
#: subset an operator acts on rather than everything the kernel publishes:
#:
#: * `symbol_errors` — the physical layer failing to decode. A cable, a transceiver, or dust.
#: * `link_recovers` — the link retrained itself. Traffic stops for the duration, every time.
#: * `link_downed` — the link dropped entirely and came back. Every in-flight transfer on it
#:   failed, and something above had to retry or lose the stage.
#: * `rcv_errors` / `xmit_discards` — packets the port could not take or had to drop, which is
#:   congestion or misconfiguration rather than a bad cable, and has a different remedy.
#: * `excessive_buffer_overrun` — the far end sent faster than this port could absorb, which
#:   on a credit-controlled fabric means the flow control itself is misconfigured.
ERROR_COUNTERS: tuple[tuple[str, str], ...] = (
    ("symbol_error", "symbol_errors"),
    ("link_error_recovery", "link_recovers"),
    ("link_downed", "link_downed"),
    ("port_rcv_errors", "rcv_errors"),
    ("port_xmit_discards", "xmit_discards"),
    ("excessive_buffer_overrun_errors", "excessive_buffer_overrun"),
)

#: Data counters, with the name each is reported under. The kernel publishes these in units of
#: four octets — a detail that is easy to miss and produces a figure four times too small,
#: which is exactly the kind of error that looks like a plausible measurement.
_DATA_COUNTERS: tuple[tuple[str, str], ...] = (
    ("port_xmit_data", "xmit_bytes"),
    ("port_rcv_data", "rcv_bytes"),
)

#: The kernel's data counters count four-octet words, not bytes.
_OCTET_WORD = 4


def _read_counter(path: str) -> int | None:
    """One counter file as an int, or `None` when absent or unreadable.

    `None` rather than `0`: a counter the driver does not publish and a counter that reads
    zero mean opposite things, and collapsing them would report an unreadable fabric as a
    flawless one.
    """
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class PortCounters:
    """One RDMA port's traffic and error history.

    Attributes:
        device: RDMA device name (`"mlx5_0"`).
        port: Port number.
        xmit_bytes: Bytes transmitted since the counters were last reset, `None` when the
            driver does not publish it.
        rcv_bytes: Bytes received, `None` when unpublished.
        errors: Counter name to value, from `ERROR_COUNTERS`. A counter the driver does not
            publish is absent rather than zero.
    """

    device: str
    port: int
    xmit_bytes: int | None = None
    rcv_bytes: int | None = None
    errors: dict[str, int] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """A stable identifier for this port, for keying a sample against a later one."""
        return f"{self.device}:{self.port}"

    @property
    def total_errors(self) -> int:
        """Summed error counters, `0` when none were readable.

        A lifetime total, so it is a screening figure rather than an alarm: a link that
        recovered twice during bring-up a month ago reads the same as one recovering twice a
        minute now. `fabric_error_total` over two samples is what separates them.
        """
        return sum(self.errors.values())

    @property
    def readable(self) -> bool:
        """Whether the driver published anything at all for this port."""
        return self.xmit_bytes is not None or bool(self.errors)


def port_counters() -> tuple[PortCounters, ...]:
    """Traffic and error counters for every active RDMA port on this node.

    Not memoized, and not sampled: the value of a counter is that it moved, so a caller takes
    two readings and compares them. Costs one small file read per counter per port, which is a
    few dozen reads on an eight-port node.

    Returns:
        One record per *active* port, in device then port order. Inactive ports are omitted:
        their counters are frozen at whatever they held when the link dropped, and reporting
        them beside live ones invites reading a stale total as a current one. Empty when the
        node has no RDMA hardware or the tree is not mounted.
    """
    from batcher._internal.hardware.fabric.rdma import RDMA_SYSFS_ROOT, active_rdma_devices

    out: list[PortCounters] = []
    for device in active_rdma_devices():
        base = os.path.join(RDMA_SYSFS_ROOT, device.name, "ports", str(device.port), "counters")
        data: dict[str, int | None] = {}
        for filename, name in _DATA_COUNTERS:
            raw = _read_counter(os.path.join(base, filename))
            data[name] = None if raw is None else raw * _OCTET_WORD
        errors = {}
        for filename, name in ERROR_COUNTERS:
            value = _read_counter(os.path.join(base, filename))
            if value is not None:
                errors[name] = value
        out.append(
            PortCounters(
                device=device.name,
                port=device.port,
                xmit_bytes=data.get("xmit_bytes"),
                rcv_bytes=data.get("rcv_bytes"),
                errors=errors,
            )
        )
    return tuple(out)


def throughput_delta(
    before: tuple[PortCounters, ...],
    after: tuple[PortCounters, ...],
    seconds: float,
) -> dict[str, float]:
    """Per-port throughput in gigabits per second between two counter samples.

    The measurement that says whether the fabric a plan was priced against is carrying the
    traffic that plan generates. A shuffle estimated at 400 Gb/s and running at 12 is either
    not using the fabric at all — the usual cause is an address that resolves to the
    management NIC — or sharing it with something else.

    Args:
        before: The earlier sample.
        after: The later sample.
        seconds: Wall time between them; a non-positive value yields an empty result rather
            than a division by zero.

    Returns:
        Port key to combined transmit-plus-receive rate in Gb/s. Ports absent from either
        sample, or whose counters were unreadable, are omitted. A counter that went *backwards*
        is also omitted: that is a driver reset between samples, not negative throughput.
    """
    if seconds <= 0.0:
        return {}
    earlier = {p.key: p for p in before}
    out: dict[str, float] = {}
    for later in after:
        first = earlier.get(later.key)
        if first is None:
            continue
        moved = 0
        for a, b in ((first.xmit_bytes, later.xmit_bytes), (first.rcv_bytes, later.rcv_bytes)):
            if a is None or b is None or b < a:
                continue
            moved += b - a
        if moved:
            out[later.key] = moved * 8 / seconds / 1e9
    return out


def fabric_error_total(counters: tuple[PortCounters, ...] | None = None) -> dict[str, int]:
    """Summed error counters across the node's active ports, by counter name.

    The screening figure for a node: an operator compares it against the fleet, and a node
    that stands out has a cable to check. It is a lifetime total, so a single reading proves
    nothing on its own — two readings a stage apart do.

    Args:
        counters: A sample to summarize, or `None` to take one.

    Returns:
        Counter name to summed value, omitting counters no port published. Empty on a node
        with no readable fabric, which is not the same as a node with no errors.
    """
    sample = port_counters() if counters is None else counters
    out: dict[str, int] = {}
    for port in sample:
        for name, value in port.errors.items():
            out[name] = out.get(name, 0) + value
    return out
