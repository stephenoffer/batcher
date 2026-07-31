"""Transfer-mode selection — move a partition the cheapest way its placement allows.

Routing every shuffle partition through a network hop (or, worse, an object store)
wastes the common case where producer and consumer are co-located. Carbonite picks
a `TransferMode` from where the data sits relative to the fetcher:

- `DEVICE_LOCAL` — the bytes are already in the memory of the device that wants them:
  no copy at all, which is the mode a device-resident pipeline exists to produce.
- `DIRECT_MEMORY` — same process: read it straight from the local partition store,
  no serialization, no socket. The concrete win over the Ray object store.
- `DEVICE_P2P` — two devices on one node with a direct path between them: the copy
  crosses the fabric or a PCIe switch and never reaches host memory.
- `SHARED_MEMORY` — same node, different process: Arrow IPC over a memory map
  (a future Rust fast path; selected here, not yet executed — see `ShuffleSession`).
- `NETWORK` — different node: credit-bounded Arrow Flight.

The two device modes rank where they do because the consumer is a device. A peer copy
on the fabric beats a memory map that still has to cross the host link afterwards, and
being already resident beats every mode including the free host one. On a node with no
accelerators neither is ever selected, so the ordering costs an existing caller nothing.

The selectors are pure (placement in, mode out) so they are trivially testable; the
`locality_ratio` over a batch of decisions is the metric that says how much of a
shuffle stayed off the network.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

__all__ = [
    "TransferMode",
    "locality_ratio",
    "locality_ratio_counts",
    "select_device_mode",
    "select_mode",
]


class TransferMode(Enum):
    """How a partition is moved from producer to consumer, cheapest first."""

    DEVICE_LOCAL = "device_local"  # already resident on the consuming device — no copy
    DIRECT_MEMORY = "direct_memory"  # same process — read from the local store
    DEVICE_P2P = "device_p2p"  # two devices, one node — peer copy, no host bounce
    SHARED_MEMORY = "shared_memory"  # same node, other process — Arrow IPC / mmap
    NETWORK = "network"  # different node — credit-bounded Flight

    @property
    def rank(self) -> int:
        """Relative cost, `0` cheapest. `DIRECT_MEMORY` < `SHARED_MEMORY` < `NETWORK`.

        The enum is declared cheapest-first and says so, but an `Enum` carries no order, so
        anything wanting to compare two modes had to re-encode the ranking at the call
        site — three places that must agree with this declaration and with each other.
        """
        return _COST_RANK[self]

    @property
    def is_local(self) -> bool:
        """Whether this mode keeps the bytes off the network (direct or shared memory)."""
        return self is not TransferMode.NETWORK


_COST_RANK = {
    TransferMode.DEVICE_LOCAL: 0,
    TransferMode.DIRECT_MEMORY: 1,
    TransferMode.DEVICE_P2P: 2,
    TransferMode.SHARED_MEMORY: 3,
    TransferMode.NETWORK: 4,
}


def select_mode(
    source_addr: str,
    local_addr: str,
    *,
    source_node: str | None = None,
    local_node: str | None = None,
) -> TransferMode:
    """Pick the transfer mode for fetching from `source_addr` into `local_addr`.

    Same Flight address ⇒ same process ⇒ `DIRECT_MEMORY`. Otherwise, when both
    node identities are known and equal ⇒ same host ⇒ `SHARED_MEMORY`. Everything
    else ⇒ `NETWORK`. Node identity is optional: with none supplied the selector
    conservatively treats a different address as remote.

    **Two unknowns are not a match.** An empty address means "not known yet" — a server
    that has not bound, a worker record built before its port was assigned — and equality
    between two of them is an artifact of both being absent, not evidence of anything.
    Read as `DIRECT_MEMORY` it sends the fetcher to a local store that does not hold the
    bucket, which surfaces as a missing partition rather than as the address bug it is.
    Unknown resolves to `NETWORK`: the mode that is always *correct*, only ever slower.
    """
    if source_addr and local_addr and source_addr == local_addr:
        return TransferMode.DIRECT_MEMORY
    if source_node and local_node and source_node == local_node:
        return TransferMode.SHARED_MEMORY
    return TransferMode.NETWORK


def select_device_mode(
    source_device: int,
    local_device: int,
    *,
    host_mode: TransferMode = TransferMode.NETWORK,
    direct: bool = False,
) -> TransferMode:
    """Pick the mode for moving a *device-resident* buffer to the device that wants it.

    The question `select_mode` cannot answer, because its inputs describe host processes and
    the answer turns on which device holds the bytes. Same device is `DEVICE_LOCAL` — the
    buffer is already where it is needed. Two devices with a direct path between them
    (`p2p.p2p_capable`) is `DEVICE_P2P`. Anything else falls back to `host_mode`, the mode the
    caller would have used had it never asked: the bytes cross host memory either way, and
    inventing a device mode for a copy that is really a host copy would over-count the
    fabric's share of a shuffle.

    A negative device index means "not on a device" — a host-side producer, a partition read
    from storage — and resolves to `host_mode` for the same reason. Two unknowns are not a
    match, exactly as in `select_mode`: equality between two absent device ids is an artifact
    of both being absent, and reading it as `DEVICE_LOCAL` would send a consumer to a buffer
    that is not there.

    Args:
        source_device: Device ordinal holding the bytes, negative when they are not on one.
        local_device: Device ordinal that wants them, negative when the consumer is the host.
        host_mode: What to report when the pair cannot copy device-to-device. Pass the result
            of `select_mode` so the two selectors compose into one answer.
        direct: Whether the pair has a direct device-to-device path, from `p2p.p2p_capable`.
            Defaults to False, so a caller that has not read the topology gets the host answer
            rather than a peer copy the bus cannot perform.

    Returns:
        The cheapest mode the placement allows.
    """
    if source_device < 0 or local_device < 0:
        return host_mode
    if source_device == local_device:
        return TransferMode.DEVICE_LOCAL
    return TransferMode.DEVICE_P2P if direct else host_mode


def locality_ratio(modes: Iterable[TransferMode]) -> float:
    """Fraction of transfers that stayed off the network (direct or shared memory).

    1.0 means a fully co-located shuffle (no bytes hit a socket); 0.0 means every
    partition crossed the network. Empty input is treated as fully local (1.0).
    """
    modes = list(modes)
    if not modes:
        return 1.0
    return sum(m.is_local for m in modes) / len(modes)


def locality_ratio_counts(off_network: int, total: int) -> float:
    """Locality ratio from running counters (off-network fetches / total fetches).

    The counter form a long-lived reducer accumulates instead of a per-fetch list,
    which would grow without bound. Empty (no fetches) is 1.0 by the same
    convention as `locality_ratio`.

    Clamped to `[0, 1]`. These are two independently-incremented counters, so a
    double-count on one of them yields a "ratio" above 1.0 — and this figure is read as a
    fraction by the tuning loop, where a 1.3 does not look like corrupt input, it looks
    like a strong signal and is acted on.

    Args:
        off_network: Fetches served without a socket.
        total: Fetches attempted.

    Returns:
        The fraction in `[0, 1]`; `1.0` when nothing has been fetched.
    """
    if total <= 0:
        return 1.0
    return min(1.0, max(0.0, off_network / total))
