"""Transfer-mode selection — move a partition the cheapest way its placement allows.

Routing every shuffle partition through a network hop (or, worse, an object store)
wastes the common case where producer and consumer are co-located. Carbonite picks
a `TransferMode` from where the data sits relative to the fetcher:

- `DIRECT_MEMORY` — same process: read it straight from the local partition store,
  no serialization, no socket. The concrete win over the Ray object store.
- `SHARED_MEMORY` — same node, different process: Arrow IPC over a memory map
  (a future Rust fast path; selected here, not yet executed — see `ShuffleSession`).
- `NETWORK` — different node: credit-bounded Arrow Flight.

The selector is pure (placement in, mode out) so it is trivially testable; the
`locality_ratio` over a batch of decisions is the metric that says how much of a
shuffle stayed off the network.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

__all__ = ["TransferMode", "locality_ratio", "locality_ratio_counts", "select_mode"]


class TransferMode(Enum):
    """How a partition is moved from producer to consumer, cheapest first."""

    DIRECT_MEMORY = "direct_memory"  # same process — read from the local store
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
    TransferMode.DIRECT_MEMORY: 0,
    TransferMode.SHARED_MEMORY: 1,
    TransferMode.NETWORK: 2,
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
