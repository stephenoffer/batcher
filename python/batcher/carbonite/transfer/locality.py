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
    """
    if source_addr == local_addr:
        return TransferMode.DIRECT_MEMORY
    if source_node is not None and local_node is not None and source_node == local_node:
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
    off_network = sum(m is not TransferMode.NETWORK for m in modes)
    return off_network / len(modes)


def locality_ratio_counts(off_network: int, total: int) -> float:
    """Locality ratio from running counters (off-network fetches / total fetches).

    The counter form a long-lived reducer accumulates instead of a per-fetch list,
    which would grow without bound. Empty (no fetches) is 1.0 by the same
    convention as `locality_ratio`.
    """
    if total <= 0:
        return 1.0
    return off_network / total
