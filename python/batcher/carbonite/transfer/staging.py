"""How a transfer crosses the host link: chunk size, how many are in flight, and pinned or not.

Every byte that reaches a device from storage, from another node, or from a device the bus
keeps at arm's length crosses the host link, and the link is almost never the thing that limits
it. Three host-side choices are:

* **Chunk size.** A copy issued as one buffer cannot overlap with anything; issued as chunks it
  pipelines against the read that fills the next one. Too small and per-copy overhead dominates,
  too large and the pipeline has one stage. The bandwidth-delay product is the size that keeps
  the link busy for exactly as long as it takes to prepare the next chunk.
* **Depth.** How many chunks are in flight. Two is the minimum that overlaps at all
  (double-buffering); more covers a longer or more variable preparation time, and each one
  costs pinned host memory that no other tenant can use.
* **Pinned or pageable.** A pageable copy is staged by the driver through its own pinned buffer
  — an extra host-to-host copy at memory bandwidth — so pinning roughly doubles achieved
  throughput on a large transfer. It is also a page-table operation that is slow to perform and
  denies the pages to the rest of the machine, so it loses on a small one.

All three are decisions, which is why they are here rather than in the code doing the copying:
Carbonite sizes what a run may take from the host, and a staging ring is host memory that no
accounting had a name for. The functions are pure — link rate and byte counts in, a plan out —
so a node's behavior is testable without one.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "MAX_CHUNK_BYTES",
    "MIN_CHUNK_BYTES",
    "StagingPlan",
    "chunk_bytes_for_link",
    "effective_gbps",
    "pinned_budget_bytes",
    "pipeline_depth",
    "plan_staging",
    "staging_seconds",
    "worth_pinning",
]

#: Smallest chunk worth issuing. Below roughly this size the per-copy cost — the launch, the
#: synchronization, the accounting — is a larger share of the transfer than the transfer, and
#: the pipeline the chunking bought is spent on overhead.
MIN_CHUNK_BYTES = 256 * 1024

#: Largest chunk. A chunk this size already covers the latency of any host link in service, and
#: past it the only effect is to coarsen the pipeline and hold more pinned memory per buffer.
MAX_CHUNK_BYTES = 64 * 1024 * 1024

#: Fraction of host memory the staging rings of one process may hold. Pinned pages cannot be
#: reclaimed under pressure — that is what pinning means — so an unbounded ring does not
#: degrade the machine, it stops it.
_PINNED_FRACTION = 0.05

#: Bytes below which pinning loses. Registering pages costs a page-table walk whose duration
#: does not depend on how much is transferred afterwards, so a small copy pays the whole cost
#: for a fraction of the benefit. Set at a megabyte, which is roughly where the two meet on a
#: PCIe gen-4 link.
_PIN_WORTH_BYTES = 1024 * 1024

#: What a pinned transfer achieves against a pageable one. The pageable path stages through the
#: driver's own bounce buffer, so the payload crosses host memory an extra time; the ratio is
#: the standard, widely reproduced figure rather than a measurement from this fleet.
_PINNED_SPEEDUP = 2.0


@dataclass(frozen=True)
class StagingPlan:
    """How one stream will cross the host link.

    Attributes:
        chunk_bytes: Size of each copy issued.
        depth: How many chunks are in flight at once. `1` means no overlap, which is what a
            plan degrades to when nothing about the link is known.
        pinned: Whether the staging buffers should be page-locked.
        streams: How many of these rings run concurrently, one per device being fed.
    """

    chunk_bytes: int = MIN_CHUNK_BYTES
    depth: int = 1
    pinned: bool = False
    streams: int = 1

    @property
    def buffer_bytes(self) -> int:
        """Total host memory the plan holds: every chunk of every ring."""
        return self.chunk_bytes * self.depth * self.streams

    @property
    def overlapped(self) -> bool:
        """Whether the plan pipelines at all, or issues one copy and waits for it."""
        return self.depth > 1

    def summary(self) -> dict:
        """The plan as one flat record, for a report or a debug note.

        Returns:
            `chunk_bytes`, `depth`, `pinned`, `streams`, and the derived `buffer_bytes`.
        """
        return {
            "chunk_bytes": self.chunk_bytes,
            "depth": self.depth,
            "pinned": self.pinned,
            "streams": self.streams,
            "buffer_bytes": self.buffer_bytes,
        }


def chunk_bytes_for_link(link_gbps: float, latency_us: float = 10.0) -> int:
    """The chunk size that keeps a link of this rate busy across its own latency.

    The bandwidth-delay product: a link moving `r` bytes per second with `t` seconds of
    round-trip latency has `r * t` bytes in flight at steady state, and a chunk smaller than
    that leaves the link idle between copies. Rounded down to a power of two, because a copy
    engine's descriptor sizes are, and clamped to the module's bounds.

    Args:
        link_gbps: The link's rate in gigabytes per second, `0.0` when unknown.
        latency_us: Round-trip latency in microseconds. The default is the order of magnitude
            of a PCIe copy launch, which is what this covers.

    Returns:
        Bytes, `MIN_CHUNK_BYTES` when the rate is unknown — the small, safe chunk, which
        wastes some bandwidth and cannot exhaust anything.
    """
    if link_gbps <= 0.0 or latency_us <= 0.0:
        return MIN_CHUNK_BYTES
    product = link_gbps * 1e9 * latency_us * 1e-6
    clamped = min(max(product, MIN_CHUNK_BYTES), MAX_CHUNK_BYTES)
    return 1 << int(clamped).bit_length() - 1


def pipeline_depth(chunk_bytes: int, link_gbps: float, prepare_us: float = 0.0) -> int:
    """How many chunks must be in flight to cover the time spent preparing the next one.

    A ring of two overlaps one copy with one preparation and is the floor for any pipeline. It
    is enough only while preparation is no slower than the copy; a decode that takes three
    times as long needs four buffers before the link stops waiting on it.

    Args:
        chunk_bytes: The chunk size, from `chunk_bytes_for_link`.
        link_gbps: The link's rate in gigabytes per second.
        prepare_us: How long the producer takes to fill one chunk, in microseconds. `0.0`
            (unknown or free) yields plain double-buffering.

    Returns:
        A depth of at least `1`, or `2` and up once the link's rate is known. Capped at `8`:
        past that the ring is holding more pinned memory than any variance it absorbs is worth.
    """
    if chunk_bytes <= 0 or link_gbps <= 0.0:
        return 1
    copy_us = chunk_bytes / (link_gbps * 1e9) * 1e6
    if prepare_us <= 0.0 or copy_us <= 0.0:
        return 2
    return max(2, min(8, 1 + int(prepare_us / copy_us + 0.999)))


def worth_pinning(transfer_bytes: int, *, threshold: int = _PIN_WORTH_BYTES) -> bool:
    """Whether page-locking the staging buffer pays for a transfer of this size.

    Args:
        transfer_bytes: Bytes the ring will carry over its lifetime, not per chunk. A ring
            reused across a stage amortizes its pinning cost over everything it moves, and
            sizing the decision on one chunk would refuse it on every stage.
        threshold: Bytes above which pinning wins.

    Returns:
        True when pinning is worth its page-table cost.
    """
    return transfer_bytes >= max(0, threshold)


def pinned_budget_bytes(host_bytes: int, fraction: float = _PINNED_FRACTION) -> int:
    """The most host memory this process's staging rings may page-lock.

    Args:
        host_bytes: The machine's usable host memory, `0` when unknown.
        fraction: Share of it the rings may hold.

    Returns:
        Bytes, `0` when host memory is unknown — which reads as "do not pin", the answer that
        cannot wedge a machine.
    """
    if host_bytes <= 0 or fraction <= 0.0:
        return 0
    return int(host_bytes * min(fraction, 1.0))


def plan_staging(
    transfer_bytes: int,
    link_gbps: float,
    *,
    streams: int = 1,
    latency_us: float = 10.0,
    prepare_us: float = 0.0,
    host_bytes: int = 0,
) -> StagingPlan:
    """The whole host-side plan for a transfer: chunk, depth, pinning, and how many rings.

    The budget is enforced last and by *shrinking the depth*, not the chunk. A shallower ring
    costs some overlap; a smaller chunk costs the link's steady state, and once the chunk is
    below the bandwidth-delay product the transfer runs slower than it would have with no
    pipelining at all. When even a depth of one does not fit, the plan is unpinned, which
    hands the pages back and lets the driver stage the copy itself.

    Args:
        transfer_bytes: Bytes the rings will carry in total.
        link_gbps: The host link's rate in gigabytes per second, `0.0` when unknown.
        streams: How many concurrent rings, usually one per device being fed.
        latency_us: Round-trip latency of the link in microseconds.
        prepare_us: How long the producer takes to fill one chunk, in microseconds.
        host_bytes: The machine's usable host memory, `0` when unknown.

    Returns:
        A `StagingPlan`. Unknown link rate yields the small unpinned chunk with no overlap,
        which is the behavior a caller had before it consulted anything.
    """
    streams = max(1, streams)
    chunk = chunk_bytes_for_link(link_gbps, latency_us)
    depth = pipeline_depth(chunk, link_gbps, prepare_us)
    pinned = worth_pinning(transfer_bytes)
    budget = pinned_budget_bytes(host_bytes)
    if pinned and budget > 0:
        per_stream = budget // streams
        if per_stream < chunk:
            pinned = False
            depth = min(depth, 2)
        else:
            depth = max(1, min(depth, per_stream // chunk))
    return StagingPlan(chunk_bytes=chunk, depth=depth, pinned=pinned, streams=streams)


def effective_gbps(plan: StagingPlan, link_gbps: float) -> float:
    """What a plan actually achieves on a link of this nominal rate.

    Two derations, both of which a caller sizing a stage against nameplate bandwidth misses. A
    plan with no overlap alternates copying and preparing, so it reaches roughly half the link.
    A pageable plan crosses host memory an extra time through the driver's bounce buffer.

    Args:
        plan: The staging plan.
        link_gbps: The link's nominal rate in gigabytes per second.

    Returns:
        Gigabytes per second, `0.0` when the link's rate is unknown.
    """
    if link_gbps <= 0.0:
        return 0.0
    rate = link_gbps if plan.overlapped else link_gbps * 0.5
    return rate if plan.pinned else rate / _PINNED_SPEEDUP


def staging_seconds(transfer_bytes: int, plan: StagingPlan, link_gbps: float) -> float:
    """How long a transfer takes under a plan, in seconds.

    Args:
        transfer_bytes: Bytes to move, across every stream.
        plan: The staging plan.
        link_gbps: The link's nominal rate in gigabytes per second.

    Returns:
        Seconds, `0.0` when there is nothing to move or no rate is known. Streams are assumed
        to share the one host link, so their concurrency divides the bytes each carries and
        not the time the transfer takes — the optimistic reading is the one that would have a
        caller size a stage for bandwidth the node does not have.
    """
    rate = effective_gbps(plan, link_gbps)
    if transfer_bytes <= 0 or rate <= 0.0:
        return 0.0
    return transfer_bytes / (rate * 1e9)
