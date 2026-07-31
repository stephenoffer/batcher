"""What each peer carried, so a slow shuffle can name the node it was slow on.

The node's own port counters (`fabric_usage`) measure the whole machine, including whatever
else is on it, and cannot attribute a rate to the peer it came from. The Flight client can:
it holds the bytes it decoded and the time it spent doing it, per peer. This is the
control-plane reading of those counters.

The figure it produces is a *straggler* diagnosis, which is the one a shuffle's existing
statistics cannot give. A locality ratio says where the bytes came from and a credit window
says how the producer was paced, and a fleet with one node on a renegotiated link reads
exactly like a fleet that is uniformly busy. Per peer it does not: one address at a fraction
of the others' rate is a node to drain.

Every entry point degrades to nothing on an engine build that predates the counters, so a
worker running an older extension keeps the statistics it had rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher._internal.logging import note_suppressed

__all__ = ["PeerTransfer", "peer_transfers", "reset_peer_transfers", "straggler_peer"]

#: Bytes a peer must have carried before its rate is trusted. A single small fetch measures
#: the connection setup, not the wire, and naming that peer the fleet's straggler sends an
#: operator to inspect a healthy node.
_MIN_BYTES = 4 * 1024 * 1024

#: How far below the fleet's median a peer must sit to be called a straggler. A fleet's rates
#: vary by a factor well under two through ordinary contention; a peer at a third of the
#: median is a different condition.
_STRAGGLER_RATIO = 0.5


@dataclass(frozen=True, slots=True)
class PeerTransfer:
    """What one shuffle peer carried into this process.

    Attributes:
        addr: The peer's advertised Flight address.
        bytes: Bytes decoded from it.
        seconds: Time spent fetching, summed across fetches rather than elapsed. A reducer
            striping one bucket over several flows contributes each flow's own duration.
        fetches: Fetches completed.
        retries: Fetches that had to be redialed because the cached connection was stale.
    """

    addr: str
    bytes: int = 0
    seconds: float = 0.0
    fetches: int = 0
    retries: int = 0

    @property
    def gbps(self) -> float:
        """Per-stream throughput from this peer in gigabits per second, `0.0` when unmeasured.

        Zero is "no opinion" and every consumer here reads it that way: a peer nothing has
        fetched from must not rank as the slowest one.
        """
        return self.bytes * 8 / self.seconds / 1e9 if self.seconds > 0 else 0.0


def peer_transfers() -> tuple[PeerTransfer, ...]:
    """Every peer this process has fetched from, ascending by address.

    Returns:
        The records, empty on a build whose engine does not carry the counters and on a
        process that has not fetched anything.
    """
    from batcher._internal.native import engine

    try:
        raw = engine().shuffle_peer_stats()
    except Exception as exc:  # an older extension has no such symbol
        note_suppressed("carbonite", "read the shuffle peer counters", exc)
        return ()
    return tuple(
        PeerTransfer(addr=addr, bytes=int(b), seconds=float(s), fetches=int(f), retries=int(r))
        for addr, b, s, f, r in raw
    )


def reset_peer_transfers() -> None:
    """Forget every peer's totals, so the next reading measures one stage.

    Best-effort: a build without the counters has nothing to reset, and a caller that measures
    a stage against a reset it did not get sees the process's totals, which is a superset
    rather than a wrong answer.
    """
    from batcher._internal.native import engine

    try:
        engine().reset_shuffle_peer_stats()
    except Exception as exc:  # an older extension has no such symbol
        note_suppressed("carbonite", "reset the shuffle peer counters", exc)


def straggler_peer(
    transfers: tuple[PeerTransfer, ...] | None = None,
    *,
    min_bytes: int = _MIN_BYTES,
    ratio: float = _STRAGGLER_RATIO,
) -> PeerTransfer | None:
    """The peer that is slow relative to the rest of the fleet, or `None` when none is.

    Compared against the *median* rather than the mean, because the mean of a fleet with one
    very slow member is dragged toward that member and hides it. A fleet of two has no useful
    median and reports nothing: with one peer either side of any threshold, "half the fleet is
    slow" is not a straggler diagnosis.

    Args:
        transfers: Records to examine, or `None` to read them live.
        min_bytes: Bytes a peer must have carried before its rate is trusted.
        ratio: Fraction of the median below which a peer is a straggler.

    Returns:
        The slowest qualifying peer, or `None` when the fleet is even, too small, or
        unmeasured.
    """
    records = peer_transfers() if transfers is None else transfers
    measured = sorted(
        (t for t in records if t.bytes >= min_bytes and t.gbps > 0.0), key=lambda t: t.gbps
    )
    if len(measured) < 3:
        return None
    rates = [t.gbps for t in measured]
    middle = len(rates) // 2
    median = rates[middle] if len(rates) % 2 else (rates[middle - 1] + rates[middle]) / 2
    slowest = measured[0]
    return slowest if median > 0 and slowest.gbps < median * ratio else None
