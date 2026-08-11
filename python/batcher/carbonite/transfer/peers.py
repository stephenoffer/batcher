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

__all__ = [
    "PeerTransfer",
    "bdp_bytes",
    "flow_totals",
    "peer_transfers",
    "reset_peer_transfers",
    "starved_ratio",
    "straggler_peer",
]

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
        starved_seconds: The part of `seconds` spent blocked awaiting the next batch — the
            credit window's own feedback. See `starved_fraction`.
    """

    addr: str
    bytes: int = 0
    seconds: float = 0.0
    fetches: int = 0
    retries: int = 0
    starved_seconds: float = 0.0

    @property
    def gbps(self) -> float:
        """Per-stream throughput from this peer in gigabits per second, `0.0` when unmeasured.

        Zero is "no opinion" and every consumer here reads it that way: a peer nothing has
        fetched from must not rank as the slowest one.
        """
        return self.bytes * 8 / self.seconds / 1e9 if self.seconds > 0 else 0.0

    @property
    def starved_fraction(self) -> float | None:
        """Share of this peer's fetch time spent waiting for data, or `None` if unmeasured.

        A credit window exists to cover the channel's bandwidth-delay product: enough batches
        in flight that the consumer never waits on the wire, and not one more, because every
        credit past that point is buffered memory bought for no throughput. This is the only
        figure that says which side of that line a channel is on.

        `None` rather than `0.0` for an unmeasured peer, because the two mean opposite things
        to a controller: zero says "already wide enough, stop growing", and a channel nobody
        has fetched from has not earned that verdict.
        """
        if self.seconds <= 0:
            return None
        return min(1.0, max(0.0, self.starved_seconds / self.seconds))


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
        PeerTransfer(
            addr=addr,
            bytes=int(b),
            seconds=float(s),
            fetches=int(f),
            retries=int(r),
            starved_seconds=float(w),
        )
        for addr, b, s, f, r, w in raw
    )


def flow_totals() -> tuple[float, float]:
    """This process's running `(starved_seconds, total_seconds)` across every shuffle peer.

    Totals rather than a ratio, because a credit controller acts once per round and needs how
    the channel behaved *during that round*. A lifetime ratio cannot say: after a few seconds
    of a long shuffle its denominator is large enough that a round of pure starvation barely
    moves it, so a controller reading it converges to a number and stops responding to the
    link. `carbonite.policies.congestion.StarvationMeter` differences these into the interval
    the control law is defined over.

    Byte-weighted by construction — a peer that carried more of the shuffle contributed more
    of both clocks — so a single trivial fetch cannot swing a verdict the way averaging
    per-peer ratios would.

    Returns:
        The pair in seconds, `(0.0, 0.0)` on a process that has fetched nothing and on a build
        whose engine predates the counter. Both mean "no opinion", never "saturated".
    """
    from batcher._internal.native import engine

    try:
        starved, total = engine().shuffle_flow_totals()
    except Exception as exc:  # an older extension has no such symbol
        note_suppressed("carbonite", "read the shuffle starvation counters", exc)
        return (0.0, 0.0)
    return (float(starved), float(total))


def bdp_bytes() -> int | None:
    """The widest path's bandwidth-delay product in bytes, or `None` when unmeasured.

    `BtlBw x RTprop` — the bytes a path holds in flight when it is exactly busy, and the target
    a credit window should be sized to rather than probed toward. The transport keeps both
    terms as filters rather than averages: the delay as a running minimum and the rate as a
    running maximum, because every error in the first is non-negative and every error in the
    second is non-positive.

    The maximum across peers, because one credit window serves every channel a session fetches
    on: sizing to the median starves the longest path.

    Returns:
        The product in bytes, or `None` on a process that has completed no fetch and on a build
        whose engine predates the filters. Both mean "no estimate", never "a pipe of no width".
    """
    from batcher._internal.native import engine

    try:
        measured = engine().shuffle_bdp_bytes()
    except Exception as exc:  # an older extension has no such symbol
        note_suppressed("carbonite", "read the shuffle bandwidth-delay product", exc)
        return None
    return None if measured is None or measured <= 0 else int(measured)


def starved_ratio() -> float | None:
    """Share of this process's shuffle-fetch time spent waiting for data, or `None`.

    The *lifetime* reading, for a diagnosis rather than for a control loop — see `flow_totals`
    for why a controller must difference the totals instead.

    Returns:
        The ratio in `[0, 1]`, or `None` when nothing has been fetched.
    """
    starved, total = flow_totals()
    return None if total <= 0 else min(1.0, max(0.0, starved / total))


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
