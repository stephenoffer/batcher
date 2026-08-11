"""What one round of a data channel actually observed, as a three-state congestion verdict.

A credit window has exactly one job: cover the channel's bandwidth-delay product. Enough
batches in flight that the consumer never waits on the wire, and not one more, because every
credit past that point is buffered memory bought for no throughput.

Deciding that needs two independent facts, and the credit loop only ever had one of them.

**Is the node in trouble?** `PressureMonitor` answers this, and it is the signal the AIMD
controller has always backed off on. It is necessary and it is not sufficient: it is a
property of the *machine*, so it cuts every channel on the node including the ones that were
behaving, and on a healthy node it never fires at all.

**Is the window doing anything?** Nothing answered this. With memory as the only evidence,
every uncongested round grew the window, so a channel on a quiet node climbed to its ceiling
whether or not the extra credits moved a single byte — which is a ramp, not a control loop,
and is how the memory pressure it eventually backs off on gets manufactured in the first
place. `bc-transport` now measures the consumer's own wait
(`carbonite.transfer.peers.flow_totals`), and its complement is what this module reads:
*occupancy*, the share of a fetch during which the consumer had data to work on.

**Both facts are node-level, and that is deliberate rather than a shortcut.** The transport's
clocks are process-wide, summed over every peer, so occupancy answers "was this worker ever
waiting on the wire" and not "was this one channel". That is the question a credit window
should be sized by: a window is granted per channel, but the memory it buys is the node's, and
a reducer with thirty-two channels that is comfortably fed does not need any of them widened.
The consequence to keep in mind is that several controllers in one process read one shared
measurement — each sees the interval since *its own* last sample, across all channels — while
the hysteresis state below stays per-controller.

The three verdicts and what a controller does with each:

- `STARVED` — the consumer waited on the wire. The window is below the BDP; grow it.
- `SATURATED` — the producer stayed ahead. The window already covers the BDP; hold. Growing
  buys buffered memory and nothing else.
- `CONGESTED` — the node is past its spill threshold. Cut, whatever the channel measured;
  protecting the node outranks filling the link.

The band between the two occupancy thresholds is a Schmitt trigger, not a midpoint. A single
threshold on a noisy ratio flaps, and a flapping verdict drives a window that grows and cuts
on alternate rounds — which is worse than either decision taken consistently. `flow_control`'s
`backpressure_high` and `backpressure_low` are that band. They have been declared, validated
and documented as buffer-occupancy thresholds since the config existed and had no runtime
consumer; this is the consumer, reading them with the meaning they were always given.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

from batcher.config import Config, active_config

if TYPE_CHECKING:
    from batcher.carbonite.memory.pressure import PressureMonitor

__all__ = [
    "ChannelCongestion",
    "CongestionSignal",
    "StarvationMeter",
    "occupancy_from_starvation",
    "probe_pressure",
]

#: Seconds of measured transfer a round must carry before its occupancy is believed.
#:
#: The interval between two credit rounds can contain a single tiny fetch, or a
#: locality-served bucket that never touched the wire at all. Dividing two nearly-equal
#: clocks there produces a ratio built from scheduler jitter, and a controller that acts on
#: it flaps. Below the floor the round reports "no opinion" and the hysteresis band holds
#: whatever the last real measurement said.
_MIN_MEASURED_SECONDS = 1e-3


class CongestionSignal(IntEnum):
    """One round's verdict on a credit-bounded channel.

    Ordered by how much the controller should hold back, so `max()` over several channels'
    verdicts is the conservative fusion and a comparison reads the way it looks.

    Examples:
        .. doctest::

            >>> from batcher.carbonite.policies import CongestionSignal
            >>> CongestionSignal.CONGESTED > CongestionSignal.STARVED
            True
    """

    #: The consumer waited on the wire: the window is below the bandwidth-delay product.
    STARVED = 0
    #: The producer stayed ahead: the window already covers the BDP, so hold it there.
    SATURATED = 1
    #: The node is past its spill threshold: cut, whatever this channel measured.
    CONGESTED = 2


def occupancy_from_starvation(starved: float | None) -> float | None:
    """Turn a measured starvation ratio into the buffer occupancy the thresholds are in.

    The transport measures how long the consumer *waited*; the config thresholds are stated
    as how full the pipeline *was*. They are complements, and converting once here is what
    keeps every reader of `backpressure_high` from having to remember which way round it is.

    Args:
        starved: Share of fetch time spent awaiting the next batch, or `None` if unmeasured.

    Returns:
        The occupancy in `[0, 1]`, or `None` when there is nothing to convert — which every
        caller must read as "no opinion" rather than as an idle channel.
    """
    if starved is None:
        return None
    return min(1.0, max(0.0, 1.0 - starved))


class StarvationMeter:
    """Turns the transport's lifetime flow clocks into a per-round starvation ratio.

    The transport keeps two running totals — time spent fetching, and the part of it spent
    waiting for the next batch. A controller acts once per round, so what it needs is the
    ratio *over that round*, and the lifetime ratio cannot supply it: a few seconds into a
    long shuffle the denominator is large enough that a round of pure starvation barely moves
    it, and the controller converges on a number and stops responding to the link at all.

    Differencing against the previous reading gives the interval. Stateful — hold one per
    credit controller, alongside its `ChannelCongestion`. The totals it reads are the
    process's, summed over every peer, so with several controllers in one process each sees
    the interval since its own last sample across all of them.
    """

    __slots__ = ("_seen_starved", "_seen_total")

    def __init__(self) -> None:
        self._seen_starved = 0.0
        self._seen_total = 0.0

    def sample(self, totals: tuple[float, float]) -> float | None:
        """Difference `totals` against the last reading; return this round's ratio.

        Args:
            totals: The transport's running `(starved_seconds, total_seconds)`.

        Returns:
            The starvation ratio for the interval since the previous sample, or `None` when
            too little transfer happened in it to divide — a round served entirely from
            locality, or one that moved a single tiny bucket. `None` is "no opinion", which
            leaves the hysteresis band holding the last real verdict.
        """
        starved, total = totals
        # A reset (`reset_peer_transfers`) rewinds the totals under us, and subtracting a
        # larger baseline would yield a negative interval that clamps to a fully saturated
        # round — telling the controller to stop growing on the strength of a bookkeeping
        # event. Re-baseline instead and report nothing for this round.
        if total < self._seen_total or starved < self._seen_starved:
            self._seen_starved, self._seen_total = starved, total
            return None
        d_starved = starved - self._seen_starved
        d_total = total - self._seen_total
        self._seen_starved, self._seen_total = starved, total
        if d_total < _MIN_MEASURED_SECONDS:
            return None
        return min(1.0, max(0.0, d_starved / d_total))


class ChannelCongestion:
    """Fuses node memory pressure and measured occupancy into one `CongestionSignal`.

    Stateful, because the occupancy band is a Schmitt trigger: inside it the previous verdict
    stands. Hold one per credit controller. The *state* is per-controller; the occupancy it
    is fed is a node-level reading (see the module docstring), so two controllers in a
    process agree about the wire and can still differ about where they sit in the band.

    An unmeasured channel reports `STARVED`, which is deliberately the *permissive* verdict.
    A window that has never been tested has not earned the right to stop growing, and it is
    also what a build with no starvation counter — or a worker whose first round has not
    completed — must fall back to if the credit window is to behave exactly as it did before
    the measurement existed.
    """

    __slots__ = ("_high", "_last", "_low")

    def __init__(self, config: Config | None = None) -> None:
        fc = (config or active_config()).flow_control
        # Validated as `0 <= low <= high <= 1` by the config gate, so no re-clamp here: a
        # second opinion on a validated invariant is how the two drift apart.
        self._high = fc.backpressure_high
        self._low = fc.backpressure_low
        # The permissive start, for the reason in the class docstring.
        self._last = CongestionSignal.STARVED

    @property
    def signal(self) -> CongestionSignal:
        """The verdict as of the last `observe`, without taking a new sample."""
        return self._last

    def observe(self, *, pressured: bool, occupancy: float | None) -> CongestionSignal:
        """Fold one round's evidence into a verdict.

        Memory outranks occupancy unconditionally. A node past its spill threshold must
        shrink its transit buffers even on a channel that is starving for data, because the
        alternative to a slow shuffle is an OOM-killed worker and only one of those is
        recoverable.

        Args:
            pressured: Whether the node is at or past its spill pressure level.
            occupancy: Share of the last round this worker had data to work on, summed over
                every peer it fetched from, or `None` when nothing has been measured.

        Returns:
            This round's verdict, which is also retained as the hysteresis state.
        """
        if pressured:
            self._last = CongestionSignal.CONGESTED
        elif occupancy is None:
            self._last = CongestionSignal.STARVED
        elif occupancy >= self._high:
            self._last = CongestionSignal.SATURATED
        elif occupancy <= self._low:
            self._last = CongestionSignal.STARVED
        # Between the thresholds the previous verdict stands: that band *is* the hysteresis,
        # and re-deciding inside it is what makes a window flap.
        return self._last


def probe_pressure(monitor: PressureMonitor | None) -> bool:
    """Whether the node is at or past the level at which a query must spill.

    Reads `level()` rather than `classify()` on purpose: this is the credit round's own
    sample, and it is the one place that is entitled to advance the monitor's de-escalation
    average. Every other Carbonite reader takes the pure `classify()` view precisely so it
    does not consume this sample.

    Args:
        monitor: The pressure monitor, or `None` on a channel with no memory governor.

    Returns:
        `False` without a monitor — the historical non-adaptive behaviour, in which a channel
        with nothing watching memory never claimed the node was in trouble.
    """
    if monitor is None:
        return False
    from batcher.carbonite.memory.pressure import PressureLevel

    return monitor.level() >= PressureLevel.SPILL
