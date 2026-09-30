"""Sizing a credit window from the path it runs over, instead of probing for it.

TCP's AIMD and CUBIC exist because a sender knows almost nothing: not the receiver's capacity,
not the path's, not how much data it will send, not who else is on the wire. It must therefore
*probe* — grow until something breaks, back off, repeat — and every constant in the law
(`alpha = 1`, `beta = 0.5`, `C = 0.4`) is a fairness heuristic tuned for anonymous flows on the
public internet.

A shuffle has none of those unknowns. The peers are the same query's workers, cooperative and
known; the objective is makespan rather than fairness among strangers; the bandwidth and delay
of each path are *measured* (`bc-transport`'s per-peer filters); and the receiver's memory
envelope is Carbonite's own number. Probing for a quantity you can measure is the mismatch this
module removes.

The measurement is BBR's, and so is the reason its estimators are shaped the way they are.
`RTprop` is a running **minimum** of observed round trips, because every error in a round-trip
sample is non-negative — queueing, a busy worker, a lost scheduler slice — so the truth is the
smallest thing ever seen. `BtlBw` is a running **maximum** of observed delivery rates, by the
mirror argument: a fetch can finish slower than the bottleneck allows but never faster. Their
product is the bandwidth-delay product, the bytes a path holds when it is exactly busy, which
is the quantity a credit window exists to match.

What that buys is convergence in *one* round trip rather than `log2(W)` of them. Slow start
doubling from 16 credits to 64 costs two round trips on a link where a round trip is 100 ms,
and a short bucket finishes before the ramp ever completes — so the transfer that most needs
the window runs its whole life below it.
"""

from __future__ import annotations

import math

from batcher.config import Config, active_config

__all__ = [
    "REFILL_WINDOW_GAIN",
    "bdp_window",
    "measured_bdp_window",
]

#: Multiplier on the bandwidth-delay product that a credit window must carry.
#:
#: **Forced by the refill batching, not chosen.** The consumer does not return a credit per
#: batch — it accumulates and sends one grant once half the window has drained
#: (`credit_exchange_inner`'s `refill_at = credits / 2`), which cuts control-message traffic by
#: that factor. The arithmetic of that choice sets the window:
#:
#: In steady state at delivery rate `L` batches/second over a path of round-trip time `R`, the
#: producer holds `L x R` batches in flight — the bandwidth-delay product in batches. On top of
#: that sit the batches the consumer has already taken but not yet acknowledged, which the
#: batching allows to reach `w / 2` before a grant is sent. The producer stalls unless its
#: window covers both:
#:
#:     w  >=  L x R  +  w / 2   =>   w / 2  >=  BDP_batches   =>   w  >=  2 x BDP_batches
#:
#: So a window set to exactly the BDP stalls the producer for roughly half of every round trip,
#: and the correct target is twice it. TCP BBR reaches the same factor of two from the
#: analogous cause (delayed and aggregated acknowledgements), which is a reassuring place to
#: land independently.
REFILL_WINDOW_GAIN = 2


def bdp_window(bdp_bytes: int, batch_bytes: int) -> int:
    """Credits needed to keep a path of this bandwidth-delay product busy.

    `ceil(BDP / batch_bytes) x REFILL_WINDOW_GAIN`, floored at 1. Ceiling rather than rounding
    because a window one batch short of the product idles the link every round trip, while one
    batch over costs a single buffered batch.

    Args:
        bdp_bytes: The path's bandwidth-delay product, `BtlBw x RTprop`.
        batch_bytes: Bytes one credit admits — a morsel.

    Returns:
        The target window in credits, at least 1. `1` for a non-positive product, which is the
        floor rather than an estimate: a caller with no measurement should pass `None` upstream
        rather than a zero.

    Examples:
        .. doctest::

            >>> from batcher.carbonite.policies import bdp_window
            >>> bdp_window(8 << 20, 1 << 20)  # an 8 MiB pipe, 1 MiB morsels
            16
    """
    if bdp_bytes <= 0 or batch_bytes <= 0:
        return 1
    return max(1, math.ceil(bdp_bytes / batch_bytes) * REFILL_WINDOW_GAIN)


def measured_bdp_window(config: Config | None = None) -> int | None:
    """The target window this process's own transfers imply, or `None` if unmeasured.

    Reads the transport's `BtlBw x RTprop` estimate and converts it to credits. `None` on a
    process that has not yet completed a fetch, and on a build whose engine predates the
    filters — both of which a caller must read as "no estimate", leaving the configured start
    and the probing ramp in charge exactly as before.

    Args:
        config: The active config, for the morsel size a credit admits.

    Returns:
        The window in credits, or `None`.
    """
    from batcher.carbonite.transfer.peers import bdp_bytes

    measured = bdp_bytes()
    if measured is None or measured <= 0:
        return None
    return bdp_window(measured, (config or active_config()).execution.morsel_bytes)
