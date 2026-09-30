"""How a GPU fan-out's shards are sized and dealt against what the devices measured.

Ray places tasks against device *counts*, so a fleet with an H100 next to an L4 running an
equal number of shards on each finishes at the L4's rate and reports the H100 as idle. This
module turns measured per-device throughput into the two numbers that correct for it:
`adaptive_shard_factor` divides a fan-out more finely when the fleet is uneven, and
`device_shard_counts` apportions shards in proportion to each device's throughput.

The bundle layout a gang reserves is decided where the gang is reserved
(`dist.executors.ray_runtime.scheduling`).

Everything degrades to the configured behavior on an unmeasured fleet: the configured shard
factor, and an even deal.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "adaptive_shard_factor",
    "device_shard_counts",
    "fleet_spread",
]

#: How far apart the fastest and slowest device in a fleet must be before the fan-out is
#: divided more finely than the configured factor. Below it the fleet is effectively uniform
#: and extra shards buy nothing but per-task overhead.
_SPREAD_TRIGGER = 1.5

#: The most the measured spread may multiply the configured shard factor. A fleet with one
#: device an order of magnitude slower than the rest is a fleet with a sick device, and
#: answering that with fifty times the shards turns one slow device into a scheduler problem.
_MAX_SPREAD_FACTOR = 4


def device_shard_counts(n_shards: int, throughputs: Sequence[float]) -> tuple[int, ...]:
    """How many shards each device should take, in proportion to what it measured.

    Round-robin is the right answer for a uniform fleet and the wrong one for every other kind.
    A node with one device twice as fast as its neighbour finishes its half early and waits,
    so the stage runs at the slow device's rate with half the fleet idle — and the fix is not
    a faster device, it is more shards on the one already there.

    Largest-remainder apportionment, so the counts sum to `n_shards` exactly and no device is
    left with zero while another holds two more than its share.

    Args:
        n_shards: Shards to deal.
        throughputs: Measured rows per second per device, positionally by ordinal. A device
            with no measurement (`0.0`) is treated as *average*, not as idle: an unmeasured
            device is one nothing has run on yet, and giving it nothing guarantees it stays
            that way.

    Returns:
        A count per device, summing to `n_shards`. Empty when there are no devices; an
        all-zero-throughput fleet is dealt evenly, which is the round-robin it had.
    """
    n_devices = len(throughputs)
    if n_devices == 0 or n_shards <= 0:
        return () if n_devices == 0 else (0,) * n_devices
    known = [t for t in throughputs if t > 0.0]
    mean = sum(known) / len(known) if known else 1.0
    weights = [t if t > 0.0 else mean for t in throughputs]
    total = sum(weights)
    exact = [n_shards * w / total for w in weights]
    counts = [int(x) for x in exact]
    # Largest remainder: hand the shards integer division left over to the devices that lost
    # the most to truncation, so the total is exact and the bias does not accumulate on one end.
    remaining = n_shards - sum(counts)
    order = sorted(range(n_devices), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in order[:remaining]:
        counts[i] += 1
    return tuple(counts)


def fleet_spread(throughputs: Sequence[float]) -> float:
    """How far apart the fastest and slowest measured device in a fleet are.

    Args:
        throughputs: Measured rows per second per device. Unmeasured devices (`0.0`) are
            ignored rather than counted as infinitely slow.

    Returns:
        The ratio, `1.0` for a uniform fleet and for one with fewer than two measurements —
        which is "no opinion", and every consumer here reads it as "leave the default alone".
    """
    known = [t for t in throughputs if t > 0.0]
    if len(known) < 2:
        return 1.0
    slowest = min(known)
    return max(known) / slowest if slowest > 0 else 1.0


def adaptive_shard_factor(configured: int, throughputs: Sequence[float]) -> int:
    """How many shards per device a fan-out should divide into, given what the fleet measured.

    The configured factor is right for a uniform fleet: enough shards to bound each one's
    device memory and to make a preempted shard cheap to redo. It is wrong for a fleet whose
    devices differ, and wrong in a way that is invisible. Ray runs at most one task per device
    at a time, so with an equal number of shards each the stage finishes when the *slowest*
    device finishes its last one, and the fast devices idle from then on. Dividing more finely
    lets a fast device take a fourth and a fifth shard while the slow one is still on its
    second, without anything having to predict which device gets which.

    The measured spread is the multiplier, capped, and only past the point where the fleet is
    genuinely uneven. A uniform fleet keeps exactly the configured factor, which is what every
    existing deployment already runs.

    Args:
        configured: The factor from `distributed.gpu_shard_oversubscribe`.
        throughputs: Measured rows per second per device, from the learned statistics.

    Returns:
        The factor to use, never below `configured` and never below `1`. An unmeasured fleet
        gets `configured` back unchanged.
    """
    base = max(1, configured)
    spread = fleet_spread(throughputs)
    if spread < _SPREAD_TRIGGER:
        return base
    return base * min(_MAX_SPREAD_FACTOR, max(1, int(spread)))
