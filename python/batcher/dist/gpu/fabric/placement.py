"""Which devices a multi-device stage gets, and how its shards are dealt across them.

Ray places tasks against device *counts*, so a stage that asks for four devices on an
eight-device node gets four devices — any four. On a node whose boards are not uniformly
connected that is a real decision made by accident: four devices spanning two NVLink islands
exchange over the bus at a fraction of the rate the same four would have on one island, and
nothing in the job's timings says which four it got.

This module makes the two decisions Ray leaves open:

* **Which devices.** `local_device_group` picks the tightest set the node has, ranking on the
  fabric first and the bus second (`fabric.p2p`), and bounds the request to what one coherent
  island holds when the stage exchanges (`kyber.gpu.exchange`).
* **Which shard goes where.** `shard_device_assignment` deals shards by measured *throughput*
  rather than round-robin, because a fleet with an H100 next to an L4 running equal shards
  finishes at the L4's rate and reports the H100 as idle.

The third decision — the bundle layout a gang reserves — is made where the gang is reserved
(`dist.executors.ray_runtime.scheduling`), against the same `plan_collective` this module's
group selection agrees with.

Everything degrades to the prior behavior on an unreadable topology: no group, no assignment
change, the caller's own bundles.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "adaptive_shard_factor",
    "device_shard_counts",
    "fleet_spread",
    "local_device_group",
    "placement_summary",
    "shard_device_assignment",
]

#: How far apart the fastest and slowest device in a fleet must be before the fan-out is
#: divided more finely than the configured factor. Below it the fleet is effectively uniform
#: and extra shards buy nothing but per-task overhead.
_SPREAD_TRIGGER = 1.5

#: The most the measured spread may multiply the configured shard factor. A fleet with one
#: device an order of magnitude slower than the rest is a fleet with a sick device, and
#: answering that with fifty times the shards turns one slow device into a scheduler problem.
_MAX_SPREAD_FACTOR = 4


def local_device_group(world_size: int, *, exchanges: bool = True) -> tuple[int, ...]:
    """The `world_size` local devices that exchange fastest with each other.

    Bounded first: a stage whose devices talk to each other is capped at the widest coherent
    island, because the device outside it makes every round of the exchange run at its rate.
    A stage of independent shards passes `exchanges=False` and is not bounded, since nothing
    crosses between its devices at all.

    Args:
        world_size: How many devices the stage wants.
        exchanges: Whether the stage's devices exchange with each other.

    Returns:
        Device ordinals ascending, empty when the topology is unreadable or the node has fewer
        devices than the (bounded) request. Empty means "no opinion" — the caller keeps
        whatever Ray assigned rather than treating it as a refusal.
    """
    from batcher._internal.hardware.fabric.p2p import tightest_peer_group
    from batcher.kyber.gpu.exchange import fabric_bounded_width, widest_fabric_island

    if world_size <= 0:
        return ()
    bounded = fabric_bounded_width(world_size, widest_fabric_island(), exchanges=exchanges)
    return tightest_peer_group(bounded)


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


def shard_device_assignment(n_shards: int, throughputs: Sequence[float]) -> tuple[int, ...]:
    """Which device each shard goes to, weighted by measured throughput.

    The per-shard form of `device_shard_counts`, interleaved rather than blocked: shards are
    dealt out in rotation so the fast device starts its second shard while the slow one is
    still on its first. Blocking them (`0,0,0,1,1`) is the same total work and a worse tail,
    because the last shard of the slow device starts only after all of its others finish.

    Args:
        n_shards: Shards to place.
        throughputs: Measured rows per second per device, positionally by ordinal.

    Returns:
        A device ordinal per shard. Empty when there are no devices.
    """
    counts = list(device_shard_counts(n_shards, throughputs))
    if not counts:
        return ()
    out: list[int] = []
    while len(out) < n_shards:
        placed = False
        for device, remaining in enumerate(counts):
            if remaining > 0:
                out.append(device)
                counts[device] -= 1
                placed = True
                if len(out) == n_shards:
                    break
        if not placed:  # pragma: no cover - counts sum to n_shards, so this cannot be reached
            break
    return tuple(out)


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


def placement_summary(world_size: int) -> dict:
    """What this node would give a stage of `world_size` devices, as one record.

    Args:
        world_size: Devices the stage wants.

    Returns:
        `group` (the chosen ordinals), `island` (the widest coherent group), `bounded`
        (whether the request was capped by the fabric), and `class` (the group's worst pair).
        Zeroed and empty on a node whose topology cannot be read.
    """
    from batcher._internal.hardware.fabric.p2p import peer_group_class
    from batcher.kyber.gpu.exchange import widest_fabric_island

    island = widest_fabric_island()
    group = local_device_group(world_size)
    return {
        "group": list(group),
        "island": island,
        "bounded": bool(island and world_size > island),
        "class": peer_group_class(group) if len(group) > 1 else "",
    }
