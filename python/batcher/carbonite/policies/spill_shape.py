"""How wide and how compressed a spilled state should be.

These are the two levers Carbonite pulls once it has already decided a query must go
out-of-core: how many buckets to shard the state into, and whether those buckets pay for
compression. Both are **result-invariant** — the mergeable algebra returns an identical
merged result for any partition count, and the spill codec is lossless — so nothing here
can change an answer. They are memory-safety and throughput levers only.

They live outside `ResourceManager` because they are pure arithmetic over a byte count,
with no reference to the manager's state, and because two of them shared a "which byte
count do I shard by?" rule that was written out twice: the *measured* spill volume when a
family has a spill history, else the learned-blended peak. That is one decision, and it is
`spill_basis` here rather than a comment saying "same basis as ..." beside a copy of it.
"""

from __future__ import annotations

from batcher._internal.mathx import ceil_div

__all__ = [
    "envelope_shortfall",
    "partitions_for_envelope",
    "partitions_for_volume",
    "should_compress",
    "spill_basis",
]

# Learned spill-partition sizing: aim each out-of-core bucket at roughly this many bytes
# of the LEARNED peak, so a bigger measured working set shards into more, smaller buckets
# (bounded memory per bucket) and a small one stays coarse. Only *shards* — the shuffle
# is result-invariant in the number of partitions.
SPILL_BYTES_PER_PARTITION = 128 * 1024 * 1024  # 128 MiB target per spill bucket
MIN_SPILL_PARTITIONS = 2
MAX_SPILL_PARTITIONS = 4096
# Above this learned peak a spill bucket compresses well enough to be worth the CPU:
# a large out-of-core state is IO-bound, so trading CPU for less disk/network wins.
SPILL_COMPRESS_ABOVE = 512 * 1024 * 1024  # 512 MiB


def spill_basis(peak: int, volume: int) -> int:
    """The byte count to shard a spilled state by.

    Prefers the *measured* spill volume when a family has a spill history: buckets shard
    only the bytes that actually reach disk, which is smaller than the total working-set
    peak (that peak includes the in-memory budget which never spills). Falls back to the
    peak when nothing has spilled yet.

    Args:
        peak: The learned-blended peak working-set bytes for the plan.
        volume: Predicted bytes this plan's family actually spills, or 0 if unknown.

    Returns:
        The basis in bytes, or 0 when the plan is un-sized and nothing can be concluded.
    """
    return volume if volume > 0 else max(0, peak)


def partitions_for_volume(basis: int, target_bytes: int = SPILL_BYTES_PER_PARTITION) -> int | None:
    """Bucket count that puts roughly `target_bytes` in each bucket.

    Args:
        basis: Bytes to shard, from `spill_basis`.
        target_bytes: Bytes to aim at per bucket. Defaults to `SPILL_BYTES_PER_PARTITION`.
            A caller that knows the real per-bucket ceiling — `memory.spill_bucket_max_bytes`
            is the size above which the reduce re-partitions a bucket by grace recursion —
            should pass it, so the *first* partitioning already lands under the ceiling
            instead of producing buckets the reduce then has to split again. Sharding twice
            for a figure that was known up front is pure re-read of the spilled state.

    Returns:
        The bucket count, or `None` when `basis` is 0 so the caller keeps its default.
    """
    if basis <= 0:
        return None
    parts = max(MIN_SPILL_PARTITIONS, ceil_div(basis, max(1, target_bytes)))
    return min(MAX_SPILL_PARTITIONS, int(parts))


def partitions_for_envelope(basis: int, envelope_bytes: int) -> int:
    """Fewest buckets that make each bucket fit `envelope_bytes`.

    **The fit is not guaranteed at the top of the range**, and the name promises more than
    the clamp can deliver. Past `MAX_SPILL_PARTITIONS x envelope_bytes` of state the count
    saturates, so each bucket is larger than the envelope by however far the basis exceeds
    that product — at PB scale against a gigabyte envelope, by three orders of magnitude.
    Raising the cap is not the answer: 4,096 buckets is already 4,096 files and 4,096 tasks,
    and the count is a scheduling cost as much as a memory one.

    What makes the shortfall safe rather than an OOM is downstream: the reduce re-partitions
    an over-large bucket by grace recursion (`dist/spill.py`), so an oversized bucket costs
    an extra split and a re-read rather than a failure. This function is a *first* shard,
    not the last word — which is worth stating, because "fewest buckets that make each
    bucket fit" reads like a guarantee that the caller can rely on, and it is not one.

    Args:
        basis: Bytes to shard, from `spill_basis`.
        envelope_bytes: The per-operator byte envelope admission offered as a counter-offer.

    Returns:
        The minimum bucket count, capped at `MAX_SPILL_PARTITIONS`; `0` when either input
        is unusable.
    """
    if basis <= 0 or envelope_bytes <= 0:
        return 0
    return min(MAX_SPILL_PARTITIONS, max(1, ceil_div(basis, envelope_bytes)))


def envelope_shortfall(basis: int, envelope_bytes: int) -> int:
    """Bytes by which `partitions_for_envelope` misses its own target, or `0` if it does not.

    The clamp above is silent, and a caller that reasons "each bucket now fits" is wrong at
    scale with nothing to tell it so. This is the number that says by how much, so a caller
    that cares can log it, widen the envelope, or size a downstream grace recursion for the
    split it is going to need anyway.

    Args:
        basis: Bytes to shard, from `spill_basis`.
        envelope_bytes: The target per-bucket envelope.

    Returns:
        `per_bucket - envelope_bytes` when the clamp binds, else `0`.

    Examples:
        .. doctest::

            >>> from batcher.carbonite.policies.spill_shape import envelope_shortfall
            >>> envelope_shortfall(1 << 30, 1 << 20)  # 1 GiB into 1 MiB buckets: fits
            0
            >>> envelope_shortfall(1 << 40, 1 << 20) > 0  # 1 TiB: past the 4,096 cap
            True
    """
    parts = partitions_for_envelope(basis, envelope_bytes)
    if parts <= 0:
        return 0
    per_bucket = ceil_div(basis, parts)
    return max(0, per_bucket - envelope_bytes)


#: Device cost factor at or above which the disk, not the CPU, is the binding constraint.
#:
#: Compression trades CPU for bytes, so whether it pays is a question about the *device*, not
#: only about the size of the state. On local flash at several gigabytes a second the codec is
#: the bottleneck and a small state should go straight down. On a network volume at a tenth of
#: that — the class this factor names — every byte not written is time not spent, and the
#: trade pays at any size worth spilling. Two is the smallest factor that is unambiguously not
#: local flash (`loopback`), which keeps this from firing on a device merely measured
#: imprecisely.
_COMPRESS_DEVICE_FACTOR = 2.0

#: Below this a state is too small for the device argument to matter either: the whole spill
#: is a handful of buffers and the codec's own setup dominates. Deliberately far under
#: `SPILL_COMPRESS_ABOVE`, since the point of the device term is to compress states that rule
#: would have written raw.
_COMPRESS_DEVICE_FLOOR = 64 << 20


def should_compress(peak: int, device_factor: float = 1.0) -> bool | None:
    """Whether a spilled state of `peak` bytes is worth compressing on this device.

    Args:
        peak: The learned-blended peak working-set bytes for the plan.
        device_factor: What a byte costs on the spill device relative to local flash, from
            `hardware.storage.device_cost_factor`. `1.0` — local flash, and every device that
            could not be identified — keeps the size-only rule this had before.

    Returns:
        The decision, or `None` for an un-sized plan so the caller keeps the configured
        default.
    """
    if peak <= 0:
        return None
    if device_factor >= _COMPRESS_DEVICE_FACTOR and peak >= _COMPRESS_DEVICE_FLOOR:
        # The disk is the bottleneck, so the CPU the codec costs is CPU that would otherwise
        # be waiting on it.
        return True
    return peak >= SPILL_COMPRESS_ABOVE
