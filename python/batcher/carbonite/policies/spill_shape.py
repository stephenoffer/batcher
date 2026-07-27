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

    Args:
        basis: Bytes to shard, from `spill_basis`.
        envelope_bytes: The per-operator byte envelope admission offered as a counter-offer.

    Returns:
        The minimum bucket count, or 0 when either input is unusable.
    """
    if basis <= 0 or envelope_bytes <= 0:
        return 0
    return min(MAX_SPILL_PARTITIONS, max(1, ceil_div(basis, envelope_bytes)))


def should_compress(peak: int) -> bool | None:
    """Whether a spilled state of `peak` bytes is large enough to pay for compression.

    Args:
        peak: The learned-blended peak working-set bytes for the plan.

    Returns:
        The decision, or `None` for an un-sized plan so the caller keeps the configured
        default.
    """
    if peak <= 0:
        return None
    return peak >= SPILL_COMPRESS_ABOVE
