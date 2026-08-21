"""Bucket mechanics every out-of-core breaker shares: write them, size them, re-split them.

`scratch` answers *where* a spilling query's bytes go. This module answers what every
breaker then does with them, which until now each one answered for itself. The aggregate,
the join, the sort, the partitioned window and the global window each opened their own
per-bucket writers, each restated the rule for how big a bucket is, each wrote their own
`try/finally` around the store, and three of them wrote their own grace recursion. The
algorithms were the same; the constants had already drifted (the join derived its own
re-partition salt by a second formula that happens to equal `split_salt`, which is the
kind of agreement nothing was checking).

Four things live here, and they are the four the breakers share:

- `spill_scratch` — the scratch lifecycle as a context manager. `TieredSpillStore` was
  *already* one, publishing its accounting before cleaning up; five callers wrote that
  sequence out by hand anyway, and one of them had the two lines in the other order.
- `BucketWriters` — one lazily-opened writer per non-empty bucket. A bucket that receives
  no rows never opens a file, which is what keeps the open-file fan-out at the number of
  buckets that actually got data rather than at `n_buckets`.
- `resident_bytes` / `over_envelope` — the size a bucket costs to read back, and whether
  that exceeds the configured envelope. Uncompressed, always: see `resident_bytes`.
- `regrace` — the salted re-partition of one over-large bucket into sub-buckets, streamed
  so the thing that did not fit is never held whole.

What is deliberately *not* here is what to do with a bucket once it is read, because that
is the operator: an aggregate combines it, a join pairs it with its opposite number, a
window ranks it. The recursion shape is shared; the fold is not.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

from batcher.config import active_config
from batcher.dist.spill.scratch import _make_store, _work_dir

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.carbonite.spill import SpillHandle, TieredSpillStore

__all__ = [
    "GRACE_DEPTH",
    "GRACE_SUB_BUCKETS",
    "BucketWriters",
    "bucket_envelope",
    "over_envelope",
    "read_reserved_bucket",
    "regrace",
    "resident_bytes",
    "spill_scratch",
    "split_salt",
]


@contextmanager
def spill_scratch(prefix: str, spill_dir: str | None) -> Iterator[TieredSpillStore]:
    """The scratch directory and tiered store for one out-of-core breaker, cleaned up after.

    Wraps the store's own `__exit__` — which publishes the accounting *before* `cleanup`
    zeroes it — and adds the one thing the store cannot know: whether the directory holding
    it is ours to remove. `_work_dir` says so; an explicit `spill_dir` is the caller's and
    is left alone.

    Both halves matter and both have been got wrong here. `cleanup` runs unconditionally
    and before the `rmtree`, because it aborts any writer still open (a partition phase
    abandoned by an exception) and deletes *both* tiers — where the `rmtree` only ever
    reached the local one, so a failed query that had overflowed left orphaned objects in
    the remote bucket, accumulating and billable, with nothing recording they existed. And
    the accounting is read first, because an out-of-core query is invisible in every other
    counter (it returns the right answer, slowly), so its tier volumes are the only record
    that the query went to disk at all.

    Args:
        prefix: The scratch directory name prefix, naming the breaker.
        spill_dir: An explicit caller-owned directory, or `None` to allocate one.

    Yields:
        The store the breaker writes its buckets to.
    """
    work_dir, owns_dir = _work_dir(spill_dir, prefix)
    store = _make_store(work_dir)
    try:
        with store:
            yield store
    finally:
        if owns_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


class BucketWriters:
    """One lazily-opened spill writer per bucket, closed together into handles.

    Every partition phase in the engine — and every grace re-split inside one — ends in the
    same four lines: look up this bucket's writer, open it if this is the bucket's first
    row, write, and at the end close them all into handles. Opening lazily is not an
    optimization detail: the partition phase holds every open writer at once, so opening
    one per *bucket* rather than per *non-empty bucket* would put the file-descriptor cap
    (`scratch._FD_SAFE_PARTITIONS`) under pressure that the data never justified.
    """

    __slots__ = ("_store", "_tag", "_writers")

    def __init__(self, store: TieredSpillStore, tag: str) -> None:
        """Open no files yet; name each bucket's file `<tag>_<bucket>`.

        Args:
            store: The tiered store the buckets are written to.
            tag: The per-bucket file name prefix, unique within the store.
        """
        self._store = store
        self._tag = tag
        self._writers: dict[int, object] = {}

    def write(self, bucket: int, batch: pa.RecordBatch) -> None:
        """Append one batch to `bucket`, opening its writer on first use.

        Args:
            bucket: The bucket index.
            batch: The batch to append. An empty batch is dropped rather than written.

        Returns:
            None.
        """
        if not batch.num_rows:
            return
        writer = self._writers.get(bucket)
        if writer is None:
            writer = self._store.writer(f"{self._tag}_{bucket}")
            self._writers[bucket] = writer
        writer.write(batch)

    def add(self, partitions: Sequence[Sequence[pa.RecordBatch]]) -> None:
        """Append a whole partitioning — `partitions[b]` is bucket `b`'s batches.

        This is the shape both `partition_batches` and `partition_batches_salted` return,
        so a partition phase is one call rather than a nested loop with writer bookkeeping
        inside it.

        Args:
            partitions: Per-bucket batch lists, indexed by bucket.

        Returns:
            None.
        """
        for bucket, batches in enumerate(partitions):
            for batch in batches:
                self.write(bucket, batch)

    def close(self) -> dict[int, SpillHandle]:
        """Close every open writer, returning `{bucket: handle}` for the buckets that got rows.

        Returns:
            The handles, keyed by bucket. Buckets that received nothing are absent.
        """
        return {bucket: writer.close() for bucket, writer in self._writers.items()}

    def close_dense(self, n_buckets: int) -> list[SpillHandle | None]:
        """Close every open writer, returning an `n_buckets`-long list, `None` where empty.

        The positional form, for a caller that walks buckets in index order — a range
        partition emitting ranges in key order, or a co-partitioned join pairing bucket `b`
        of one side with bucket `b` of the other.

        Args:
            n_buckets: The list length, at least as large as any bucket written.

        Returns:
            One entry per bucket: its handle, or `None` if it received no rows.
        """
        handles: list[SpillHandle | None] = [None] * n_buckets
        for bucket, handle in self.close().items():
            handles[bucket] = handle
        return handles


def resident_bytes(handle: SpillHandle | None) -> int:
    """What reading `handle` back would cost in RAM — its **uncompressed** size.

    The one statement of this rule, because the wrong answer is available and plausible.
    `handle.nbytes` is the size on disk, and for a compressible bucket (repeated group
    keys, low-entropy payloads) that can be several times smaller than what decompressing
    it into memory costs. Budgeting a grace split against the on-disk size therefore lets
    an over-large bucket through the guard and into the fold that cannot hold it, which is
    an OOM at exactly the point spilling existed to prevent one.

    Args:
        handle: The bucket, or `None` for a bucket that was never written.

    Returns:
        The resident byte size, `0` for `None`.
    """
    if handle is None:
        return 0
    return handle.logical_nbytes or handle.nbytes


def bucket_envelope() -> int:
    """The per-bucket memory envelope — the configured ceiling, capped by the budget.

    `<= 0` when unbounded.

    This used to return `spill_bucket_max_bytes` alone, and that default is a fixed 128 MiB.
    Because a bucket is read back **whole**, the envelope is the one number that has to track
    the memory budget, and it did not: lowering `max_memory_bytes` to 1 MiB still authorized
    128 MiB buckets. Every consumer inherited it — the ordered sizer cut a *single* bucket for
    a 7 MiB input, and `over_envelope` (the aggregate, join and partitioned-window grace
    recursion) never fired at all, because no bucket under a small budget ever reached the
    128 MiB trigger. So the out-of-core paths were not bounded by the envelope they exist to
    respect, and the failure surfaces as the engine refusing a bucket rather than as anything
    a result can show.

    `MemoryConfig.streaming_state_budget_bytes` resolves the same tension the same way, for
    the reason it gives: a cap has to scale with the configured envelope rather than be a
    fixed magic number. This is the `min` of the two, so an explicit `spill_bucket_max_bytes`
    still lowers the bucket and never raises it above what can be read back.

    Returns:
        The per-bucket byte ceiling, or `<= 0` when the user opted out of any bound.
    """
    config = active_config()
    configured = config.memory.spill_bucket_max_bytes
    budget = config.spill_budget_bytes()
    if budget <= 0:  # explicit `unbounded_memory` — the caller declined a bound
        return configured
    return min(configured, budget) if configured > 0 else budget


# How many times a bucket still over the envelope may be re-split, and how many ways.
#
# The depth bound is for the case no hash can fix: a bucket dominated by ONE key's rows
# re-hashes to one sub-bucket however it is salted, because every row of a group — or of a
# window partition, or of a join key — must land together by construction. Past the bound the
# bucket is processed as it stands, which is what every one of these paths did unconditionally
# before the recursion existed. That is a genuine ceiling, not a tuning choice: lifting it
# needs per-group spillable state in `bc-runtime`, since partitioning is the only tool the
# mergeable algebra offers here. (Routing the aggregate's floor case to
# `combine_finalize_spilling` was tried and reverted — it grace-partitions by the same group
# key, so it cannot split what this recursion already could not: peak RSS was identical over
# 86 buckets that reached the floor, 716 MB either way, plus a re-read and re-write of each.)
GRACE_DEPTH = 3
GRACE_SUB_BUCKETS = 8


def over_envelope(handle: SpillHandle | None, depth: int, *, max_depth: int = GRACE_DEPTH) -> bool:
    """Whether `handle` should be re-split before being read, at recursion `depth`.

    Args:
        handle: The bucket about to be processed. Measured, deliberately, *before* being
            read: the decision has to happen without first pulling in the thing that does
            not fit.
        depth: The current grace-recursion depth, `0` at the top.
        max_depth: The recursion floor; past it the bucket is processed as it stands.

    Returns:
        `True` when a re-split is both warranted and still permitted.
    """
    envelope = bucket_envelope()
    return envelope > 0 and resident_bytes(handle) > envelope and depth < max_depth


def split_salt(depth: int) -> int:
    """The re-partition salt for grace recursion `depth`.

    Non-zero, and distinct per level, and both halves are load-bearing. Zero is the
    unsalted, cluster-wide bucket assignment that a *shuffle* must agree on and a *local*
    re-split must not reuse: bucket assignment reads the low hash bits at a power-of-two
    count, so an unsalted re-partition of a 16-way bucket into 8 sub-buckets sends every row
    to `bucket & 7` — one sub-bucket, at every level, so the recursion rewrites and re-reads
    the whole bucket three times and changes nothing. Varying by depth means keys that
    collided at one level spread at the next instead of re-colliding identically. It never
    varies by row, so equal keys still co-locate and each sub-bucket stays an exact
    independent unit of work.

    Args:
        depth: The recursion depth, `0` at the top.

    Returns:
        An odd 64-bit salt.

    Examples:
        .. doctest::

            >>> from batcher.dist.spill.buckets import split_salt
            >>> split_salt(0) != split_salt(1) and split_salt(0) % 2 == 1
            True
    """
    return (0x9E3779B97F4A7C15 * (depth + 1)) % (1 << 64) | 1


def regrace(
    nat,
    store: TieredSpillStore,
    handle: SpillHandle,
    key_idx: Sequence[int],
    salt: int,
    tag: str,
    *,
    n_sub: int = GRACE_SUB_BUCKETS,
) -> dict[int, SpillHandle]:
    """Re-partition one over-large bucket into salted sub-buckets on disk.

    Streamed: one batch is read, sharded and appended before the next arrives, so the
    bucket that did not fit is never held whole — which is the whole point of splitting it.
    The parent is released as soon as it has been re-partitioned, because it is the largest
    file in the recursion and holding it would make each level cost *more* disk instead of
    the same.

    Args:
        nat: The native engine.
        store: The tiered store holding `handle` and receiving the sub-buckets.
        handle: The bucket to split.
        key_idx: Positions of the partition columns in the bucket's schema. The *same*
            columns the bucket was originally partitioned on — re-splitting on anything
            else would let one key's rows land in two sub-buckets.
        salt: The re-partition salt, from `split_salt`.
        tag: A file-name prefix unique to this split within the store.
        n_sub: How many sub-buckets to split into.

    Returns:
        `{sub_bucket: handle}` for the sub-buckets that received rows.
    """
    writers = BucketWriters(store, tag)
    for batch in store.read_stream(handle):
        writers.add(nat.partition_batches_salted([batch], list(key_idx), n_sub, salt))
    subs = writers.close()
    store.release(handle)
    return subs


def read_reserved_bucket(store: TieredSpillStore, handle: SpillHandle | None):
    """Read one bucket whole, with its resident footprint reserved against the buffer pool.

    Reading a bucket back is the one step of spilling that can undo it — the state went to
    disk because it did not fit — and a plain `read` tells the pool nothing, so a concurrent
    query sizing its own state sees headroom this breaker is already using.

    Args:
        store: The store holding the bucket.
        handle: The bucket, or `None`.

    Returns:
        The batches, or `None` when `handle` is `None`.
    """
    if handle is None:
        return None
    with store.read_reserved(handle) as stream:
        return list(stream)
