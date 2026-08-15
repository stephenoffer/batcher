"""How a source's splits are divided among the workers — the three assignment strategies.

`_balance` bin-packs by row count, which is right for a source read from object storage:
every worker is equidistant from S3, so only load matters. `_contiguous` gives that up to
keep the source's global row order, which the order-sensitive callers require.

Neither is right for a split that is *already resident on a specific worker* — the
`FlightFetchSplit`s of an intermediate a previous stage left partitioned on the fleet.
Bin-packing those sends worker `i` to fetch buckets held by arbitrary peers, so an
`N`-worker fleet reads about `1 - 1/N` of its own intermediate over the network, having
first serialized it, when the same bytes were sitting in the reading process's heap.
`balance_with_affinity` routes each such split to the worker that holds it, turning that
read into a zero-copy local-store hit. Placement never changes the result — only which
transfer mode the read resolves to — so it is always safe to apply.

The one way locality can *lose* is imbalance: an intermediate concentrated on one worker
would serialize the next stage onto that worker while the rest idle. The assignment
measures the balance it produced and falls back to `_balance` when it is worse than the
tolerance, so a skewed intermediate keeps its parallelism.
"""

from __future__ import annotations

import heapq
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from batcher._internal.mathx import ceil_div

__all__ = [
    "assign_clustered_splits",
    "assign_splits",
    "balance_with_affinity",
    "has_affinity",
    "split_weights",
]

# Split count past which a weight that is not already known is taken as 1 rather than
# asked for. `FileSplit.row_count()` rebuilds a single-file reader and reads that file's
# footer, so weighing a million whole-file splits is a million metadata round trips on the
# driver — the exact cost `FileSource.splits` stops planning sub-file splits to avoid
# (`_MAX_FOOTER_PLAN_FILES`), paid again one layer up while the whole cluster idles. Past
# this many splits there is ample parallelism for equal-count packing to balance well, and
# splits that already carry their count (row-group splits, shuffle buckets) keep using it
# at any scale. Same default and the same reasoning as the footer-planning cap.
_MAX_WEIGHED_SPLITS = max(1, int(os.environ.get("BATCHER_MAX_WEIGHED_SPLITS", "10000")))

# How much heavier the busiest worker may get, relative to an even share, before locality
# is judged to have cost more parallelism than it saved network. At 2.0 a worker may carry
# twice its share — the point where the stage's critical path is the locality win's
# rough break-even against a same-node read of the whole intermediate.
_IMBALANCE_TOLERANCE = 2.0


S = TypeVar("S")


def _balance(splits: list[S], workers: int) -> list[list[S]]:
    """Greedily bin-pack splits into `workers` groups balanced by row count.

    Splits with an unknown row count are weighted as 1 so they spread evenly.
    Largest-first assignment keeps the per-worker load roughly equal.

    Weights are computed once (`split_weights`) rather than per comparison: this used to ask
    each split for its row count twice, and for a whole-file split that question is a
    footer read, so assigning N files cost 2N metadata round trips before a worker started.
    The least-loaded worker comes off a heap for the same reason — a linear scan per split
    is O(splits x workers), which is the driver's whole prologue on a wide fleet. Measured
    with an already-known weight (so the footer reads are not even in it): 50,000 splits
    across 256 workers, 2.70 s -> 0.13 s; 200,000 across 1,024, 27.90 s -> 0.45 s. That is
    time the entire cluster spends idle.
    """
    groups: list[list[S]] = [[] for _ in range(workers)]
    if workers <= 0 or not splits:
        return groups
    weights = split_weights(splits)
    # (load, worker): ties break on the lower worker index, as the linear scan did.
    heap = [(0, w) for w in range(workers)]
    for i in sorted(range(len(splits)), key=lambda i: weights[i], reverse=True):
        load, w = heapq.heappop(heap)
        groups[w].append(splits[i])
        heapq.heappush(heap, (load + weights[i], w))
    return groups


def _contiguous(splits: list[S], workers: int) -> list[list[S]]:
    """Group splits into `workers` contiguous, source-ordered runs (order preserved).

    Unlike `_balance` (which reorders splits largest-first for even load), group 0 holds the
    source's first splits, group 1 the next, and so on — each a contiguous near-equal-count
    run. Callers whose correctness needs the concatenation of per-partition results to
    reproduce the source's global row order (distributed `LIMIT` / `with_row_index`) require
    this: a `_balance` assignment puts non-adjacent splits in one partition, so a per-partition
    prefix interleaves rows from different parts of the source.
    """
    groups: list[list[S]] = [[] for _ in range(workers)]
    if workers <= 0 or not splits:
        return groups
    weights = split_weights(splits)
    target = max(1, ceil_div(sum(weights), workers))  # ceil per group
    w, load = 0, 0
    for s, weight in zip(splits, weights, strict=True):
        groups[w].append(s)
        load += weight
        if load >= target and w < workers - 1:
            w, load = w + 1, 0
    return groups


def assign_splits(
    splits: list[S],
    workers: int,
    *,
    preserve_order: bool = False,
    worker_addrs: Sequence[str] | None = None,
) -> list[list[S]]:
    """Divide `splits` among `workers`, picking the strategy the caller's needs allow.

    Order preservation outranks locality, which outranks pure load balance: `_contiguous`
    is a correctness requirement of the caller rather than a placement preference, so a
    locality reshuffle must never override it.

    Args:
        splits: The splits to assign.
        workers: How many groups to produce.
        preserve_order: Keep the source's global row order (contiguous runs per group).
        worker_addrs: Shuffle address per worker, enabling locality-aware assignment for
            splits that are already resident on one of them.

    Returns:
        One list of splits per worker, in worker order.
    """
    if preserve_order:
        return _contiguous(splits, workers)
    if worker_addrs and has_affinity(splits):
        return balance_with_affinity(splits, workers, worker_addrs, _balance)
    return _balance(splits, workers)


@dataclass(frozen=True, slots=True)
class _ClusterGroup:
    """Every split sharing one clustering value, bin-packed as a single indivisible unit.

    `rows` is the sum of the members' *already known* row counts, exposed under the name
    `split_weights` reads so a group balances by its real size. It is `None` when no member
    knew its own count, which is the same "unknown, weigh as 1" the assignment gives any
    split whose count would cost a metadata round trip -- and `row_count_needs_a_sweep`
    ensures nothing goes looking for it, because a group's sweep is its whole partition's.
    """

    row_count_needs_a_sweep = True

    splits: tuple[object, ...]
    rows: int | None


def assign_clustered_splits(
    splits: list[S],
    workers: int,
) -> list[list[S]]:
    """Divide `splits` among `workers` without ever separating two that share a value.

    The assignment half of the co-location guarantee. A file-per-split reader (Delta,
    Iceberg) emits many splits per partition value, so the split set alone never proves a
    value sits on one worker; grouping the value's splits and bin-packing whole groups does,
    at any split granularity, and the fine splits survive inside the group so the read
    parallelism *within* a worker is unchanged.

    Groups are packed by their summed row counts through the same balancer every other
    assignment uses, so a heavy partition is still spread against light ones as well as an
    indivisible unit can be.

    The number of groups is a hard ceiling on parallelism -- a value cannot be in two places
    -- which is why the caller checks it against the fleet width before choosing this path.

    Args:
        splits: The splits to assign. Every one must declare a clustering.
        workers: How many groups to produce.

    Returns:
        One list of splits per bucket, at most one bucket per clustering group -- fewer than
        `workers` when the layout has fewer partitions than the fleet has room for. A set that
        declares no common
        clustering falls back to plain load balancing -- there is no grouping to respect --
        which no real caller reaches, because `partition_descriptors` refuses that set
        outright rather than quietly assigning it under a plan that has no combine in it.
    """
    from batcher.io.splits import group_by_clustering

    grouped = group_by_clustering(splits)
    if grouped is None:
        return _balance(splits, workers)
    units = [_ClusterGroup(tuple(g), _summed_rows(g)) for g in grouped]
    # Never more buckets than there are groups. `_balance` hands back exactly as many buckets
    # as it is asked for, and an indivisible group cannot fill two, so asking for the caller's
    # full partition count on a table with fewer partitions returns the surplus as *empty*
    # buckets -- and every empty bucket still costs a task, a CPU reservation and a schema-only
    # round trip. A 64-file, 16-partition Delta table asked for 64 and would have run 48 no-op
    # tasks. The descriptor list length is the authority on how many partitions there are
    # (`partition_descriptors`), so returning fewer is the supported way to say so.
    return [
        [s for unit in bucket for s in unit.splits]
        for bucket in _balance(units, min(workers, len(units)))
    ]


def _summed_rows(splits: Sequence[object]) -> int | None:
    """The splits' total row count, or None when not one of them already knew its own.

    Only *known* counts are summed, never asked for: `_weight` documents why a count that
    costs a footer read (or a subtree sweep) must not be requested during assignment.
    """
    known = [r for s in splits if (r := _known_weight(s)) is not None]
    return sum(known) if known else None


def has_affinity(splits: Sequence[object]) -> bool:
    """Whether any split declares an `affinity()` — i.e. is already resident somewhere.

    Args:
        splits: The splits about to be assigned to workers.

    Returns:
        `True` when locality-aware assignment can do better than pure bin-packing.
    """
    return any(callable(getattr(s, "affinity", None)) for s in splits)


def _split_affinity(split: object) -> str | None:
    """The address holding `split`, or `None` for a split that lives nowhere in particular."""
    fn = getattr(split, "affinity", None)
    if not callable(fn):
        return None
    return fn() or None


def _known_weight(split: object) -> int | None:
    """A split's row count if it already carries one, without asking it to find out.

    A row-group split, a shuffle bucket, and an IPC intermediate all record their exact
    count when they are built. A whole-file split does not, and asking it means opening
    that file's footer — which is the difference between a free weight and a network round
    trip, and the reason this is a separate question from `_weight`.
    """
    rows = getattr(split, "rows", None)
    return rows if isinstance(rows, int) else None


def _weight(split: object) -> int:
    """A split's load weight — its row count, or 1 when unknown (as `_balance` weights it).

    A split whose count costs a **sweep of its whole subtree** is weighed as unknown rather
    than asked. The cap above bounds how many splits may each pay one metadata round trip;
    it cannot bound a split whose single question costs a round trip per file *behind* it.
    `PartitionDirSplit` is that shape — it stands for a whole Hive partition directory, and
    `row_count()` re-lists that subtree and opens every footer in it. Weighing forty of them
    over a thousand-file table measured 130 ms of driver time locally, and it scales with the
    **table**, not the split count: a date-partitioned petabyte would sweep every file in it
    on the driver, during assignment, which is the entire cost the distributed-listing reader
    exists to remove.

    There is nothing cheaper to weigh them by — the driver has only the directory names — so
    they pack by equal count, exactly as `split_weights` does for every split above its cap.
    """
    rows = _known_weight(split)
    if rows is None and not getattr(split, "row_count_needs_a_sweep", False):
        count = getattr(split, "row_count", None)
        rows = count() if callable(count) else None
    return rows or 1


def split_weights(splits: Sequence[object]) -> list[int]:
    """Every split's load weight, each computed at most once.

    Shared with `descriptor_rows`, which sizes a task's CPU share by its data and used to
    ask the same expensive question a second time.

    Above `_MAX_WEIGHED_SPLITS` a weight that is not already known is taken as 1 instead of
    being read off storage: at that scale the metadata reads dominate the driver and
    equal-count packing balances a large file set well enough. Splits that carry their own
    count are still weighed exactly, at any scale, because that costs nothing.
    """
    known = [_known_weight(s) for s in splits]
    if len(splits) > _MAX_WEIGHED_SPLITS:
        return [w or 1 for w in known]
    return [(w if w is not None else _weight(s)) or 1 for s, w in zip(splits, known, strict=True)]


def balance_with_affinity(
    splits: Sequence[S],
    workers: int,
    worker_addrs: Sequence[str],
    fallback: Callable[[Sequence[S], int], list[list[S]]],
) -> list[list[S]]:
    """Group `splits` into `workers` groups, keeping each on the worker that holds it.

    A split whose `affinity()` matches `worker_addrs[i]` goes to group `i`, so worker `i`
    reads it out of its own process. Splits with no affinity, or one naming an address
    outside this fleet (a worker that was replaced since the intermediate was published),
    fall to the least-loaded group — the same greedy rule `_balance` uses.

    Falls back to `fallback` (the plain load-balancer) when locality would leave the
    busiest worker carrying more than `_IMBALANCE_TOLERANCE` times an even share, so a
    concentrated intermediate keeps its parallelism instead of serializing onto one node.

    Args:
        splits: The splits to assign.
        workers: How many groups to produce.
        worker_addrs: Shuffle address per worker, indexed by worker.
        fallback: The load-only assignment to use when locality would unbalance the stage.

    Returns:
        One list of splits per worker, in worker order.
    """
    if workers <= 0:
        return []
    if not splits or len(worker_addrs) < workers:
        return fallback(splits, workers)

    home = {addr: i for i, addr in enumerate(worker_addrs[:workers]) if addr}
    groups: list[list[S]] = [[] for _ in range(workers)]
    loads = [0] * workers
    # Weight and locate each split exactly once. Both questions can cost a round trip
    # (`_weight` a footer read, `affinity()` a lookup), and the comprehensions below used
    # to ask each of them twice per split on the driver's critical path.
    weights = split_weights(splits)
    placed = [home.get(_split_affinity(s) or "") for s in splits]
    # Resident splits first, so a homeless split fills whichever worker locality left
    # lightest rather than displacing a split that had a home.
    for s, i, weight in zip(splits, placed, weights, strict=True):
        if i is not None:
            groups[i].append(s)
            loads[i] += weight
    homeless = [(i, weights[i]) for i, home_of in enumerate(placed) if home_of is None]
    heap = [(loads[w], w) for w in range(workers)]
    heapq.heapify(heap)
    for i, weight in sorted(homeless, key=lambda pair: pair[1], reverse=True):
        load, w = heapq.heappop(heap)
        groups[w].append(splits[i])
        loads[w] += weight
        heapq.heappush(heap, (load + weight, w))

    total = sum(loads)
    if total and max(loads) > _IMBALANCE_TOLERANCE * (total / workers):
        return fallback(splits, workers)
    return groups
