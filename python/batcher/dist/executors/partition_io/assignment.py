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

from collections.abc import Callable, Sequence
from typing import TypeVar

from batcher._internal.mathx import ceil_div

__all__ = ["assign_splits", "balance_with_affinity", "has_affinity"]

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
    """
    groups: list[list[S]] = [[] for _ in range(workers)]
    loads = [0] * workers
    ordered = sorted(splits, key=lambda s: s.row_count() or 0, reverse=True)
    for s in ordered:
        i = min(range(workers), key=lambda w: loads[w])
        groups[i].append(s)
        loads[i] += s.row_count() or 1
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
    target = max(1, ceil_div(sum(s.row_count() or 1 for s in splits), workers))  # ceil per group
    w, load = 0, 0
    for s in splits:
        groups[w].append(s)
        load += s.row_count() or 1
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


def _weight(split: object) -> int:
    """A split's load weight — its row count, or 1 when unknown (as `_balance` weights it)."""
    rows = getattr(split, "rows", None)
    if rows is None:
        count = getattr(split, "row_count", None)
        rows = count() if callable(count) else None
    return rows or 1


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
    # Resident splits first, so a homeless split fills whichever worker locality left
    # lightest rather than displacing a split that had a home.
    resident = [(s, home[a]) for s in splits if (a := _split_affinity(s)) in home]
    homeless = [s for s in splits if _split_affinity(s) not in home]
    for s, i in resident:
        groups[i].append(s)
        loads[i] += _weight(s)
    for s in sorted(homeless, key=_weight, reverse=True):
        i = min(range(workers), key=lambda w: loads[w])
        groups[i].append(s)
        loads[i] += _weight(s)

    total = sum(loads)
    if total and max(loads) > _IMBALANCE_TOLERANCE * (total / workers):
        return fallback(splits, workers)
    return groups
