"""The shape of a mergeable reduction: how `n` partials collapse to one without any node
reading more than `fan_in` of them.

A shuffle's reduce side is an associative fold, so it is free to choose *any* bracketing of
its inputs. The two extremes are not equivalent at scale. Folding them in a line —
``combine(combine(combine(p0, p1), p2), ...)`` — is what a reducer does when it walks its
input list, and its critical path is `n - 1` combines on **one** node. Bracketing them as a
balanced tree of arity `f` costs the same total combines but spreads them over
``ceil(n / f)`` independent tasks per level and ``ceil(log_f n)`` levels, so the critical
path is `f * ceil(log_f n)`.

That difference is the whole of this module, and it is the difference between a shuffle that
scales and one that does not. With `W` mappers and `W` reducers, the linear fold makes each
reducer do Θ(W) work, so the reduce phase *grows* as nodes are added while the map phase
shrinks — the serial term in Amdahl's law, arriving exactly when the cluster gets big enough
to need it. The tree makes it Θ(log W), which is the term that lets total time keep falling
as `W` rises.

Nothing here schedules anything or knows what a partial is. It is the arithmetic — how many
levels, which inputs form which chunk — so the disk shuffle, the Flight shuffle and the
top-N merge can share one answer and one set of tests rather than re-deriving it three
times. The callers supply the combine.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeVar

__all__ = ["chunks", "fold_width", "reduce_levels", "tree_reduce"]

#: The widest a fold is allowed to get, however many partials there are.
#:
#: `fold_width` derives its width from the leaf count, and left unbounded that grows without
#: limit -- 40,000 leaves would fold 200 at a time, which is precisely the per-node inbound
#: fan-in `flow_control.shuffle_fan_in` exists to keep bounded as a cluster reaches thousands
#: of nodes. 32 is the ceiling because `flow_control.shuffle_fetch_fan_in` already puts a flat
#: gather's concurrent peers there, so it is a width this engine is on record as willing to
#: open at once. It is inert everywhere the widening was measured (256 leaves fold at 16), and
#: only a fleet past ~1,000 map partitions reaches it, where the tree keeps a third level
#: instead of widening further.
_MAX_FOLD_WIDTH = 32

T = TypeVar("T")


def fold_width(fan_in: int, sources: int) -> int:
    """The arity the tree folds at — which is not the arity that *selects* a tree.

    `flow_control.shuffle_fan_in` answers two questions that want different answers, and
    conflating them is a cliff. It is the **trigger** (`workers > fan_in` picks the tree over
    the flat reduce) and it was also the **fold width** (how many partials one combiner
    reads). Raising the constant to widen the fold silently un-selects the tree on every fleet
    at or below the new value: measured at TPC-H sf100, raising it from 8 to 32 took a
    150M-group `GROUP BY` from 1,486 ms at 64 workers — a genuine 1.39x win — to **11,608 ms
    at 32 workers**, because 32 workers stopped being "more than the fan-in" and fell to the
    flat reduce that shape is worst on. So the trigger stays the configured value and only the
    width moves.

    The width matters because the tree's cost is not only combines. A level is
    `n_reducers x ceil(sources / width)` actor calls submitted from a single-threaded driver,
    and the leaves are map *partitions* (`workers x map_partition_multiplier`), so at 64
    workers there are 256 of them and a width of 8 means three levels and ~2,368 calls — with
    the cluster measured at 20% busy while they went out. Measured at 64 workers, best of
    three:

    | width | `distinct` | `group_by`, 150M groups |
    |---|---|---|
    | 8 | 2,427 ms | 1,971 ms |
    | 16 | **1,464 ms** | 1,529 ms |
    | 32 | 1,545 ms | **1,424 ms** |

    Both are ~1.5x better than 8, and the two agree closely enough that the exact value
    matters less than getting off three levels. `ceil(sqrt(sources))` is the width that makes
    the tree **two levels** whatever the fan-out is, which is what keeps this from being a
    constant that has to be re-tuned per cluster size: 256 leaves fold at 16, 128 at 12, 64 at
    8. It is floored at `fan_in` so a configured *wider* fold is never narrowed, capped at
    `_MAX_FOLD_WIDTH` so a very large fleet keeps a third level rather than opening an
    unbounded fan-in, and a no-op wherever the tree already has one level.

    Args:
        fan_in: The configured fan-in, which is the floor and the trigger.
        sources: How many partials the tree's leaf level holds.

    Returns:
        The fold arity, never below `fan_in`.

    Examples:
        .. doctest::

            >>> from batcher.dist.reduction import fold_width
            >>> fold_width(8, 256)   # 64 workers x 4 map partitions: two levels, not three
            16
            >>> fold_width(8, 64)    # already two levels at the configured width
            8
    """
    fan_in = max(1, fan_in)
    if sources <= fan_in:
        return fan_in
    return max(fan_in, min(math.isqrt(sources - 1) + 1, _MAX_FOLD_WIDTH))


def chunks(items: Sequence[T], size: int) -> list[Sequence[T]]:
    """Split `items` into consecutive runs of at most `size`.

    Consecutive rather than strided on purpose: a shuffle's mapper ids are assigned in
    worker order, so adjacent partials tend to share a node, and a chunk that is contiguous
    in that order is the one most likely to combine without leaving the machine.

    Args:
        items: The sequence to divide.
        size: The maximum run length, at least 1.

    Returns:
        The runs, in order. Empty when `items` is empty.

    Examples:
        .. doctest::

            >>> from batcher.dist.reduction import chunks
            >>> [list(c) for c in chunks([0, 1, 2, 3, 4], 2)]
            [[0, 1], [2, 3], [4]]
    """
    step = max(1, size)
    return [items[i : i + step] for i in range(0, len(items), step)]


def reduce_levels(n_sources: int, fan_in: int) -> int:
    """How many *interior* combine levels a tree reduce of `n_sources` partials needs.

    The last level is the caller's: once at most `fan_in` partials remain, one combine
    finishes the bucket. So this counts the levels *before* that one, which is what a
    scheduler needs to know to size its stage numbering and what a test needs to pin the
    Θ(log n) claim.

    The count is ``max(0, ceil(log_fan_in(n)) - 1)``, computed by repeated division rather
    than by a logarithm so it cannot disagree with the loop that actually runs — a
    floating-point `log` is off by one at exact powers of the base often enough to matter,
    and being off by one here means either an unreduced frontier or a wasted stage.

    Args:
        n_sources: The number of partials entering the reduction.
        fan_in: The maximum partials any one node may read, at least 2.

    Returns:
        The number of interior levels, `0` when the frontier already fits in one combine.

    Examples:
        .. doctest::

            >>> from batcher.dist.reduction import reduce_levels
            >>> reduce_levels(4, 4), reduce_levels(8, 4), reduce_levels(1000, 4)
            (0, 1, 4)
    """
    f = max(2, fan_in)
    levels = 0
    n = n_sources
    while n > f:
        n = -(-n // f)  # ceil(n / f)
        levels += 1
    return levels


def tree_reduce(
    sources: Sequence[T],
    combine_chunk,
    fan_in: int,
    *,
    on_level=None,
) -> list[T]:
    """Collapse `sources` to at most `fan_in` partials by repeated `fan_in`-way combines.

    Each round groups the current frontier into chunks of `fan_in` and calls
    `combine_chunk(chunk, level, index)` once per chunk; the returned values are the next
    frontier. A chunk of one is passed through untouched rather than combined, because
    combining a single partial with nothing is pure cost — it is the level's straggler, and
    at `n = f^k + 1` there is exactly one of them per level.

    The caller finishes: what comes back is a frontier of at most `fan_in` partials, which
    is precisely what one `combine_finalize` consumes. Splitting it there rather than
    finalizing here is what lets the same function serve a reduce that finalizes, one that
    republishes partial state for a later stage, and one that merges sorted runs.

    **This is result-preserving for any `fan_in`,** and that is a property of the algebra
    rather than of the schedule: `combine` is associative and commutative, so every
    bracketing of the same multiset of partials yields the same state. `fan_in` therefore
    trades critical-path length against per-node fan-out and can never trade correctness.
    (Floating-point reductions move in their last bits under re-association, the same stated
    exception the partition count already carries.)

    Args:
        sources: The partials to reduce, in any order.
        combine_chunk: Called as `(chunk, level, index)` for each multi-element chunk;
            returns the merged partial. `level` counts interior rounds from 0.
        fan_in: The maximum partials any one combine may read, at least 2.
        on_level: Optional callback `(level, n_chunks)` after each round, for progress
            reporting and stage numbering.

    Returns:
        The remaining frontier, at most `fan_in` long. Returned as-is when `sources`
        already fits, so a small shuffle pays nothing for the tree existing.
    """
    f = max(2, fan_in)
    frontier: list[T] = list(sources)
    level = 0
    while len(frontier) > f:
        groups = chunks(frontier, f)
        frontier = [
            combine_chunk(g, level, i) if len(g) > 1 else g[0] for i, g in enumerate(groups)
        ]
        if on_level is not None:
            on_level(level, len(groups))
        level += 1
    return frontier
