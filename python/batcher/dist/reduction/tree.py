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

from collections.abc import Sequence
from typing import TypeVar

__all__ = ["chunks", "reduce_levels", "tree_reduce"]

T = TypeVar("T")


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
