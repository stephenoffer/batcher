"""When a UNION streams branch by branch, and how its branches are addressed.

A UNION ALL's result *is* its branches' results concatenated, so yielding each branch's
own stream in order is bounded in memory and identical to the materialized concatenation
— row for row and in order. That makes it the most streamable of the pipeline breakers,
and `dispatch` routes it here rather than to `_collect`.

This module holds only the *proof obligations*, not the driving: `dispatch` owns the
strategy choice, as it does for every other shape. They live apart because the
preconditions are the subtle half — each one is a wrong answer rather than a slow one if
skipped — and they deserve to be read, and tested, as a unit.
"""

from __future__ import annotations

from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan

__all__ = [
    "union_branch_sources",
    "union_streams_branchwise",
    "union_streams_interleaved",
]


def union_branch_sources(plan) -> list[tuple[LogicalPlan, int]]:
    """Each UNION branch paired with the single source id it scans, or `[]` if any branch
    does not scan exactly one.

    The branch-wise stream drives one branch at a time, and the drivers it delegates to
    address their input as `sources[0]` — so a branch is streamable only when it has a
    single source to be relabelled onto. A branch spanning two sources (a join inside a
    union) returns nothing here, and the union materializes as it did before.

    Args:
        plan: The `Union` node whose branches to inspect.

    Returns:
        `(branch, source_id)` for every branch, or an empty list if any branch reads more
        or fewer than one source.
    """
    from batcher.plan.visitor import scanned_source_ids

    pairs: list[tuple[LogicalPlan, int]] = []
    for branch in plan.inputs:
        ids = scanned_source_ids(branch)
        if len(ids) != 1:
            return []
        pairs.append((branch, next(iter(ids))))
    return pairs


def union_streams_branchwise(plan, sources: list[Source]) -> bool:
    """Whether this UNION's result equals its branches' streams concatenated in order.

    Four preconditions, each of which is a wrong answer rather than a slow one if skipped:

    - **UNION ALL only.** `UNION` (distinct) needs a global dedup, which is precisely the
      whole-relation state this path does not have.
    - **Bounded sources only.** An unbounded branch never terminates, so branch `k + 1`
      would never emit and the "concatenation" would silently be branch 0 forever. Those
      keep raising the explicit `PlanError` the router falls through to.
    - **One source per branch**, so each can be relabelled onto `sources[0]`
      (`union_branch_sources`).
    - **No type promotion due.** The materialized path lets the engine widen a column's
      type across branches (Int32 with Int64 becomes Int64); streaming yields each branch's
      batches as the engine produced them, so it applies only where every branch's types
      already agree. Restating the promotion rule here would be a second copy of what the
      engine owns, and two copies are what drift.

    Args:
        plan: The `Union` node being routed.
        sources: The bound sources for the whole plan.

    Returns:
        True when the branch-wise stream is provably equivalent to materializing.
    """
    from batcher.io.source import is_bounded

    if plan.distinct or not all(is_bounded(s) for s in sources):
        return False
    if not union_branch_sources(plan):
        return False
    schemas = [branch.available_schema() for branch in plan.inputs]
    if any(s is None for s in schemas):
        return False  # an opaque branch (a UDF) — cannot prove the types already agree
    return all(s.arrow.types == schemas[0].arrow.types for s in schemas[1:])


def union_streams_interleaved(plan, sources: list[Source]) -> bool:
    """Whether this UNION's branches may be *interleaved* rather than concatenated.

    Concatenation needs every branch to end, which an unbounded one never does — so a
    union over streams could not stream at all: branch 0 would emit forever and branch 1
    never, and the router refused it with a `PlanError` rather than return that. Spark
    unions streaming DataFrames, and a union of two topics is an ordinary shape (two
    regions, two versions of a producer, a backfill beside a live feed).

    Interleaving is sound where concatenation is, minus the ordering claim — and UNION ALL
    never made one: it is a multiset union, so a row from whichever branch has one next is
    as correct as any other order. The remaining preconditions are `union_streams_branchwise`'s
    and hold for the same reasons: ALL rather than distinct (a global dedup is exactly the
    whole-relation state this path lacks), one source per branch so each can be relabelled,
    and types that already agree across branches so no promotion is due.

    Args:
        plan: The `Union` node being routed.
        sources: The bound sources for the whole plan.

    Returns:
        True when at least one branch is unbounded and interleaving is provably equivalent
        to the materialized multiset.
    """
    from batcher.io.source import is_bounded

    if plan.distinct or all(is_bounded(s) for s in sources):
        return False  # all-bounded unions concatenate, which also preserves order
    if not union_branch_sources(plan):
        return False
    schemas = [branch.available_schema() for branch in plan.inputs]
    if any(s is None for s in schemas):
        return False  # an opaque branch (a UDF) — cannot prove the types already agree
    return all(s.arrow.types == schemas[0].arrow.types for s in schemas[1:])


def interleave(streams: list) -> object:
    """Yield from `streams` round-robin until every one is exhausted.

    One batch from each in turn, so a busy branch cannot starve a quiet one of its place
    in the output and the driver never holds more than one branch's one batch.

    **A branch parked on an idle source delays the others**, because pulling from it is a
    blocking read — the same property the stream-stream join has, and for the same reason:
    there is one driver thread and a source's `iter_batches` decides when it returns. A
    stop signal reaches the sources themselves, so a query still stops promptly; what it
    does not do is skip ahead past a quiet branch mid-poll.

    Args:
        streams: The per-branch iterators, already relabelled onto their own source.

    Returns:
        A generator over every branch's batches, round-robin.
    """

    def gen():
        live = list(streams)
        while live:
            still: list = []
            for stream in live:
                batch = next(stream, None)
                if batch is None:
                    continue  # this branch has ended; drop it from the rotation
                still.append(stream)
                if batch.num_rows:
                    yield batch
            live = still

    return gen()
