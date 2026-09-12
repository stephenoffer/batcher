"""Which inputs of a spilling breaker are themselves breakers, and how to stage them.

The out-of-core sort, window and join stage each input by running that input's sub-plan
**once per input morsel** (`spill_breakers.sort::stage_and_partition`,
`spill_breakers.window`, and the join's per-side map all call
`execute_plan(map_ir, [[batch]])` in a loop). That is exactly right for the linear
scan/filter/project chain those paths were written for, and silently wrong for a *breaker*
underneath: each morsel yields a partial, the partials are partitioned and processed, and
nothing ever combines them.

This module is the guard. It sits in its own file rather than beside the dispatcher because
it answers a question none of the three breakers should answer twice — *is this input a
breaker, and what does the plan look like once it has been spilled* — and because the
dispatcher it serves is already at the size limit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    Project,
    Sort,
    Window,
)

if TYPE_CHECKING:
    from batcher.io.source import Source

__all__ = ["peel_to_breaker", "stage_breaker_inputs"]


def peel_to_breaker(plan: LogicalPlan) -> LogicalPlan | None:
    """The spillable breaker under a chain of leading row-wise/limit ops, or `None`.

    Peels `Project`/`Filter`/`Limit` and returns the underlying node if it is a spillable
    breaker (`Distinct`/`Aggregate`/`Join`/`Sort`/`Window`) — the marker that an operator's
    input must itself be spilled out-of-core rather than streamed per batch.
    """
    node = plan
    while isinstance(node, (Project, Filter, Limit)):
        node = node.input
    return node if isinstance(node, (Distinct, Aggregate, Join, Sort, Window)) else None


def stage_breaker_inputs(
    plan: LogicalPlan, sources: list[Source], num_partitions: int
) -> tuple[LogicalPlan, list[Source]] | None:
    """Spill each of `plan`'s breaker-bearing inputs, spliced back as a staged scan.

    Returns `(rewritten plan, extended sources)` with every input that peels to a spillable
    breaker replaced by a `Scan` of that breaker's out-of-core result, or `None` when `plan`
    has no such input — the ordinary case, which leaves the caller's existing dispatch
    untouched.

    **Why the ordering and binary breakers need this and the aggregate did not.** Each of
    them stages its input by running that input's sub-plan **once per input morsel**
    (`spill_breakers.sort::stage_and_partition`, `spill_breakers.window`, and the join's
    per-side map all call `execute_plan(map_ir, [[batch]])` in a loop). That is exactly
    right for the linear scan/filter/project chain the paths were written for, and silently
    wrong for a breaker: each morsel yields a *partial*, the partials are partitioned and
    processed, and nothing ever combines them.

    Neither predicate could see it. `supports_spilling_sort` asks whether the leading key
    range-partitions and whether the input names one source; `supports_spilling_join` asks
    whether each side names one source. `Sort(Aggregate(Scan(0)))` and
    `Join(Aggregate(Scan(0)), Scan(1))` answer yes to everything asked.

    Measured, on a 30 MiB fixture (four staging chunks) with 7 groups: both shapes returned
    **28 rows where the answer is 7**, one row per (chunk, group). TPC-H q1 is the same
    shape at scale — at sf10 under a 2 GiB envelope it returned 1,224 rows for 4, silently,
    where the same query uncapped returned 4.

    **Re-entering `spill_collect` on the rewritten plan is what keeps this bounded**, and is
    why this returns a plan rather than a table. Over its staged inputs the operator's
    children are bare `Scan`s — the linear shape the per-morsel map is correct for — so it
    takes its own ordinary out-of-core route. Handing the result to the in-memory engine
    instead would spill a high-cardinality `GROUP BY` and then materialize its whole output
    to sort it, trading one OOM for another a step later, and only the low-cardinality case
    would have looked fixed.

    Args:
        plan: The breaker whose inputs to stage — its `input`, or its `left`/`right`.
        sources: The plan's bound sources. Never mutated; the staged ones are appended to a
            copy, so the indices the untouched side's scans hold stay valid.
        num_partitions: The fan-out to spill each inner breaker with.

    Returns:
        The rewritten plan and its extended source list, or `None` when nothing was staged
        (no breaker input, or one this path cannot spill).
    """
    import dataclasses

    # Lazy: `aggregate` imports this module, so a top-level import here would close the
    # cycle. Same convention `aggregate` already uses for `spill_breakers`.
    from batcher.dist.spill.aggregate import spill_collect
    from batcher.io.source import InMemorySource
    from batcher.plan.logical import Scan
    from batcher.plan.schema import SchemaRef

    staged_sources = list(sources)
    replacements: dict[str, LogicalPlan] = {}
    for attr in ("input", "left", "right"):
        side = getattr(plan, attr, None)
        if not isinstance(side, LogicalPlan) or peel_to_breaker(side) is None:
            continue
        inner = spill_collect(side, sources, num_partitions)
        if inner is None:
            # This side has no out-of-core path, so there is no bounded rewrite to offer.
            # Declining wholesale (rather than staging only the other side) keeps the caller
            # on the single route it already knows how to reason about.
            return None
        # A zero-row table has no batches (pyarrow drops empty chunks) and `InMemorySource`
        # needs one to carry the schema — the same guard `_apply_above` makes.
        batches = inner.to_batches() or [pa.RecordBatch.from_pylist([], schema=inner.schema)]
        replacements[attr] = Scan(len(staged_sources), SchemaRef.from_arrow(inner.schema))
        staged_sources.append(InMemorySource(batches))
    if not replacements:
        return None
    return dataclasses.replace(plan, **replacements), staged_sources
