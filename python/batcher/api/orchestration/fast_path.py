"""The small-query fast path: Kyber and the engine, and nothing else.

On a query that returns in a millisecond, the conductor *is* the cost. Measured at 10,000
rows with the event log off: `collect()` takes 1.935 ms, of which `execute_plan` plus the
Arrow table build is **0.360 ms**. The other 82% is Carbonite admission, adaptive morsel
sizing, pressure classification, the resource decision, profile assembly, the event bus, and
the learned-stats close-out. DuckDB answers the same query in 0.780 ms, so that 82% is the
entire gap and then some — Batcher's *engine* is 3.4-4.2x faster than DuckDB at every scale
from 10,000 to 10,000,000 rows.

This module is the escape hatch, off by default (`execution.fast_path`). It runs the same
optimized plan through the same `core.execute_local` the ordinary path calls, so the result
is identical by construction. What it gives up is **adaptivity and observability**, and the
gate below exists to make sure it only gives them up where they cannot matter.

What it does not do, and why each is safe under `eligible`:

`Carbonite admission and the memory reservation`
    Both defend against a plan that will not fit. The gate caps resident input rows, and the
    plan's own state is bounded by that input, so a plan that clears the gate cannot approach
    the envelope. A query that might spill is refused rather than admitted cheaply.
`Adaptive morsel sizing`
    Result-invariant by design (it batches data, never changes output) and inert unless the
    `PressureMonitor` reports ELEVATED — which the row cap makes unreachable for these plans.
`Profile assembly, the event log, the event bus`
    Pure observability. The trade is explicit: a fast-path query does not appear in
    `explain(analyze=True)`, the dashboard, or the JSON event log.
`Every write side of the learned-stats loop`
    **This is the real cost.** Two things go: `_close_learning_loops` (the measured output
    cardinality, the filter's measured selectivity, the per-column distinct-count and
    quantile sketches) and the per-operator `ExecMetrics` that calibrate the cost model — the
    engine call passes `feedback=None`. Recording the metrics alone was a quarter of what
    this path had left to spend, and keeping half a feedback loop is a worse contract than
    keeping none: it is harder to reason about which estimates are stale.

    The *read* side is untouched. `_optimize` still consults learned stats and column NDV, so
    a fast-path query plans exactly as well as an ordinary one — it just does not improve the
    next one. The intra-query adaptive loop is already disabled below 20M rows
    (`api/adaptive/gating.py`), so only the cross-query half is at stake, but that half is
    the moat and turning this on is opting out of it.

The plan is still optimized, through Kyber's ordinary plan cache, so plan *quality* is
unchanged; only the measurement that would sharpen the next plan is skipped.

## Re-issuing the same query costs less again

Skipping the orchestration leaves the *derivation* — optimize, resolve, serialize — and for
a repeated query that derivation cannot have changed. `run_fast` therefore stores it in
`prepared.py`, and the next identical query skips this module too, reaching the engine from
a dict lookup. Measured over five small shapes at 1,000 rows, against the ordinary path:
control-plane overhead 5.30 ms -> 0.21 ms (**25x**), end-to-end 5.89 ms -> 0.81 ms (7.3x),
with the engine itself accounting for 0.60 ms of what remains.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from batcher.config import active_config

if TYPE_CHECKING:
    from batcher.io.source import Source
    from batcher.plan.logical import LogicalPlan

__all__ = ["eligible", "run_fast"]

#: Resident input rows above which the fast path declines. Memory is the reason: the skipped
#: admission and reservation are what stand between a large plan and an OOM, so the gate has
#: to keep the input small enough that neither could have fired. Ten million rows of the
#: widest plausible small-query row is comfortably inside any envelope the engine runs under,
#: and it is two orders of magnitude above the scale where the orchestration dominates
#: (the fixed cost is ~1.6 ms; by 1M rows execution is already 3.2 ms and growing).
MAX_FAST_PATH_ROWS = 10_000_000

#: Plan nodes above which the fast path declines. Not a cost bound — the optimizer is cached
#: either way — but a complexity bound: a deep plan is one whose breakers and join order the
#: adaptive and admission machinery exist to get right, and skipping them there is exactly
#: the trade this path must not make silently.
MAX_FAST_PATH_NODES = 8


def eligible(
    plan: LogicalPlan,
    sources: list[Source],
    *,
    distributed: bool,
    adaptive: bool,
    spill: bool,
    backend: str,
    cache: bool,
) -> bool:
    """Whether `plan` may take the fast path — cheap, structural, and conservative.

    Every test is a config read, an `isinstance`, or a cached count; nothing here reads a
    file or a footer, because a gate that cost what it saves would be pointless. A `False`
    is always safe (the ordinary path runs), so every uncertain case answers `False`.

    Args:
        plan: The logical plan about to run.
        sources: The plan's bound sources, in scan order.
        distributed: Whether the conductor routed this query to the cluster.
        adaptive: Whether stage-boundary re-optimization was resolved on.
        spill: Whether the caller asked for an out-of-core run.
        backend: The requested execution backend.
        cache: Whether the caller asked for the result to be cached.

    Returns:
        `True` when the query may skip the orchestration.
    """
    if not active_config().execution.fast_path:
        return False
    # Each of these routes to a different executor, and the fast path is the single-node
    # in-memory one. `cache` is orchestration in its own right (`api.executors`).
    if distributed or adaptive or spill or cache or backend != "cpu":
        return False
    return _plan_is_simple(plan) and _sources_are_small_and_resident(sources)


def _plan_is_simple(plan: LogicalPlan) -> bool:
    """Whether the plan is small and holds no Python callback.

    A `map_batches` disqualifies it outright: a UDF runs in Python through
    `core.execute_with_udfs`, which is a different executor entirely, and its cost dwarfs the
    orchestration this path exists to remove.
    """
    from itertools import islice

    from batcher import core
    from batcher.plan.visitor import walk

    if core.has_map_batches(plan):
        return False
    # Bounded walk: stop one past the cap rather than sizing the whole plan, so a
    # pathologically deep plan costs the same to reject as a shallow one to accept.
    return len(list(islice(walk(plan), MAX_FAST_PATH_NODES + 1))) <= MAX_FAST_PATH_NODES


def _sources_are_small_and_resident(sources: list[Source]) -> bool:
    """Whether every source is already-materialized Arrow, and small enough in total.

    Restricted to `InMemorySource` deliberately. A file or table source would make
    `resolve_sources` do a real read here, and then the skipped admission would be standing
    between an unbounded scan and the memory envelope — the one place it genuinely earns its
    cost. An in-memory source is already resident, so the row count is exact and free rather
    than an estimate from a footer.
    """
    from batcher.io.source.inmemory import InMemorySource

    if not sources:
        return False
    total = 0
    for source in sources:
        if not isinstance(source, InMemorySource):
            return False
        rows = source.row_count()
        if rows is None:
            return False
        total += rows
        if total > MAX_FAST_PATH_ROWS:
            return False
    return True


def run_fast(
    plan: LogicalPlan,
    sources: list[Source],
    columns: list[str],
    *,
    remember_as: tuple | None = None,
) -> pa.Table:
    """Optimize `plan` and run it, with no orchestration around the call.

    Mirrors the three steps the ordinary path takes between admission and the learning
    close-out — optimize, resolve, execute — and reuses the same functions for each, so the
    two paths cannot drift into computing different things.

    When `remember_as` is given, the derivation is additionally stored in the
    prepared-execution cache, so the *next* run of this query skips even these three steps
    and reaches the engine directly. That is where most of the remaining latency goes: the
    optimize is a plan-cache hit and still costs ~43 us, and the routing this path already
    skipped costs another ~120 us before it is reached.

    Args:
        plan: The logical plan to run.
        sources: The plan's bound sources, in scan order.
        columns: The requested output columns, for the empty-result schema.
        remember_as: The prepared-cache key to store this derivation under, or `None` not
            to store it.

    Returns:
        The result table — the same rows, names and types the ordinary path returns.
    """
    from batcher import core
    from batcher.api._join_helpers import _empty_result_schema
    from batcher.api.orchestration.run import _execute_in_memory, _optimize
    from batcher.api.orchestration.stages import resolve_sources

    # The hub is still handed to `_optimize`: it *reads* learned stats and column NDV to
    # plan, and dropping that would cost plan quality, which this path must never do. Only
    # the write side is off.
    ctx = core.ExecutionContext(columns=columns, hub=core.default_hub(), profile=None)
    opt, logical_opt, _decisions = _optimize(plan, sources, ctx)
    resolved = resolve_sources(sources, opt, ctx)
    table = _execute_in_memory(logical_opt, plan, opt, ctx, resolved, feedback=None)
    if remember_as is not None:
        from batcher.api.orchestration.prepared import Prepared, remember

        remember(
            remember_as,
            sources,
            Prepared(
                physical=opt,
                logical=logical_opt,
                # Computed here, off the hot path, so a later empty result costs nothing to
                # schema. `_execute_in_memory` builds the same thing when it needs it.
                empty_schema=_empty_result_schema(plan, columns),
                config=active_config(),
                source_refs=(),
            ),
        )
    return table
