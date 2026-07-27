"""Free a finished query's shuffle buckets, so a reused fleet does not grow without bound.

The Flight `PartitionStore` is append-only: a bucket published by a mapper stays resident
until something explicitly evicts it. The eviction calls all existed —
`ShuffleSession.{release, clear_plan, clear}`, bound through `bc_py` to the Rust
`PartitionStore` and unit-tested there — but on the **batch** path nothing ever called
them. Only `dist/streaming/pipeline.py` released anything.

With `distributed.reuse_session_fleet` on by default, the fleet outlives the query that
created it, so that omission meant every bucket of every stage of every query stayed in
worker memory until the node died. It presents as an out-of-memory kill on the Nth query
of a session with no indication that queries 1..N-1 are the cause.

# Why eviction is deliberately coarse

An unregistered ticket reads back as an **empty bucket, not an error** — the epoch
invariant documented in `dist/shuffle_replication.py`. So evicting a bucket that someone
still intends to read does not fail loudly; it silently returns zero rows. That makes
premature eviction a wrong-answer bug, and it is why this evicts only at points where
everything downstream is provably finished:

- `query_shuffle_scope`'s exit, which *is* the definition of "this query is over", and
- `FlightMaterializedSource.cleanup()`, which runs once the next stage has consumed the
  intermediate.

A refcount-when-fetched scheme would free memory sooner and is the obvious improvement.
It is also exactly where the silent-loss bug would come from, so it is not what this does.

# Why this cannot wedge an actor

Worth writing down, because it is the first thing to suspect if a distributed run ever
hangs after this landed. `clear_plan` reaches Rust as `block_on(exchange.clear_plan(...))`
inside `allow_threads`, and a Ray actor processes one task at a time — so if that call
could block, it would stall the *next* query on a warm session fleet rather than this one,
which is a miserable thing to debug.

It cannot. `PartitionStore::remove_prefix` takes the write lock only to `retain` over the
map, and the fetch path (`get_with_gauge`) clones the partition's `Arc<Vec<RecordBatch>>`
and **releases the read lock before streaming a single batch**. So an in-flight fetch of a
gigabyte partition never holds a lock that eviction waits on; the two contend only for the
few microseconds of a hashmap operation.

The driver side is belt-and-braces regardless: `ray.wait` with a short timeout, results
never `ray.get`-ed, every exception swallowed.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.logging import note_suppressed

__all__ = ["evict_plan", "fleet_actors_for_eviction"]

#: How long the driver waits for the evictions it fired. Short on purpose: eviction is
#: memory hygiene, not correctness, and a query must never be delayed — let alone failed —
#: by a slow teardown. Anything unfinished is simply left; a dead worker has already freed
#: its memory, and a live one clears the plan on its next teardown or when the fleet dies.
_EVICT_TIMEOUT_S = 2.0


def fleet_actors_for_eviction() -> list[Any]:
    """The worker actors holding this query's buckets, or an empty list if there are none.

    Prefers the fleet installed for the current query and falls back to the module-global
    session fleet, which is the one that actually leaks: a per-query fleet is killed
    outright at the end of the query and takes its memory with it.

    Returns:
        The actor handles to evict on, or an empty list when no fleet is live.
    """
    from batcher.dist.fleet import _fleet

    try:
        fleet = _fleet.current_fleet()
        if fleet is not None and fleet.actors:
            return list(fleet.actors)
        with _fleet._SESSION_LOCK:
            session = _fleet._SESSION
            return list(session.actors) if session is not None else []
    except Exception as exc:  # pragma: no cover - teardown must never raise
        note_suppressed("dist", "resolve fleet for bucket eviction", exc)
        return []


def evict_plan(actors: list[Any], plan_id: int) -> None:
    """Ask every worker in `actors` to drop the buckets it published for `plan_id`.

    Fire-and-forget with a bounded wait. The results are deliberately never `ray.get`-ed:
    a worker that died still holds nothing, and one that is wedged must not wedge the
    driver behind it. Every failure is swallowed for the same reason — this runs in a
    `finally`, where raising would replace the query's real outcome (success, or its
    actual error) with a memory-hygiene failure.

    Args:
        actors: The worker actor handles to evict on.
        plan_id: The query's shuffle fence id, as minted by `mint_query_plan_id`.
    """
    if not actors:
        return
    try:
        import ray

        refs = [a.clear_plan.remote(plan_id) for a in actors]
        ray.wait(refs, num_returns=len(refs), timeout=_EVICT_TIMEOUT_S)
    except Exception as exc:  # pragma: no cover - teardown must never raise
        note_suppressed("dist", f"evict shuffle buckets for plan {plan_id}", exc)
