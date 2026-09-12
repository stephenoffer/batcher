"""Keeping a streamed pipeline's stage pools alive: probe, grow, replace, replay.

The scheduling loop in `schedule` decides *what* to run; this module owns everything it does
when the fleet under it changes. Two shapes, and they are not variations of one another:

**A stage that is behind grows.** `_grow_stage` adds one actor to a stage whose backlog the
loop could not place, registered exactly as a replacement is so a grown pool and a healed one
are the same object.

**A stage that loses an actor replays.** A relay's published morsels live on *its* Flight
server, so losing the actor loses them and every morsel derived from them -- but each one's
parent is still held on the stage below, which is the whole reason a morsel is held until its
subtree finishes. The repair is therefore local: re-queue those parents and let the
deterministic replay reproduce the same paths, so `results` overwrites idempotently instead of
accumulating duplicates. A producer is the exception: it holds its partition's open iterator,
so there is no finer unit to replay than the partition.

Split out of `schedule` when that module outgrew the size limit. The dependency points one
way -- `schedule` imports this, never the reverse -- so the loop's state machine and its
repair logic can be read separately.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import ray

if TYPE_CHECKING:  # the loop's shared state, defined by the module that drives these steps
    from batcher.dist.streaming.pipeline.schedule import _Context


def _probe(pool) -> dict:
    from batcher.dist.streaming.consumers import probe_consumer_hosts

    return probe_consumer_hosts(pool)


def _probe_addrs(pool) -> dict:
    """Each actor's Flight address, fetched once at pool construction rather than per morsel."""
    try:
        return dict(zip(pool, ray.get([a.addr.remote() for a in pool]), strict=True))
    except AttributeError:
        return {}  # a terminal consumer runs no server of its own and publishes nothing


def _grow_stage(ctx: _Context, k: int) -> None:
    """Add one actor to stage `k`, registered exactly as a replacement actor is.

    Best-effort: a cluster with no room refuses the actor, and a stage that cannot grow must
    keep running at the size it has rather than fail the query over an optimization.
    """
    try:
        fresh = ctx.spawn[k]()
    except Exception as exc:  # pragma: no cover - depends on live cluster capacity
        from batcher._internal.logging import note_suppressed

        note_suppressed("dist", "grow a streaming stage pool", exc)
        return
    if ctx.alive is not None:
        ctx.alive.add(fresh)
    ctx.pools[k].append(fresh)
    ctx.free[k].extend([fresh] * ctx.depth)
    ctx.hosts[k].update(_probe([fresh]))
    ctx.addr_of.update(_probe_addrs([fresh]))


def _replace_actor(ctx: _Context, dead_actor, k: int) -> None:
    """Drop a lost stage-`k` actor, void what it was holding, and spawn a replacement.

    Voiding is the part that matters. A relay's published morsels live on *its* Flight server,
    so losing the actor loses them — and every morsel derived from them further up. Each one's
    parent is still held on the stage below (that is why a morsel is held until its subtree
    finishes), so the repair is to re-queue those parents and let the deterministic replay
    reproduce the same paths.
    """
    if dead_actor in ctx.dead:
        return
    ctx.dead.add(dead_actor)
    while dead_actor in ctx.free[k]:
        ctx.free[k].remove(dead_actor)  # every slot of it, not just the first
    with contextlib.suppress(ValueError):
        ctx.pools[k].remove(dead_actor)
    ctx.outstanding.pop(dead_actor, None)
    if k < ctx.last:
        _void_published_by(ctx, dead_actor, k)
    factory = ctx.spawn[k]
    if factory is None:
        return
    fresh = factory()
    if ctx.alive is not None:
        ctx.alive.add(fresh)
    ctx.pools[k].append(fresh)
    ctx.free[k].extend([fresh] * ctx.depth)
    ctx.hosts[k].update(_probe([fresh]))
    ctx.addr_of.update(_probe_addrs([fresh]))


def _void_published_by(ctx: _Context, dead_actor, k: int) -> None:
    """Forget every morsel a lost relay published and re-queue the parents that produced them."""
    replay: dict[tuple, tuple] = {}
    for path in ctx.morsels.paths_held_by(dead_actor):
        record = ctx.morsels.pop(path)
        parent = record["parent"]
        if parent is not None and parent in ctx.morsels:
            replay[parent] = ()
    for entry in [e for e in ctx.ready[k + 1] if e[2] is dead_actor]:
        ctx.ready[k + 1].remove(entry)
    for parent in replay:
        record = ctx.morsels.get(parent)
        if record is None:
            continue
        record["pending"] = None
        holder = record["holder"]
        ctx.ready[k].append((ctx.addr_of.get(holder), record["ticket"], holder, parent, 0))


def _lose_producer(ctx: _Context, dead_producer, *, exc) -> None:
    """Re-queue a lost producer's whole partition, and forget everything derived from it.

    A producer holds its partition's open iterator, so there is no finer unit to replay than
    the partition. Every path descended from it starts with the partition index, so the replay
    regenerates exactly the paths being dropped here and `results` overwrites idempotently.
    """
    if dead_producer in ctx.dead:
        return
    ctx.dead.add(dead_producer)
    with contextlib.suppress(ValueError):
        ctx.free[0].remove(dead_producer)
    with contextlib.suppress(ValueError):
        ctx.pools[0].remove(dead_producer)
    st = ctx.state.pop(dead_producer, None)
    if st is not None:
        pidx = st["pidx"]
        _void_partition(ctx, pidx)
        ctx.part_attempts[pidx] = ctx.part_attempts.get(pidx, 0) + 1
        if ctx.part_attempts[pidx] > ctx.max_attempts:
            raise exc  # a partition that keeps killing its producer is not recoverable
        ctx.pending_parts.append((pidx, st["desc"]))
    factory = ctx.spawn[0]
    if factory is None:
        return
    fresh = factory()
    if ctx.alive is not None:
        ctx.alive.add(fresh)
    ctx.pools[0].append(fresh)
    ctx.free[0].append(fresh)


def _void_partition(ctx: _Context, pidx: int) -> None:
    """Drop every morsel descended from partition `pidx`, wherever it is waiting."""
    for path in ctx.morsels.paths_under_partition(pidx):
        record = ctx.morsels.pop(path)
        holder = record["holder"]
        if record["parent"] is not None:
            ctx.outstanding[holder] = max(0, ctx.outstanding.get(holder, 0) - 1)
    for k in range(1, len(ctx.pools)):
        for entry in [e for e in ctx.ready[k] if e[3] and e[3][0] == pidx]:
            ctx.ready[k].remove(entry)
