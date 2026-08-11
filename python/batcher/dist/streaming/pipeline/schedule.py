"""The overlap loop: stream partitions through N stage pools, morsel by morsel.

`driver` builds the pools and hands them here; this module owns the scheduling — which
morsel goes to which actor, how far a stage may run ahead of the one above it, and what
happens when an actor is preempted mid-morsel.

**The shape.** Stage 0 actors open a partition and publish its output one morsel at a time on
their node-local Flight server. Each stage above fetches a morsel in place, runs its own
sub-plan, and either republishes (a middle stage) or returns rows (the last one). Only
`(addr, ticket)` ever crosses Ray; no morsel is resident on the driver at any hop.

**Two nested windows bound memory.** The Flight credit window bounds a morsel's wire buffer.
The *production* window bounds how far a stage may run ahead: an actor holding `credits`
published-but-unreleased morsels is not given more work, so its resident output is bounded by
`credits` morsels regardless of how large the partition is.

**Morsels are named by their path, and that is what makes recovery cheap.** A stage-0 morsel
is `(pidx, seq)`; a morsel produced from it by stage *k* is its parent's path plus the output
index. The path is *deterministic*: replay the same ancestor through a fresh actor and the
same paths come back with the same rows, because every sub-plan here is breaker-free and
deterministic. So a recovery re-run overwrites the results it recomputes instead of
duplicating them — which is the one failure a scheduler like this can have that no exception
would ever reveal.

**A morsel is held until its descendants are done.** Releasing it as soon as the stage above
had consumed it would be cheaper by one morsel per hop, and would mean a preempted middle
stage had no input left to replay — turning a lost actor into a whole re-read of its
partition. Holding instead makes the recovery local: re-dispatch the parent, which is still
sitting on the stage below.
"""

from __future__ import annotations

import contextlib
from collections import deque

import ray

from batcher.dist.streaming.consumers import take_consumer

__all__ = ["run_streamed"]


def _worker_loss_errors() -> tuple[type[BaseException], ...]:
    """Ray exception types that signal a lost actor/task (safe to recover), built lazily
    so `ray` stays an optional import."""
    try:
        import ray as _ray

        return (_ray.exceptions.RayActorError, _ray.exceptions.RayTaskError)
    except Exception:  # pragma: no cover - ray optional
        return ()


class _Morsels:
    """Every published-but-unsettled morsel, and the parent each one came from.

    Small enough to be a dict of dicts and important enough not to be: settling a morsel has
    to cascade to its parent, and the cascade is where a scheduling bug turns into a memory
    leak (a morsel nobody releases) or a wrong answer (a morsel released while its subtree is
    still being recomputed).
    """

    __slots__ = ("_by_path", "_ids")

    def __init__(self) -> None:
        self._by_path: dict[tuple, dict] = {}
        # A stable small integer per path, because a Flight ticket's fields are integers. The
        # mapping persists for the whole run, so a replayed ancestor is re-issued the *same*
        # id and therefore republishes under the same tickets — the idempotence the module
        # docstring depends on.
        self._ids: dict[tuple, int] = {}

    def id_of(self, path: tuple) -> int:
        """A stable integer naming `path`, minted once and reused on every replay."""
        if path not in self._ids:
            self._ids[path] = len(self._ids)
        return self._ids[path]

    def add(self, path: tuple, *, holder, ticket, parent: tuple | None) -> None:
        self._by_path[path] = {
            "holder": holder,
            "ticket": ticket,
            "parent": parent,
            "pending": None,  # children not yet settled; None until this morsel is consumed
        }

    def get(self, path: tuple) -> dict | None:
        return self._by_path.get(path)

    def pop(self, path: tuple) -> dict | None:
        return self._by_path.pop(path, None)

    def paths_held_by(self, actor) -> list[tuple]:
        return [p for p, rec in self._by_path.items() if rec["holder"] is actor]

    def paths_under_partition(self, pidx: int) -> list[tuple]:
        return [p for p in self._by_path if p and p[0] == pidx]

    def __contains__(self, path: tuple) -> bool:
        return path in self._by_path


def run_streamed(
    pools: list[list],
    partitions: list,
    plan_id: int,
    credits: int,
    *,
    spawn: list | None = None,
    alive: set | None = None,
    ceilings: list[int] | None = None,
) -> dict:
    """Stream every partition through the stage pools, overlapped and credit-bounded.

    Args:
        pools: One actor pool per stage, bottom-up. `pools[0]` holds producers (they open a
            partition), every pool above holds a relay, and the last holds the terminal
            consumers that return rows.
        partitions: The input partition descriptors.
        plan_id: The query's plan id, scoping every Flight ticket.
        credits: The production window — published-but-unreleased morsels one actor may hold.
        spawn: One zero-argument actor factory per stage, used to replace a preempted actor
            and to add one when a stage falls behind. `None` (or a `None` entry) makes a loss
            at that stage re-raise, which is what a single-actor test wants.
        alive: A set every replacement actor is registered in, so the caller can tear down
            actors it never spawned itself.
        ceilings: The actor count each stage may grow to. Omitted, or equal to the pool it
            was given, means that stage never scales — which is every stage whose
            `concurrency` is not a `(min, max)` range.

    Returns:
        `{path: output_batches}` for every morsel that reached the last stage.
    """
    from batcher.config import active_config

    loss_errors = _worker_loss_errors()
    max_attempts = max(1, active_config().distributed.recovery_max_attempts)
    last = len(pools) - 1
    spawn = list(spawn or [None] * len(pools))

    morsels = _Morsels()
    free = [deque(pool) for pool in pools]
    hosts = [_probe(pool) if k else {} for k, pool in enumerate(pools)]
    ready: list[deque] = [deque() for _ in pools]  # ready[k]: morsels awaiting stage k
    outstanding: dict = {}  # actor -> published-but-unreleased morsels it holds
    addr_of: dict = {}  # actor -> its Flight address
    dead: set = set()
    state: dict = {}  # producer -> its current partition's streaming state
    pending_parts = deque(enumerate(partitions))
    part_attempts: dict = {}
    open_inflight: dict = {}
    publish_inflight: dict = {}
    work_inflight: dict = {}
    results: dict = {}

    ctx = _Context(
        pools=pools,
        spawn=spawn,
        alive=alive,
        free=free,
        hosts=hosts,
        ready=ready,
        morsels=morsels,
        outstanding=outstanding,
        addr_of=addr_of,
        dead=dead,
        state=state,
        pending_parts=pending_parts,
        part_attempts=part_attempts,
        max_attempts=max_attempts,
        credits=credits,
        plan_id=plan_id,
        last=last,
        results=results,
        ceilings=list(ceilings) if ceilings else [len(p) for p in pools],
        floors=[len(p) for p in pools],
    )
    for k, pool in enumerate(pools):
        if k:
            addr_of.update(_probe_addrs(pool))

    while True:
        _start_partitions(ctx, open_inflight)
        _issue_publishes(ctx, publish_inflight)
        _dispatch(ctx, work_inflight)
        # After dispatch, so `ready[k]` holds exactly what this round could NOT place: a
        # backlog measured before dispatching would count morsels an idle actor was about to
        # take and grow the pool for work that was never queued.
        _rescale_stages(ctx)

        waitset = [*open_inflight, *publish_inflight, *work_inflight]
        if not waitset:
            _assert_not_stalled(ctx)
            break
        ref = ray.wait(waitset, num_returns=1)[0][0]
        if ref in open_inflight:
            _on_open(ctx, open_inflight, ref, loss_errors)
        elif ref in publish_inflight:
            _on_publish(ctx, publish_inflight, ref, loss_errors)
        else:
            _on_work(ctx, work_inflight, ref, loss_errors)
    return results


class _Context:
    """Everything the loop's steps share, in one object so each step stays a small function."""

    __slots__ = (
        "addr_of",
        "alive",
        "ceilings",
        "credits",
        "dead",
        "floors",
        "free",
        "hosts",
        "last",
        "max_attempts",
        "morsels",
        "outstanding",
        "part_attempts",
        "pending_parts",
        "plan_id",
        "pools",
        "ready",
        "results",
        "spawn",
        "state",
    )

    def __init__(self, **kw) -> None:
        for name, value in kw.items():
            setattr(self, name, value)


def _probe(pool) -> dict:
    from batcher.dist.streaming.consumers import probe_consumer_hosts

    return probe_consumer_hosts(pool)


def _probe_addrs(pool) -> dict:
    """Each actor's Flight address, fetched once at pool construction rather than per morsel."""
    try:
        return dict(zip(pool, ray.get([a.addr.remote() for a in pool]), strict=True))
    except AttributeError:
        return {}  # a terminal consumer runs no server of its own and publishes nothing


# --- issuing work ----------------------------------------------------------------------


def _start_partitions(ctx: _Context, open_inflight: dict) -> None:
    """Give every free producer a partition — the initial fan-out and every recovery re-run."""
    while ctx.free[0] and ctx.pending_parts:
        prod = ctx.free[0].popleft()
        pidx, desc = ctx.pending_parts.popleft()
        ctx.state[prod] = {
            "pidx": pidx,
            "desc": desc,
            "seq": 0,
            "outstanding": 0,
            "done": False,
            "open": False,
        }
        open_inflight[prod.open.remote(desc)] = prod


def _issue_publishes(ctx: _Context, publish_inflight: dict) -> None:
    """Ask each opened producer with window headroom for its next morsel (one at a time)."""
    from batcher.carbonite.transfer import ShuffleTicket

    publishing = {p for p, _pidx, _seq, _t in publish_inflight.values()}
    for prod, st in ctx.state.items():
        has_headroom = st["open"] and not st["done"] and st["outstanding"] < ctx.credits
        if has_headroom and prod not in publishing:
            seq = st["seq"]
            st["seq"] += 1
            ticket = ShuffleTicket(ctx.plan_id, 0, st["pidx"], seq)
            publish_inflight[prod.publish_next.remote(ticket)] = (prod, st["pidx"], seq, ticket)


def _dispatch(ctx: _Context, work_inflight: dict) -> None:
    """Hand ready morsels to free actors, stage by stage, preferring a same-node actor."""
    for k in range(1, len(ctx.pools)):
        while ctx.ready[k]:
            addr, ticket, holder, path, attempts = ctx.ready[k][0]
            actor = _take(ctx, k, addr)
            if actor is None:
                break  # no free actor with window headroom; try again next iteration
            ctx.ready[k].popleft()
            ref = (
                actor.run_split.remote(addr, ticket)
                if k == ctx.last
                else actor.consume.remote(addr, ticket, ctx.plan_id, k, ctx.morsels.id_of(path))
            )
            work_inflight[ref] = (k, actor, holder, path, addr, ticket, attempts)


def _rescale_stages(ctx: _Context) -> None:
    """Grow a stage that is behind, shrink one that is idle — the per-stage autoscaler.

    A pipeline's stages do not run at the same rate, and the slow one decides the whole
    query: a CPU decode stage that cannot keep up leaves the GPU stage above it waiting on an
    empty queue, which is the single most expensive shape in the corpus these pipelines exist
    for. Pool sizes were fixed at construction, so a stage that fell behind stayed behind for
    the whole run whatever the cluster had spare.

    The signal is the backlog *after* dispatch: morsels queued for stage `k` that no actor
    could take. Anything left there is work the stage is not keeping up with, and it is the
    same `pending > 0 and n < max` rule the inference actor pool already scales on, reused
    rather than restated so the two paths cannot drift.

    It grows and does not shrink, which is the opposite of the actor-pool path and is
    deliberate. There, `pending` is a partition queue that only drains, so a reap at the tail
    is safe. Here the backlog rises and falls morsel by morsel, so the same rule would reap an
    actor the moment a stage caught up and re-spawn it on the next morsel — and each of those
    actors holds a *loaded model*, whose reload costs tens of seconds against the fraction of
    a second of idling it saves. The pools are torn down when the query ends, so nothing is
    leaked by keeping them; a stage that grew because it was behind stays wide.

    A second reason not to reap here: a relay's published morsels live on *its* Flight server,
    so killing one that still holds any voids a subtree and forces a replay. Growth has no
    such hazard.

    The field guidance goes further and is worth knowing before anyone adds the reap back:
    "for GPU workloads with expensive initialization (model loading), always use a fixed pool;
    autoscaling is appropriate for lightweight CPU-only transforms where load varies and
    initialization cost is negligible" (`../optimization-guides`,
    `foundations/data/streaming/streaming-execution.md`). That is why scaling here is opt-in
    rather than default: a stage gets a fixed pool unless its `concurrency` is a `(min, max)`
    range, which is the user saying the trade is theirs to make.
    """
    from batcher.dist.executors.map import _autoscale_action

    for k in range(1, len(ctx.pools)):
        if ctx.spawn[k] is None or ctx.ceilings[k] <= ctx.floors[k]:
            continue  # a fixed-size stage: nothing was asked for and nothing is done
        action = _autoscale_action(
            len(ctx.ready[k]),
            len(ctx.pools[k]),
            len(ctx.free[k]),
            ctx.floors[k],
            ctx.ceilings[k],
        )
        if action == "up":
            _grow_stage(ctx, k)


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
    ctx.free[k].append(fresh)
    ctx.hosts[k].update(_probe([fresh]))
    ctx.addr_of.update(_probe_addrs([fresh]))


def _take(ctx: _Context, k: int, addr: str):
    """A free stage-`k` actor with production-window headroom, or `None`."""
    pool = ctx.free[k]
    if not pool:
        return None
    if k == ctx.last:
        return take_consumer(pool, ctx.hosts[k], addr)
    # A relay publishes what it produces, so it is subject to the same window as a producer:
    # one holding `credits` unreleased morsels must not be given more work, or the bound this
    # pipeline advertises would hold at the first hop and nowhere else.
    eligible = deque(a for a in pool if ctx.outstanding.get(a, 0) < ctx.credits)
    if not eligible:
        return None
    chosen = take_consumer(eligible, ctx.hosts[k], addr)
    pool.remove(chosen)
    return chosen


# --- completions -----------------------------------------------------------------------


def _on_open(ctx: _Context, open_inflight: dict, ref, loss_errors) -> None:
    prod = open_inflight.pop(ref)
    try:
        ctx.addr_of[prod] = ray.get(ref)
    except loss_errors as exc:
        _lose_producer(ctx, prod, exc=exc)
        return
    if prod in ctx.state:
        ctx.state[prod]["open"] = True


def _on_publish(ctx: _Context, publish_inflight: dict, ref, loss_errors) -> None:
    prod, pidx, seq, ticket = publish_inflight.pop(ref)
    try:
        more = ray.get(ref)
    except loss_errors as exc:
        _lose_producer(ctx, prod, exc=exc)
        return
    st = ctx.state.get(prod)
    if st is None:
        return  # the producer was lost between issuing this publish and its completion
    if not more:
        st["done"] = True
        if st["outstanding"] == 0:
            _recycle(ctx, prod)
        return
    st["outstanding"] += 1
    path = (pidx, seq)
    ctx.morsels.add(path, holder=prod, ticket=ticket, parent=None)
    ctx.ready[1].append((ctx.addr_of[prod], ticket, prod, path, 0))


def _on_work(ctx: _Context, work_inflight: dict, ref, loss_errors) -> None:
    k, actor, holder, path, addr, ticket, attempts = work_inflight.pop(ref)
    try:
        out = ray.get(ref)
    except loss_errors as exc:
        _replace_actor(ctx, actor, k)
        # The morsel is still published on its holder, so re-dispatch it — unless the holder
        # itself is gone, in which case its own ancestor is already being replayed and this
        # morsel will come back under the same path.
        if holder not in ctx.dead and (path in ctx.morsels):
            if attempts + 1 > ctx.max_attempts:
                raise exc
            ctx.ready[k].append((addr, ticket, holder, path, attempts + 1))
        return
    ctx.free[k].append(actor)
    if k == ctx.last:
        ctx.results[path] = out
        _settle(ctx, path)
        return
    published = int(out or 0)
    if published == 0:
        _settle(ctx, path)  # this stage produced nothing from it, so nothing depends on it
        return
    record = ctx.morsels.get(path)
    if record is None:
        return  # the morsel was voided while this call was in flight
    record["pending"] = published
    _publish_children(ctx, k, actor, path, published)


def _publish_children(ctx: _Context, k: int, actor, path: tuple, published: int) -> None:
    """Register the morsels a relay just published and queue them for the stage above."""
    from batcher.carbonite.transfer import ShuffleTicket

    morsel_id = ctx.morsels.id_of(path)
    addr = ctx.addr_of.get(actor)
    for i in range(published):
        child = (*path, i)
        ticket = ShuffleTicket(ctx.plan_id, k, morsel_id, i)
        ctx.morsels.add(child, holder=actor, ticket=ticket, parent=path)
        ctx.outstanding[actor] = ctx.outstanding.get(actor, 0) + 1
        ctx.ready[k + 1].append((addr, ticket, actor, child, 0))


# --- settling and recovery ---------------------------------------------------------------


def _settle(ctx: _Context, path: tuple) -> None:
    """Release `path` now that its whole subtree is done, and cascade to its parent."""
    while True:
        record = ctx.morsels.pop(path)
        if record is None:
            return
        holder, parent = record["holder"], record["parent"]
        if holder not in ctx.dead:
            with contextlib.suppress(Exception):
                holder.release.remote(record["ticket"])
        if parent is None:
            st = ctx.state.get(holder)
            if st is not None:
                st["outstanding"] -= 1
                if st["done"] and st["outstanding"] == 0:
                    _recycle(ctx, holder)
            return
        ctx.outstanding[holder] = max(0, ctx.outstanding.get(holder, 0) - 1)
        parent_record = ctx.morsels.get(parent)
        if parent_record is None or parent_record["pending"] is None:
            return
        parent_record["pending"] -= 1
        if parent_record["pending"] > 0:
            return
        path = parent  # the parent's subtree is done too: settle it in the same loop


def _recycle(ctx: _Context, prod) -> None:
    """A producer whose partition is fully drained takes the next one, or goes idle."""
    ctx.state.pop(prod, None)
    if prod not in ctx.dead:
        ctx.free[0].append(prod)


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
    with contextlib.suppress(ValueError):
        ctx.free[k].remove(dead_actor)
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
    ctx.free[k].append(fresh)
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


def _assert_not_stalled(ctx: _Context) -> None:
    """Nothing in flight and nothing issuable, yet morsels waiting, means a pool emptied.

    Returning would hand back a *partial* result that looks complete: every unconsumed
    morsel's rows would be missing from the answer with no error anywhere.
    """
    waiting = sum(len(q) for q in ctx.ready)
    if not waiting:
        return
    from batcher._internal.errors import ResourceError

    raise ResourceError(
        f"distributed streaming pipeline stalled with {waiting} morsel(s) unconsumed and no "
        "actor left to run them (every actor of a stage was lost and none could be replaced)"
    )
