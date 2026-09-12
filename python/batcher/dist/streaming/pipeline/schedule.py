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
from collections import Counter, deque

import ray

from batcher.dist.streaming.consumers import take_consumer
from batcher.dist.streaming.pipeline.recovery import (
    _grow_stage,
    _lose_producer,
    _probe,
    _probe_addrs,
    _replace_actor,
)

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
    depth = _consumer_depth()
    # Stage 0 takes whole partitions one at a time (its window is `credits`, and `open` binds
    # it to exactly one); every stage above takes morsels `depth` at a time, so the free list
    # holds *slots*, not actors. Filled actor-major so an idle pool round-robins -- every
    # actor gets its first morsel before any gets its second, the same ordering
    # `_emptiest_actor` enforces on the non-streamed pool and for the same reason: filling
    # actor 0 to its depth before actor 1 receives anything leaves the tail of the pool idle
    # whenever there are fewer morsels in flight than slots.
    free = [deque(pool) for pool in pools[:1]]
    free.extend(deque(a for _ in range(depth) for a in pool) for pool in pools[1:])
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
        depth=depth,
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
        for ref in _completed(waitset):
            if ref in open_inflight:
                _on_open(ctx, open_inflight, ref, loss_errors)
            elif ref in publish_inflight:
                _on_publish(ctx, publish_inflight, ref, loss_errors)
            else:
                _on_work(ctx, work_inflight, ref, loss_errors)
    return results


def _completed(waitset: list) -> list:
    """Block for the first finished call, then take **every** other one already finished.

    One event per pass was the loop's own throughput ceiling, and it is charged per morsel:
    each pass re-runs the issue, dispatch and rescale steps and a `ray.wait` over every
    outstanding call in the pipeline, so a run of 1,562 morsels through two stages pays that
    round of bookkeeping ~3,100 times no matter how much finished at once. Everything the
    driver does here is bookkeeping -- the actors are the only things doing work -- so the
    right size for a pass is "all the news there is", not "one item of it".

    The non-blocking ask comes **first**, and that ordering is the whole cost model. Every
    `ray.wait` is an RPC to the raylet over the whole outstanding set, so a pass that blocks
    for one and then asks for the rest pays two of them on every morsel of a busy pipeline.
    Under load something has almost always finished already, so asking `timeout=0` for
    everything answers the pass in one RPC and the blocking wait is reached only when the
    driver genuinely has nothing to do -- which is the one moment its own overhead does not
    matter.
    """
    done, _rest = ray.wait(waitset, num_returns=len(waitset), timeout=0)
    if done:
        return done
    done, _rest = ray.wait(waitset, num_returns=1)
    return done


class _Context:
    """Everything the loop's steps share, in one object so each step stays a small function."""

    __slots__ = (
        "addr_of",
        "alive",
        "ceilings",
        "credits",
        "dead",
        "depth",
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


def _consumer_depth() -> int:
    """Morsels a stage-`k>0` actor may have in flight at once -- its submit-ahead depth.

    Without it a consumer fetched a morsel over Flight, ran it, returned to the driver, and
    only then learned about the next one, so a device idled through every fetch and every
    Arrow assembly. The non-streamed actor pool has had `distributed.map_inflight_depth` for
    exactly this since it was written; the streamed path, which exists *specifically* to keep
    a device fed, had no equivalent. Same knob, same envelope adaptation (measured GPU
    utilization raises it), so the two paths cannot drift.

    Depth 1 restores the historical one-at-a-time behaviour exactly, including the autoscaler
    signal -- see `_stage_backlog`.
    """
    from batcher.dist.executors.map import _actor_inflight_depth

    return _actor_inflight_depth()


def _stage_backlog(ctx: _Context, k: int, depth: int) -> int:
    """Morsels stage `k` has not started: the driver's queue plus what is queued *on* actors.

    The second term is what makes this depth-safe, and leaving it out is why submit-ahead was
    landed and reverted once already. The autoscaler grows a stage while it is behind, and
    read "behind" as `len(ready[k])` -- morsels the driver could not place. At depth 2 there
    are twice as many slots to place them in, so the queue drained, the stage read as keeping
    up, and it never grew past its floor: measured on eight T4s, **four of them sat idle at
    36% while the depth-1 run used all eight at 59%**.

    A morsel sitting in an actor's second slot is not *being worked on*, it is waiting behind
    the call that is -- backlog held by the actor instead of by the driver. Counting it says
    so. At depth 1 no actor can hold a second morsel, so this is `len(ready[k])` exactly and
    nothing about a depth-1 run changes.
    """
    if depth <= 1:
        return len(ctx.ready[k])
    idle_slots = Counter(ctx.free[k])
    queued = sum(max(0, depth - idle_slots.get(a, 0) - 1) for a in ctx.pools[k])
    return len(ctx.ready[k]) + queued


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
            "inflight": 0,
            "done": False,
            "open": False,
        }
        open_inflight[prod.open.remote(desc)] = prod


def _issue_publishes(ctx: _Context, publish_inflight: dict) -> None:
    """Fill each opened producer's window with `publish_next` calls, up to `credits`.

    **Queued, not one at a time.** A producer used to be asked for morsel *k+1* only after
    the driver had observed *k* return, so the actor sat idle across a full round of this
    loop -- a `ray.wait` over every stage's inflight set, then the handler, then the next
    issue -- between one morsel's work and the next. On a decode stage that gap is paid
    3,125 times over a 400,000-image run, and it is why the staged form measured 2.1x slower
    than the fused one at *both* intermediate widths: the cost was per morsel, not per byte.
    Queueing the whole window instead lets the actor start *k+1* the instant *k* finishes,
    which is what the credit window was always meant to buy.

    The window is unchanged in size, only in what fills it: a morsel that has been asked for
    is as resident as one that has been published, so an issued-but-unreturned call counts
    against `credits` exactly as a published-but-unreleased morsel does. Without that the
    producer would hold `credits` published morsels *plus* `credits` queued calls' worth of
    output, doubling the bound this window exists to enforce.

    Queued **`depth` deep, not `credits` deep**, and the difference is the driver's own cost
    rather than the producer's. Hiding the round-trip needs one call waiting behind the
    running one; past that, each extra queued call is another `ObjectRef` in the set this loop
    calls `ray.wait` on every pass. At `credits` (16) over 64 producers that set is a thousand
    refs, scanned per pass through a raylet RPC, and the scan costs more than the idling it
    removes. `depth` is the same submit-ahead the consumer stages take, for the same reason.
    """
    from batcher.carbonite.transfer import ShuffleTicket

    for prod, st in ctx.state.items():
        while (
            st["open"]
            and not st["done"]
            and st["inflight"] < ctx.depth
            and st["outstanding"] + st["inflight"] < ctx.credits
        ):
            seq = st["seq"]
            st["seq"] += 1
            st["inflight"] += 1
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
        idle_slots = Counter(ctx.free[k])
        action = _autoscale_action(
            _stage_backlog(ctx, k, ctx.depth),
            len(ctx.pools[k]),
            # Reapable only when *fully* idle -- every slot free -- matching
            # `_drive_actor_pool`. An actor with one call running and one slot open is busy.
            sum(1 for a in ctx.pools[k] if idle_slots.get(a, 0) >= ctx.depth),
            ctx.floors[k],
            ctx.ceilings[k],
        )
        if action == "up":
            _grow_stage(ctx, k)


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
    st["inflight"] -= 1
    if not more:
        st["done"] = True
        _maybe_recycle(ctx, prod, st)
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
                _maybe_recycle(ctx, holder, st)
            return
        ctx.outstanding[holder] = max(0, ctx.outstanding.get(holder, 0) - 1)
        parent_record = ctx.morsels.get(parent)
        if parent_record is None or parent_record["pending"] is None:
            return
        parent_record["pending"] -= 1
        if parent_record["pending"] > 0:
            return
        path = parent  # the parent's subtree is done too: settle it in the same loop


def _maybe_recycle(ctx: _Context, prod, st: dict) -> None:
    """Move `prod` on to the next partition once it is drained **and** quiet.

    Drained is `done` with no published morsel still in use above. Quiet is the half the
    queued issue loop adds: a `publish_next` still sitting on the actor's queue will return
    after this, and `_recycle` replaces the producer's state with the next partition's -- so
    a straggler would file its morsel under the wrong `pidx`, against sequence numbers the
    new partition is about to reuse. The window empties on its own, since every queued call
    past the end of a partition returns `False`.
    """
    if st["done"] and st["outstanding"] == 0 and st["inflight"] == 0:
        _recycle(ctx, prod)


def _recycle(ctx: _Context, prod) -> None:
    """A producer whose partition is fully drained takes the next one, or goes idle."""
    ctx.state.pop(prod, None)
    if prod not in ctx.dead:
        ctx.free[0].append(prod)


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
