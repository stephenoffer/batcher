"""Build the stage pools of a streaming inference pipeline and run one query through them.

`dist/executors/map.py` distributes a linear `map_batches` chain *embarrassingly* — one actor
runs the whole CPU→GPU chain per partition, so the GPU sits idle while its actor reads and
preprocesses. This package is the distributed image of the single-node `ml/pipeline.py`: it
splits the chain into stages **by resource class**, gives each its own actor pool, and streams
partitions stage→stage so the stages **overlap** — while a model runs morsel *k*, the stage
below prepares *k+1*.

The split is at *every* resource boundary, not one. A single cut put everything above the
first model into one actor, so two chained models shared a device and took turns, and a CPU
postprocess ran on the GPU actor — spending device time on host work and forcing the two to
scale together. `split_into_resource_stages` cuts at each boundary and each piece gets a pool
sized on its own terms: a GPU stage from the fleet's devices and its measured utilization, a
host stage from the worker count.

The hand-off is Carbonite Arrow Flight, not the Ray object store: each stage PUBLISHES its
output **one morsel at a time** on its node-local `ShuffleSession` and returns only a small
`(addr, ticket)`; the stage above FETCHES it in place. The result equals running the stages in
sequence — every stage runs the identical sub-plan through `core.execute_with_udfs` — so only
the scheduling overlaps. `schedule` owns that scheduling and the credit windows that bound it.
"""

from __future__ import annotations

import contextlib
import dataclasses

import pyarrow as pa
import ray

from batcher._internal.logging import note_suppressed
from batcher._internal.mathx import clamp
from batcher.config import active_config
from batcher.dist.executors.map import release_inference_pools
from batcher.dist.executors.ray_runtime import reset_scheduling_envelope
from batcher.dist.streaming.consumers import consumer_pool_bounds, record_consumer_feedback
from batcher.dist.streaming.pipeline.schedule import run_streamed
from batcher.dist.streaming.producers import ProducerActor, consumer_batch_rows
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan

__all__ = ["stream_distributed_pipeline"]


def stream_distributed_pipeline(
    plan: LogicalPlan, sources: list[Source], workers: int, hub=None
) -> pa.Table:
    """Run a linear map pipeline as N overlapped, credit-bounded stages.

    Each stage's pool runs only that stage and streams its output to the next over Flight, so
    a model is fed by the stage below instead of waiting for it, and every stage's resident
    output stays bounded by the production credit window regardless of partition size. The
    result is identical to the single-node sequential composition (and to the non-overlapped
    distributed map); only the scheduling overlaps. The caller guarantees the plan divides
    (the dispatch hook checks `split_into_resource_stages`); other shapes use `_distributed_map`.

    Args:
        plan: The linear `Scan → map → … → map` plan to run.
        sources: The plan's bound sources.
        workers: The worker count the caller sized the run for.
        hub: The metadata hub each pool's measured utilization is recorded into, or `None`.

    Returns:
        The pipeline's rows as one Arrow table.
    """
    from batcher.dist.executors.partition_io import partition_descriptors
    from batcher.dist.executors.plan_analysis import split_into_resource_stages
    from batcher.dist.executors.ray_runtime import _ensure_ray
    from batcher.dist.flight_worker import new_plan_id
    from batcher.plan.visitor import scanned_source_ids

    _ensure_ray(workers)
    stages = split_into_resource_stages(plan, fold_leading_scan=fold_leading_scan(plan, workers))
    sid = next(iter(scanned_source_ids(plan)))
    partitions = partition_descriptors(sources[sid], producer_fanout(stages[0], workers))
    if not partitions:
        return _empty(plan)

    credits = max(1, active_config().flow_control.default_credits)
    # Release the session-warm inference pools first. They belong to the *non-overlapped* map
    # path (`_resident_pool_for`), which this pipeline does not use — it gives every stage its
    # own pool — so across a streamed run they are a whole-cluster reservation serving nothing.
    #
    # Leaving them was a deadlock, not an inefficiency, and the identical one
    # `_evict_stale_configurations` already documents for its own case: a preceding
    # `collect(distributed=True)` on the non-streamed path leaves three actors holding the
    # machine's 96 cores, this pipeline's consumers pend behind them forever, and the driver
    # waits on actors that can never start. The next non-streamed inference `collect()`
    # rebuilds the pool, paying the model load once — which is the documented cost of
    # `release_inference_pools`, and cheaper than a query that never returns.
    release_inference_pools()
    bounds = _pool_bounds(stages, workers, len(partitions))
    # Every stage's pool is resident at the same time — that is what "overlapped" means — so
    # the grant Carbonite sized for ONE fleet of `workers` actors has to be divided across
    # all of them. Re-wrapping the actor classes under the divided envelope is what makes
    # the second pool placeable at all; see `_overlapped_envelope`.
    token = _narrow_for_overlap(workers, bounds)
    try:
        _ensure_ray(workers)  # re-wraps the actor classes under the divided grant
        pools, spawns, ceilings = _build_pools(stages, bounds, credits)
    finally:
        if token is not None:
            reset_scheduling_envelope(token)
    alive: set = {actor for pool in pools for actor in pool}
    try:
        results = run_streamed(
            pools,
            partitions,
            new_plan_id(),
            credits,
            spawn=spawns,
            alive=alive,
            ceilings=ceilings,
        )
        # Every pool stage measured its own utilization; record it so the next run's `num_gpus`
        # request adapts. Recording only the last stage's, as the two-stage form did, leaves a
        # model in the middle of the chain invisible to the sizing that is supposed to feed it.
        for pool, stage in zip(pools[1:], stages[1:], strict=True):
            if stage.num_gpus > 0:
                record_consumer_feedback(pool, plan, hub)
    finally:
        for actor in alive:
            with contextlib.suppress(Exception):
                ray.kill(actor)

    # Concatenate morsels in path order — a valid grouping of the input multiset (the result
    # is an unordered relation; callers that need order sort).
    batches: list[pa.RecordBatch] = []
    for _path, out in sorted(results.items()):
        if out:
            batches.extend(out)
    return pa.Table.from_batches(batches) if batches else _empty(plan)


def fold_leading_scan(plan: LogicalPlan, workers: int) -> bool:
    """Whether a scan with no CPU map above it should run *inside* the accelerator stage.

    Folding is right whenever the read has nowhere better to run: on a homogeneous fleet the
    hop over Flight buys nothing and costs a serialization of the raw partition, which is what
    `split_into_resource_stages` documents.

    It is wrong on a heterogeneous one, and by a margin that is easy to miss because nothing
    fails. Folded, the Parquet decode runs inside the GPU actor — on the accelerator node's own
    cores, in the accelerator node's own memory, the two resources that stage needs to keep its
    device fed — while every accelerator-free node in the cluster does nothing at all. Sampled
    per node through an image-inference run on an 8-GPU / 9-CPU-node fleet: **CPU nodes 2.1%
    busy, GPU nodes 24.6% CPU and 36.9% device**.

    So: split the scan off exactly when the fleet has accelerator-free nodes with the cores to
    host it (`cpu_only_can_host`, which is already `False` on a homogeneous or accelerator-less
    cluster). Everywhere else this returns `True` and nothing changes.

    Args:
        plan: The pipeline being split, used only to see whether it asks for an accelerator.
        workers: The worker count the run is sized for.

    Returns:
        `True` to fold the scan into the stage above it, `False` to give it its own stage.
    """
    from batcher.plan.logical import MapBatches
    from batcher.plan.visitor import walk

    if not any(getattr(n, "num_gpus", 0.0) > 0 for n in walk(plan) if isinstance(n, MapBatches)):
        return True  # no accelerator stage: there is no node class to keep the read off
    if not _accelerator_stage(_stage_above_scan(plan)):
        # The read's neighbour is a *host* stage, so folding already puts it on the CPU nodes
        # -- and splitting it off buys nothing while costing a whole Flight hop for every row
        # and a read pool sized like the device fleet. Measured on a
        # `scan -> map(concurrency=64) -> map(num_gpus=1)` pipeline over 256 shards, which is
        # the ordinary two-stage inference shape: split, the stages come out **[8, 64, 8]** --
        # eight actors reading 256 shards and republishing to sixty-four that could have read
        # them directly. The argument for splitting is about which *nodes* run the read, and it
        # is answered here without the hop.
        return True
    try:
        from batcher.dist.executors.ray_runtime.scaling import cpu_only_can_host

        return not cpu_only_can_host(max(1, workers), active_config().execution.cpu_share_io)
    except Exception as exc:  # pragma: no cover - a placement courtesy, never a failure
        note_suppressed("dist", "read the fleet shape before splitting the leading scan", exc)
        return True


def _stage_above_scan(plan: LogicalPlan):
    """The bottom-most `MapBatches` of a linear chain -- the stage a leading scan feeds."""
    from batcher.plan.logical import MapBatches, Scan

    node, lowest = plan, None
    while isinstance(node, MapBatches):
        lowest, node = node, node.input
    return lowest if isinstance(node, Scan) else None


def _accelerator_stage(node) -> bool:
    """Whether `node` asks for a device -- a GPU, a named accelerator, or a custom resource."""
    if node is None:
        return True  # unknown shape: keep the conservative split
    return bool(
        getattr(node, "num_gpus", 0.0) > 0
        or getattr(node, "accelerator_type", None)
        or getattr(node, "resources", None)
    )


def producer_fanout(stage, workers: int) -> int:
    """How wide stage 0 may open: its explicit `concurrency` when it has one, else `workers`.

    Stage 0 reads partitions, so its pool size and the partition count are the same number —
    which is why both are taken from here, and why an explicit `concurrency` has to reach the
    *partitioning* and not only the pool. It did not: the fan-out was `workers`, the relational
    fleet width, and `map_batches(..., concurrency=N)` on the host stage of a streamed pipeline
    was silently discarded. A caller asking for forty-eight decode actors got sixteen.

    This is the fix `executors.map._pool_partition_count` already made on the non-streamed
    path, in its own words: "`_drive_actor_pool` then clamps the pool to
    `min(max_size, len(partitions))` and the caller's number is silently reduced to the worker
    count". The streaming path inherited the shape and not the fix, so the same public argument
    meant two different things depending on which path a plan happened to take.

    Args:
        stage: The first resource stage, carrying its `concurrency` spec.
        workers: The worker count the run was sized for.

    Returns:
        The actor and partition count for stage 0, at least 1.
    """
    from batcher.dist.executors.map import _explicit_pool_ceiling

    explicit = _explicit_pool_ceiling(getattr(stage, "concurrency", None))
    return max(1, explicit) if explicit else workers


def _pool_bounds(stages, workers: int, num_partitions: int) -> list[tuple[int, int]]:
    """`(start, ceiling)` per stage: how many actors it opens with and may grow to.

    Stage 0 is bounded by the partitions it opens, so it has nothing to grow into: its start
    and ceiling are the same number and the scheduler leaves it alone. Every stage above is
    fed by the Flight hand-off rather than by a partition, so its bounds come from its own
    `concurrency` spec.
    """
    producers = clamp(num_partitions, 1, producer_fanout(stages[0], workers))
    bounds = [(producers, producers)]
    bounds.extend(consumer_pool_bounds(stage, workers, num_partitions) for stage in stages[1:])
    return bounds


def _build_pools(stages, bounds, credits: int):
    """One actor pool per stage, each sized and placed on its own terms.

    Returns `(pools, spawns, ceilings)`: the live pools, a factory per stage that mints a
    replacement for a preempted actor **or an extra actor when a stage falls behind**, and
    the count each stage may grow to. Every stage's morsel size comes from the stage *above*
    it, because a published morsel is one call on that stage — publishing at the engine's own
    granularity hands a model whatever the scan happened to emit, which for wide rows is a
    handful.

    A stage whose `concurrency` is a plain int (or absent) gets `ceiling == len(pool)`, so
    nothing about its scheduling changes; only a `(min, max)` spec, which the public API
    documents as autoscaling, opens smaller and grows.
    """
    # Imported AFTER `_ensure_ray` (the caller's first act) so these are the Ray-remote-wrapped
    # classes: `ray_runtime._wrap_tasks` rebinds them with the ambient scheduling grant.
    from batcher.dist.executors.map import _gpu_options, _MapActor
    from batcher.dist.streaming.relay import RelayActor

    last = len(stages) - 1
    spawns = []
    for k, stage in enumerate(stages):
        target_rows = consumer_batch_rows(stages[k + 1].sub_plan) if k < last else 0
        if k == 0:
            spawns.append(
                _producer_factory(stage, credits, target_rows, _producer_cpu_width(bounds[0][1]))
            )
            continue
        cls = _MapActor if k == last else RelayActor
        spawns.append(
            _actor_factory(
                cls,
                stage,
                credits,
                target_rows,
                terminal=k == last,
                resources=_gpu_options(stage.num_gpus, stage.accelerator_type),
            )
        )
    pools = [
        [spawn() for _ in range(start)] for spawn, (start, _hi) in zip(spawns, bounds, strict=True)
    ]
    return pools, spawns, [hi for _lo, hi in bounds]


def _narrow_for_overlap(workers: int, bounds) -> object | None:
    """Divide the ambient per-actor grant across the stages that are resident at once.

    Carbonite sizes `SchedulingEnvelope` for a *single* fleet: `num_cpus` is the machine's
    cores divided by `workers`, so `workers` actors fill the cluster exactly. An overlapped
    pipeline runs several pools at once and every one of them was taking that same whole-fleet
    grant, so a two-stage run asked for twice the machine and a three-stage run three times it.

    On a 96-core box that is not a slowdown, it is a **deadlock**: `workers=3` grants 32 cores
    an actor, the three producers take all 96, and the three consumers pend forever behind
    them — `Pending Demands: {'CPU': 32.0}: 3+` against `96.0/96.0 CPU`, with the driver
    blocked in `probe_consumer_hosts` waiting on actors that can never start. Nothing times
    out, so the query hangs rather than failing.

    The rescale is the obvious one: a non-overlapped run places `workers` actors of
    `env.num_cpus`, so the budget is their product; spread that over the actors this pipeline
    will actually hold. Sized against each stage's **ceiling**, not its opening count, so a
    stage that autoscales into its documented maximum cannot re-create the deadlock later.
    Memory divides with it, for the same reason and by the same factor.

    Args:
        workers: The worker count the run was sized for — the fleet width the grant assumed.
        bounds: `(start, ceiling)` per stage, from `_pool_bounds`.

    Returns:
        The context token to `reset_scheduling_envelope` with, or `None` when there is no
        ambient envelope to narrow or the pipeline is no wider than a single fleet.
    """
    from batcher.dist.executors.ray_runtime import current_envelope, set_scheduling_envelope

    env = current_envelope()
    if env is None:
        return None
    resident = sum(max(1, hi) for _lo, hi in bounds)
    if resident <= workers:
        return None  # no wider than the fleet the grant was sized for; leave it alone
    share = workers / resident
    # Floored at a tenth of a core rather than at one: on a small cluster a fair share is
    # genuinely fractional, and Ray schedules fractional CPUs. Flooring at 1.0 would restore
    # the over-subscription this exists to remove.
    narrowed = dataclasses.replace(
        env,
        num_cpus=max(0.1, env.num_cpus * share),
        memory_bytes=int(env.memory_bytes * share),
    )
    return set_scheduling_envelope(narrowed)


def _shipping_options() -> dict:
    """`.options(...)` that make `import batcher` work in a streaming stage's actor.

    Only the `runtime_env`, deliberately: the resource grant for these actors is decided by
    the caller's envelope narrowing above, and `task_options` would also rewrite `num_cpus`.

    The streaming actors are declared with a bare `@ray.remote` and created with a bare
    `.remote(...)`, so unlike the Flight fleet (which goes through `fleet_actor_options` ->
    `task_options`) nothing attached the package. On a job where a **foreign** `ray.init` ran
    first — `tests/_ray_cluster.init_test_ray`, and any user who attaches to their own cluster
    — the actor died in its creation task with
    `RaySystemError: System error: No module named 'batcher'`, taking every distributed
    streaming test with it. `worker_runtime_env()` returns `None` when Batcher initialized Ray
    itself, which is why this was invisible on the default path.
    """
    from batcher.dist.executors.ray_runtime.scheduling import worker_runtime_env

    env = worker_runtime_env()
    return {"runtime_env": env} if env else {}


def _producer_factory(stage, credits: int, target_rows: int, cpu_workers: int):
    def spawn():
        opts = _shipping_options()
        cls = ProducerActor.options(**opts) if opts else ProducerActor
        return cls.remote(stage.sub_plan, credits, target_rows, cpu_workers)

    return spawn


def _producer_cpu_width(producers: int) -> int:
    """Threads each stage-0 producer runs its sub-plan with: its share of the fleet's cores.

    The producer pool is the host stage's whole parallelism, so between them its actors should
    hold the cluster -- `cluster cores / pool size`, capped by the smallest alive node because
    an actor's threads all run in one process on one node. That is `_agg_actor_width`, which a
    CPU map/aggregate pool already sizes itself with, and the argument carries over unchanged.

    It has to be said here because the producer had no such lever at all. Every other pool in
    the engine sets one -- a device actor gets `_INFERENCE_CPU_WORKERS`, a CPU aggregate pool
    gets `_agg_actor_width`, a stateless task gets its own CPU share -- and the streamed host
    stage inherited none of them, so it decoded on one thread per actor. On a fleet with eight
    times more cores than producers that is seven eighths of the machine left idle, and it is
    the reason a staged image pipeline never cleared 17% cluster CPU.

    **Whether it helps is a property of the user's function, not of the fleet**, and nothing
    here can see which. Threads share an interpreter, so a stage that releases the GIL scales
    and one that does not gets *slower* the wider it goes. Measured on this box, one stage over
    eight batches, 1 thread against 2/4/8: a PIL JPEG decode went 1.93x / 3.58x / 4.12x, and a
    NumPy `sort`/`tanh`/`sqrt` pipeline went 1.56x / 0.94x / **0.69x**. Both are ordinary
    preprocess stages. The fleet share is the right default because these pipelines exist for
    the decode/tokenize shape, which is the first row; a stage in the second row should be
    given more *actors* and fewer threads (`concurrency`), which is the lever the user has and
    this function deliberately does not override.
    """
    from batcher.dist.executors.map import _agg_actor_width

    return _agg_actor_width(max(1, producers))


def _actor_factory(
    cls, stage, credits: int, target_rows: int, *, terminal: bool, resources: dict | None = None
):
    """A factory minting one actor of `cls` for `stage`, under `resources` plus the shipping env.

    Both option fragments are applied in **one** `.options(...)` call, on the raw remote class.
    Applying them in two — the accelerator request when the pool was built, the `runtime_env`
    when an actor was spawned — raised `AttributeError: 'ActorOptionWrapper' object has no
    attribute 'options'`, because Ray's `.options()` returns a thin wrapper exposing only
    `remote`/`bind`. It needed both fragments to be non-empty to fire, so it was invisible
    until a GPU stage ran on a cluster the user had attached to themselves: `_shipping_options`
    is empty whenever Batcher started Ray, and `resources` is empty for a CPU-only chain. That
    is every stage-overlapped CPU->GPU inference pipeline on a real cluster.

    Args:
        cls: The Ray-remote actor class for this stage.
        stage: The resource stage this actor runs.
        credits: The Flight production credit window a relay takes.
        target_rows: Morsel width, taken from the stage above.
        terminal: Whether this is the last stage (returns rows instead of republishing).
        resources: The accelerator `.options(...)` fragment, or `None` for a host stage.

    Returns:
        A zero-argument callable that spawns one actor.
    """

    def spawn():
        opts = {**(resources or {}), **_shipping_options()}
        bound = cls.options(**opts) if opts else cls
        # A terminal consumer returns its rows to the driver, so it runs no Flight server and
        # takes no credit window; a relay republishes and takes both.
        return (
            bound.remote(stage.sub_plan)
            if terminal
            else bound.remote(stage.sub_plan, credits, target_rows)
        )

    return spawn


def _empty(plan: LogicalPlan) -> pa.Table:
    """A typed empty result for this plan.

    `pa.table({})` returns a table with *no columns at all*, so an empty distributed pipeline
    disagreed with the single-node run it is supposed to be identical to on the result schema
    — no names, no types. A caller that concatenates it against a non-empty run then fails on
    a schema mismatch, and one that inspects `.schema` silently sees an empty relation where
    it should see a typed one.
    """
    from batcher.dist.executors.plan_analysis import empty_result_table

    return empty_result_table(plan, plan.available_columns())
