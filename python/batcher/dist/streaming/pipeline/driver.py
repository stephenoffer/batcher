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
    stages = split_into_resource_stages(plan)
    sid = next(iter(scanned_source_ids(plan)))
    partitions = partition_descriptors(sources[sid], workers)
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


def _pool_bounds(stages, workers: int, num_partitions: int) -> list[tuple[int, int]]:
    """`(start, ceiling)` per stage: how many actors it opens with and may grow to.

    Stage 0 is bounded by the partitions it opens, so it has nothing to grow into: its start
    and ceiling are the same number and the scheduler leaves it alone. Every stage above is
    fed by the Flight hand-off rather than by a partition, so its bounds come from its own
    `concurrency` spec.
    """
    producers = clamp(num_partitions, 1, workers)
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
            spawns.append(_producer_factory(stage, credits, target_rows))
            continue
        opts = _gpu_options(stage.num_gpus, stage.accelerator_type)
        cls = (
            (_MapActor if k == last else RelayActor).options(**opts)
            if opts
            else (_MapActor if k == last else RelayActor)
        )
        spawns.append(_actor_factory(cls, stage, credits, target_rows, terminal=k == last))
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


def _producer_factory(stage, credits: int, target_rows: int):
    def spawn():
        return ProducerActor.remote(stage.sub_plan, credits, target_rows)

    return spawn


def _actor_factory(cls, stage, credits: int, target_rows: int, *, terminal: bool):
    def spawn():
        # A terminal consumer returns its rows to the driver, so it runs no Flight server and
        # takes no credit window; a relay republishes and takes both.
        return (
            cls.remote(stage.sub_plan)
            if terminal
            else cls.remote(stage.sub_plan, credits, target_rows)
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
