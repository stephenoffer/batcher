"""The GPU consumer pool of the streaming pipeline: how many, and which one next.

`pipeline` owns the overlap loop — producers publish morsels, consumers run inference on
them. This module owns the pool around it: sizing it to the cluster's GPUs, feeding each
run's measured utilization back for the next one, and choosing *which* free consumer takes
the next morsel. That last choice is a locality decision: a morsel handed to a consumer on
the producing node is fetched over loopback rather than the cluster network, and a random
consumer is remote with probability `1 - 1/nodes`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import ray

from batcher._internal.mathx import clamp

if TYPE_CHECKING:
    from batcher.plan.logical import LogicalPlan

__all__ = [
    "consumer_pool_bounds",
    "consumer_pool_size",
    "probe_consumer_hosts",
    "record_consumer_feedback",
    "take_consumer",
]


def consumer_pool_size(gpu_stage, workers: int, num_partitions: int) -> int:
    """Actor count for the GPU consumer stage: its explicit `concurrency`, else a
    GPU-aware default (one actor per GPU), clamped to the partition count."""
    from batcher.dist.executors.map import _resolve_pool_size
    from batcher.ml.gpu import gpu_aware_pool_default

    default = gpu_aware_pool_default(
        gpu_stage.num_gpus,
        workers,
        num_partitions,
        getattr(gpu_stage, "accelerator_type", None),
        resources=dict(getattr(gpu_stage, "resources", ()) or ()),
    )
    size = _resolve_pool_size(gpu_stage.concurrency, num_partitions, default)
    return clamp(num_partitions, 1, size)


def consumer_pool_bounds(stage, workers: int, num_partitions: int) -> tuple[int, int]:
    """`(start, ceiling)` actor counts for a stage — what it opens with and may grow to.

    `concurrency=(min, max)` is documented as "the pool autoscales to the backlog", and on
    the `_drive_actor_pool` path it does: the pool opens at `min` and grows toward `max`
    while work queues. The streaming pipeline resolved the same spec *statically*, to the
    partition count clamped into the range, so the identical public argument meant two
    different things depending on which path a plan happened to take — and on this one a
    starved stage stayed starved for the whole query.

    A plain int or an absent spec keeps today's behaviour exactly: `start == ceiling`, which
    the scheduler reads as "do not scale". So only the spelling that *asks* for autoscaling
    changes, and the pool it opens with is the `min` the user wrote rather than a size
    derived from a partition count they never mentioned.

    This is also the semantics Ray Data gives the same argument, which matters because users
    arrive here with that expectation: "the `concurrency` parameter only imposes *limits* on
    how many tasks/actors can run; actual scheduling is governed by `num_cpus` and `num_gpus`"
    (`../optimization-guides`, `foundations/data/ray-data-optimization.md`). A range is a
    bound the pool grows within as demand and capacity allow — not a target computed from the
    input's shape.

    Args:
        stage: The resource stage, carrying its `concurrency` spec and accelerator ask.
        workers: The worker count the run was sized for.
        num_partitions: Input partitions, the ceiling on stage-0 parallelism.

    Returns:
        `(start, ceiling)`, both at least 1 and with `ceiling >= start`.
    """
    spec = getattr(stage, "concurrency", None)
    fixed = consumer_pool_size(stage, workers, num_partitions)
    if not isinstance(spec, tuple):
        return fixed, fixed
    lo, hi = int(spec[0]), int(spec[1])
    ceiling = max(1, hi)
    return clamp(lo, 1, ceiling), ceiling


def record_consumer_feedback(consumers, plan: LogicalPlan, hub) -> None:
    """Persist the GPU consumers' peak utilization for next-run `num_gpus` adaptation
    (a no-op when `hub` is None or no GPU was observed)."""
    from batcher.dist.executors.map import _record_gpu_feedback

    samples = [s for s in ray.get([c.gpu_stats.remote() for c in consumers]) if s is not None]
    _record_gpu_feedback(hub, plan, max(samples) if samples else None)


def probe_consumer_hosts(consumers) -> dict:
    """Node host per consumer actor, for locality-aware morsel assignment.

    One fan-out at pool construction (and once per replacement), not per morsel. Returns
    `{}` on any failure: locality is an optimization, so a probe that cannot run must
    leave the pipeline scheduling exactly as it did before rather than fail the query.
    """
    try:
        hosts = ray.get([c.node_host.remote() for c in consumers])
    except Exception as exc:  # pragma: no cover - best-effort probe
        from batcher._internal.logging import note_suppressed

        note_suppressed("dist", "probe streaming consumer hosts", exc)
        return {}
    return {c: h for c, h in zip(consumers, hosts, strict=False) if h}


def take_consumer(free_consumers, consumer_hosts: dict, addr: str):
    """Pop a free consumer, preferring one on the node that produced this morsel.

    A same-node hand-off keeps the Flight fetch on loopback instead of the cluster
    network — the difference that grows with the fleet, since a random consumer is remote
    with probability `1 - 1/nodes`. Strictly best-effort: it only ever reorders *already
    free* consumers, so a morsel never waits for a local one to come back. That matters
    more than the locality does; idling a GPU to save a hop is a net loss.
    """
    if consumer_hosts:
        from batcher.carbonite.transfer.lifecycle import host_of

        host = host_of(addr)
        for c in free_consumers:
            if consumer_hosts.get(c) == host:
                free_consumers.remove(c)
                return c
    return free_consumers.popleft()
