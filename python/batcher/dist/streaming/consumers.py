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


#: A partition ceiling high enough that `gpu_aware_pool_default`'s "never more actors than
#: partitions" clamp cannot bind while we ask it how many actors the *devices* want. The same
#: sentinel, for the same reason, as `executors.map._UNCLAMPED_PARTITIONS`.
_UNCLAMPED_PARTITIONS = 1 << 30


def consumer_pool_size(gpu_stage, workers: int, num_partitions: int) -> int:
    """Actor count for the GPU consumer stage: its explicit `concurrency`, else one per device.

    **Not clamped to the partition count**, and that is the whole point. A consumer does not
    read a partition: morsels arrive over Flight and `take_consumer` hands each one to whichever
    consumer is free, so the number of consumers is independent of how the *source* happened to
    shard. Clamping them to it inverted the causality — a 2.4 GB corpus takes the four-partition
    floor, and four partitions then decided that eight of a twelve-GPU fleet got no actor at
    all. The data was choosing how many accelerators were allowed to work, which is sublinear
    scaling by construction: adding devices to the cluster changed nothing.

    This is the same fix `executors.map._pool_partition_count` made on the batch path, applied
    to the streaming one, which kept the inherited clamp. Stage 0 is genuinely partition-bound
    and is still sized that way by the driver.

    An explicit `concurrency` is honored as written, including a `(min, max)` range, which the
    caller above resolves as an autoscaling bound rather than a target.

    Args:
        gpu_stage: The consumer stage, carrying its `concurrency` and accelerator ask.
        workers: The worker count the run was sized for, used as the non-accelerator fallback.
        num_partitions: Input partitions — the ceiling on *stage 0*, not on this stage.

    Returns:
        The number of consumer actors to open, at least 1.
    """
    from batcher.dist.executors.map import _resolve_pool_size
    from batcher.ml.gpu import gpu_aware_pool_default

    default = gpu_aware_pool_default(
        gpu_stage.num_gpus,
        workers,
        _UNCLAMPED_PARTITIONS,
        getattr(gpu_stage, "accelerator_type", None),
        resources=dict(getattr(gpu_stage, "resources", ()) or ()),
    )
    size = _resolve_pool_size(gpu_stage.concurrency, num_partitions, default)
    return max(1, size)


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
        num_partitions: Input partitions, the ceiling on stage-0 parallelism — and on stage 0
            only, since a consumer is fed by the Flight hand-off rather than by a partition.

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


#: How long the locality probe waits for a consumer to answer before giving up on locality.
#:
#: The probe is best-effort by design, but an unbounded `ray.get` is not a best-effort call —
#: it is an unbounded one. An actor that cannot be placed never answers, so a pipeline whose
#: consumers were pending behind a full cluster hung *here*, in an optimization, with no error
#: and no timeout: `probe_consumer_hosts` at the top of the driver's stack for as long as the
#: process lived. Thirty seconds is far longer than a live actor's reply (sub-millisecond) and
#: far longer than a cold actor's start, so it never costs locality on a healthy fleet — it
#: only converts a capacity problem into a slower run instead of a silent hang.
_HOST_PROBE_TIMEOUT_S = 30.0


def probe_consumer_hosts(consumers) -> dict:
    """Node host per consumer actor, for locality-aware morsel assignment.

    One fan-out at pool construction (and once per replacement), not per morsel. Returns
    `{}` on any failure *or timeout*: locality is an optimization, so a probe that cannot run
    must leave the pipeline scheduling exactly as it did before rather than fail the query —
    and, equally, must not hold it (see `_HOST_PROBE_TIMEOUT_S`).
    """
    try:
        hosts = ray.get([c.node_host.remote() for c in consumers], timeout=_HOST_PROBE_TIMEOUT_S)
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
