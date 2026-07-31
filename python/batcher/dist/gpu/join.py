"""Run a translated join across every GPU, by splitting the probe side and broadcasting the build.

A join was the one relational shape still pinned to a single device, which is backwards: it is
the shape whose whole premise is that one input is large. The star-schema query it exists for —
a big fact table joined to a small dimension, then aggregated — could not use a second GPU no
matter how many the cluster had.

Splitting the **probe** side and giving every worker the whole **build** side is correct for
the join types whose output is driven by left rows (`plan.distribution.BROADCAST_SAFE_JOINS`).
Each shard emits the rows its own probe slice produces, and unioning them never duplicates a
build row. `right` and `full` must emit an unmatched build row exactly once, and every shard
sees the whole build side, so each would emit it — those keep the single-device path.

The build side is *read* by each worker rather than shipped to it. Reading a small dimension N
times from storage costs less than moving it through the object store N times, and it keeps
the rule that bulk Arrow does not travel as Ray objects. Whether the build side is small enough
for this at all is not decided here: the fan-out runs only when the planner already marked the
join `broadcast`, which is a cost decision Kyber owns and that the CPU join path reads the same
way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.io.source import Source

__all__ = ["sharded_gpu_join"]


def sharded_gpu_join(
    left: Source,
    right: Source,
    left_ops: list[dict],
    right_ops: list[dict],
    join_ir: dict,
    ops: list[dict],
    *,
    gpu_count: int,
    sharded: bool,
    broadcast: bool,
) -> pa.Table | None:
    """Run a translated join across the cluster's GPUs, or `None` when it does not apply.

    Args:
        left: The probe side's source; split across devices.
        right: The build side's source; read whole by every device.
        left_ops: The probe chain's operator IR.
        right_ops: The build chain's operator IR.
        join_ir: The join node's IR.
        ops: The operator chain above the join.
        gpu_count: The cluster's live device count.
        sharded: Whether the working set exceeds one device.
        broadcast: Kyber's verdict that this join's build side is small enough to replicate.

    Returns:
        The join's result, or `None` when the fan-out does not apply — a join type that is not
        broadcast-safe, a build side Kyber will not replicate, a chain above the join with no
        shardable split, an unsplittable probe side, or an unreadable cluster.
    """
    from batcher.plan.distribution import BROADCAST_SAFE_JOINS, ShardSplit, shard_plan

    if join_ir.get("join_type") not in BROADCAST_SAFE_JOINS:
        return None
    if not broadcast:
        # "Is the build side small enough" is Kyber's decision and it has already been made.
        # Re-deriving it here would give the two backends two different answers to one
        # question, and the wrong one is an out-of-memory on every device at once.
        return None
    # The chain ABOVE the join divides the same way any chain does; an empty one merges by
    # plain concatenation.
    above = shard_plan(ops) if ops else ShardSplit([], [], [])
    if above is None:
        return None

    from batcher.dist.gpu.aggregate import shard_descriptors
    from batcher.dist.gpu.dispatch import whole_source_descriptor

    build = whole_source_descriptor(right)
    if build is None:
        return None
    probes = shard_descriptors(left, gpu_count, sharded=sharded, preserve_order=False)
    if probes is None:
        return None

    shards = _run_join_shards(
        probes,
        build,
        left_ops,
        right_ops,
        join_ir,
        above.shard_ops,
        gpu_count=gpu_count,
        probe_schema=_schema_of(left),
        build_bytes=_build_side_bytes(build, right),
    )
    if not shards:
        return None
    from batcher.dist.gpu.aggregate import fold_shards

    return fold_shards(shards, above)


def _schema_of(source):
    """A source's schema, or `None` when it will not describe itself.

    Used only to price the probe shards for packing, so a source that declines costs the join
    its fractional share and nothing else — the tasks then ask for whole devices, as they did.
    """
    try:
        return source.schema()
    except Exception as exc:
        note_suppressed("dist", "read the probe schema for gpu join packing", exc)
        return None


def _build_side_bytes(build: dict, right) -> float:
    """How much device memory the replicated build side occupies in *each* task.

    A broadcast join hands every task the whole build side. Co-tenants on one device therefore
    hold one copy each, not one between them, and a packing decision that charged the build
    side once would put four tasks on a device that is about to hold four copies of it. This
    is the figure that keeps the join from being the one shape where packing OOMs a device.

    Returns:
        Bytes, `0.0` when the build side's size cannot be read — which makes the packing
        under-state the need, so the caller must keep the subdivision ladder behind it.
    """
    from batcher.dist.gpu.resources import descriptor_bytes

    schema = _schema_of(right)
    if schema is None:
        return 0.0
    from batcher.plan.types import schema_row_bytes

    return float(descriptor_bytes(build, schema_row_bytes(schema)))


def _run_join_shards(
    probes: list,
    build: dict,
    left_ops,
    right_ops,
    join_ir: dict,
    above_ops: list[dict],
    *,
    gpu_count: int = 0,
    probe_schema=None,
    build_bytes: float = 0.0,
) -> list:
    """Join every probe shard against the whole build side, recovering from a failed shard.

    Recovery is the aggregate path's, minus the CPU substitute: reconstructing a join shard
    through the engine would need a two-source plan whose second input is this build side, and
    a half-supported fallback is worse than an honest one. A failed shard is subdivided on the
    device, and a shard that still fails abandons the fan-out — the caller then runs the join
    as a single dispatch, or on the CPU engine.
    """
    import ray

    from batcher.carbonite.resilience import gather_with_backups
    from batcher.config import active_config
    from batcher.dist.executors.ray_runtime import speculation_policy
    from batcher.dist.gpu.resources import gpu_shard_options
    from batcher.dist.gpu.shards import ShardReport, is_memory_failure, run_subdivided
    from batcher.dist.gpu.tasks import gpu_join_task, gpu_task_options

    dc = active_config().distributed
    opts, packing = gpu_shard_options(
        probes, probe_schema, gpu_count=gpu_count, resident_bytes=build_bytes
    )
    task = ray.remote(**opts)(gpu_join_task)
    # A probe shard that did not fit its packed share is retried on a whole device: the share is
    # the thing that was just shown to be too small, and a join's retry also carries the whole
    # replicated build side, which is the part of the footprint subdividing the probe cannot
    # shrink. Identical to `task` when nothing was packed.
    retry_task = ray.remote(**gpu_task_options())(gpu_join_task) if packing.packed else task

    report = ShardReport("gpu-join", len(probes), packing=packing)

    def _launch(i: int):
        return task.remote(probes[i], build, left_ops, right_ops, join_ir, above_ops)

    def _on_failure(i: int, _ref, exc):
        if not is_memory_failure(exc) or dc.gpu_shard_subdivide <= 1:
            raise exc
        note_suppressed("dist", f"gpu join shard {i} did not fit; subdividing", exc)
        report.note_subdivided()
        return run_subdivided(
            probes[i],
            lambda d: ray.get(retry_task.remote(d, build, left_ops, right_ops, join_ir, above_ops)),
            parts=int(dc.gpu_shard_subdivide),
            rounds=int(dc.gpu_shard_subdivide_rounds),
            cause=exc,
        )

    refs = [_launch(i) for i in range(len(probes))]
    results = gather_with_backups(refs, _launch, speculation_policy(), on_failure=_on_failure)
    report.publish()
    return [t for t in results if t is not None and t.num_rows]
