"""Run a translated union across every GPU, by sharding each of its inputs.

A union was the last relational shape still pinned to a single device, and it is a strange one
to leave there: a `UNION ALL` is how a partitioned table, a multi-day backfill, or a set of
per-region extracts get read at all, so its inputs are large by construction and there are
several of them. One device had to hold every one of them at once.

The fan-out is the simplest of the three, because a union has no key and nothing to align: a
shard is a contiguous slice of *one* input, it runs that input's own chain, and the slices
reassemble. The reducer above the union then divides the way it does anywhere else.

**`UNION DISTINCT` deliberately keeps the single-device path.** Deduplicating each slice and
then the merge is exact on its own — `DISTINCT` is idempotent and mergeable — but only while
nothing reduces above it. Put an aggregate on top and a row appearing in two different shards
survives both slice-wise dedups and is counted twice, which the merge can no longer see. The
honest fix is a hash shuffle on the whole row, which is the CPU path's job and not a fan-out
this module can fake. `UNION ALL` is the shape that actually grows, so declining costs little
and claiming otherwise would cost a wrong answer.

Order is part of the answer here and is preserved: shards are gathered in `(input, slice)`
order, and each input's slices are contiguous, so the concatenation reproduces what one device
would have produced. That is the same reason `shard_descriptors` is asked for ordered splits
for a row-local chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.io.source import Source

__all__ = ["sharded_gpu_union"]


def sharded_gpu_union(
    sources: list[Source],
    input_ops: list[list[dict]],
    distinct: bool,
    ops: list[dict],
    *,
    gpu_count: int,
    sharded: bool,
) -> pa.Table | None:
    """Run a translated union across the cluster's GPUs, or `None` when it does not apply.

    Args:
        sources: Each union input's source.
        input_ops: Each input's own operator chain, positionally matching `sources`.
        distinct: Whether the union deduplicates. `True` declines the fan-out.
        ops: The operator chain above the union.
        gpu_count: The cluster's live device count.
        sharded: Whether the working set exceeds one device.

    Returns:
        The union's result, or `None` when the fan-out does not apply — a deduplicating union,
        a chain above it with no shardable split, an input that cannot be split, or an
        unreadable cluster — so the caller can use the single-device dispatch instead.
    """
    from batcher.plan.distribution import ShardSplit, shard_plan

    if distinct:
        return None
    # An empty chain above the union merges by plain concatenation, which is ordered.
    above = shard_plan(ops) if ops else ShardSplit([], [], [], ordered=True)
    if above is None:
        return None

    built = _shard_plan_per_input(sources, input_ops, above, gpu_count, sharded=sharded)
    if built is None:
        return None
    plan, shard_bytes = built
    shards = _run_union_shards(plan, shard_bytes=shard_bytes, gpu_count=gpu_count)
    if not shards:
        return None
    from batcher.dist.gpu.aggregate import fold_shards

    return fold_shards(shards, above)


def _shard_plan_per_input(sources, input_ops, above, gpu_count: int, *, sharded: bool):
    """`([(descriptor, ops), ...], largest_shard_bytes)`, or `None` when an input cannot split.

    All-or-nothing: an input that has to be read whole would have to run beside the sharded
    ones as a shard of its own, and its size is exactly why the fan-out was wanted. Falling
    back to the single-device dispatch is the honest answer, and it is what the caller does.

    Each shard carries its *own* input's chain followed by the shared chain above the union, so
    one task body serves every input without knowing which one it is reading.

    The byte figure is measured *here*, where each descriptor is still beside the source it came
    from. One task body runs every input, so one device share is chosen for all of them, and it
    has to be the largest across inputs of possibly very different widths — a narrow input's
    row width would size the share for a payload input's shards and guarantee they do not fit.
    """
    from batcher.dist.gpu.aggregate import shard_descriptors

    # Divide the device budget across the inputs: N inputs each fanning out to `gpu_count`
    # shards would ask for N times the cluster. Ray would pipeline the surplus rather than
    # fail, but each shard would be sized as though it had a device to itself.
    per_input = max(1, gpu_count // max(1, len(sources)))
    plan: list[tuple[dict, list[dict]]] = []
    widest = 0
    for source, chain in zip(sources, input_ops, strict=True):
        descriptors = shard_descriptors(
            source, per_input, sharded=sharded, preserve_order=above.ordered
        )
        if descriptors is None:
            return None
        shard_ops = [*chain, *above.shard_ops]
        plan.extend((descriptor, shard_ops) for descriptor in descriptors)
        widest = max(widest, _input_shard_bytes(source, descriptors))
    return (plan, widest) if plan else None


def _input_shard_bytes(source, descriptors: list[dict]) -> int:
    """The largest shard of one union input, in bytes, or `0` when it cannot be priced.

    A zero here is not a failure: it makes the packing under-state the need for that input,
    which the shared decision then resolves to whole devices rather than to a share the input's
    shards may not fit.
    """
    from batcher.dist.gpu.resources import largest_shard_bytes

    try:
        schema = source.schema()
    except Exception as exc:
        note_suppressed("dist", "read a union input schema for gpu shard packing", exc)
        return 0
    from batcher.plan.types import schema_row_bytes

    return largest_shard_bytes(descriptors, schema_row_bytes(schema))


def _run_union_shards(
    plan: list[tuple[dict, list[dict]]], *, shard_bytes: int = 0, gpu_count: int = 0
) -> list:
    """Run every shard on a device, recovering a failed one rather than the query.

    The same two-rung ladder the chain fan-out uses, and for the same reasons: a shard that did
    not *fit* is subdivided and run on the device in pieces, which is exact because the pieces
    of a slice concatenate back into the slice; anything else is recomputed by the native CPU
    engine, which produces the identical rows for the identical chain.

    Results are returned in shard order, which for a union is `(input, slice)` order — and that
    order is the answer, not a convenience.
    """
    import ray

    from batcher.carbonite.resilience import gather_with_backups
    from batcher.config import active_config
    from batcher.dist.executors.ray_runtime import engine_config_json, speculation_policy
    from batcher.dist.gpu.aggregate import _await_recoveries, _Recovering
    from batcher.dist.gpu.resources import share_for_bytes
    from batcher.dist.gpu.shards import ShardReport, is_memory_failure, run_subdivided
    from batcher.dist.gpu.tasks import (
        cpu_shard_partial,
        gpu_shard_partial,
        gpu_task_options,
    )

    dc = active_config().distributed
    cfg_json = engine_config_json()
    packing = share_for_bytes(shard_bytes, len(plan), gpu_count=gpu_count)
    gpu_task = ray.remote(**gpu_task_options(num_gpus=packing.fraction))(gpu_shard_partial)
    # A slice that did not fit its packed share is retried on a whole device rather than divided
    # against the share that was just shown to be too small. Identical when nothing was packed.
    retry_task = ray.remote(**gpu_task_options())(gpu_shard_partial) if packing.packed else gpu_task
    cpu_task = ray.remote(max_retries=int(dc.task_max_retries))(cpu_shard_partial)

    report = ShardReport("gpu-union", len(plan), packing=packing)

    def _launch(i: int):
        descriptor, shard_ops = plan[i]
        return gpu_task.remote(descriptor, shard_ops)

    def _on_failure(i: int, _ref, exc):
        descriptor, shard_ops = plan[i]
        if is_memory_failure(exc) and dc.gpu_shard_subdivide > 1:
            try:
                note_suppressed("dist", f"gpu union shard {i} did not fit; subdividing", exc)
                report.note_subdivided()
                return run_subdivided(
                    descriptor,
                    lambda d: ray.get(retry_task.remote(d, shard_ops)),
                    parts=int(dc.gpu_shard_subdivide),
                    rounds=int(dc.gpu_shard_subdivide_rounds),
                    cause=exc,
                )
            except Exception as sub_exc:
                exc = sub_exc
        if not dc.gpu_shard_cpu_fallback:
            raise exc
        note_suppressed("dist", f"gpu union shard {i}; recomputing on the CPU engine", exc)
        report.note_recovered()
        # Wrapped so the barrier does not block on it: every recovery is already running by
        # the time the last one is awaited.
        return _Recovering(cpu_task.remote(descriptor, shard_ops, cfg_json))

    refs = [_launch(i) for i in range(len(plan))]
    results = gather_with_backups(refs, _launch, speculation_policy(), on_failure=_on_failure)
    results = _await_recoveries(results)
    report.publish()
    return [t for t in results if t is not None and t.num_rows]
