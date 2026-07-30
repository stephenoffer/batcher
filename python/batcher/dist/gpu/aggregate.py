"""Run a translated GPU chain ending in an aggregate across every GPU in the cluster.

The single-device GPU path is bounded by one device's memory, which is the wrong ceiling for
the workloads a GPU is worth using for. This shards the chain instead: each device reads its
own slice of the source, runs the whole chain plus the *partial* stage of the mergeable
decomposition, and the small per-group partials are folded once at the end. Per-device memory
is then a function of the shard count rather than of the input, so the same query runs on data
many times larger than any one device.

Two properties make this safe rather than merely faster. The decomposition is expressed in the
plan IR (`core.gpu_plan.mergeable`), so partial and combine run through the same translator
every other operator does — the multi-device answer equals the single-device one by
construction. And a shard that cannot run on a device is recomputed by the **native CPU
engine** on a CPU worker, which produces the identical partial; losing a device costs that
shard's time and nothing else, where the older path abandoned the accelerated run entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed
from batcher.config import active_config

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.io.source import Source

__all__ = ["sharded_gpu_aggregate"]


def sharded_gpu_aggregate(
    source: Source, ops: list[dict], *, gpu_count: int, sharded: bool
) -> pa.Table | None:
    """Run a translated chain across the GPUs, reducing per device and folding once.

    Args:
        source: The scan's source; must be splittable for the fan-out to be worth it.
        ops: The bottom-up operator IR chain. Its reducing prefix runs per device; anything
            above the reducer runs once on the folded result.
        gpu_count: The cluster's live device count.
        sharded: Whether the working set exceeds one device, so the chain must fan out.
            `False` still runs on a worker (which reads the source itself) but as one shard.

    Returns:
        The chain's result, or `None` when the fan-out does not apply — an unsplittable source,
        no shardable reducer, an unreadable cluster — so the caller can use the single-device
        dispatch or the CPU engine instead.
    """
    from batcher.plan.distribution import shard_plan

    split = shard_plan(ops)
    if split is None:
        return None
    shard_ops, merge_ops, tail_ops = split

    descriptors = _shard_descriptors(source, gpu_count, sharded=sharded)
    if descriptors is None:
        return None
    partials = _run_shards(descriptors, shard_ops)
    if not partials:
        return None
    return _merge(partials, [*merge_ops, *tail_ops])


def _shard_descriptors(source: Source, gpu_count: int, *, sharded: bool):
    """One partition descriptor per shard, or `None` when the source cannot be fanned out.

    A shard reads itself from storage, so the driver never materializes the source to hand it
    out. An in-memory source has no splits to describe, and is left to the caller's
    ship-the-table path.
    """
    try:
        import ray

        from batcher.dist.executors.partition_io import (
            WholeSourceSplit,
            _scan_splits,
            partition_descriptors,
        )
        from batcher.dist.executors.ray_runtime import _ensure_ray
    except Exception as exc:
        note_suppressed("dist", "import the GPU fan-out dependencies", exc)
        return None
    if not ray.is_initialized() or gpu_count < 1:
        return None
    splits = _scan_splits(source, gpu_count)
    if len(splits) == 1 and isinstance(splits[0], WholeSourceSplit):
        return None
    if sharded:
        # Oversubscribe past the device count: each shard is then bounded (no single-device
        # OOM on a large source), work load-balances finely across a heterogeneous fleet, and
        # a preempted shard's retry is 1/N of the work. Ray runs at most `gpu_count`
        # single-device tasks at once, so the surplus pipelines behind them.
        factor = max(1, int(active_config().distributed.gpu_shard_oversubscribe))
        n_shards = gpu_count * factor
    else:
        n_shards = 1
    _ensure_ray(gpu_count)
    return partition_descriptors(source, n_shards)


def _run_shards(descriptors: list, shard_ops: list[dict]) -> list:
    """Reduce every shard on a device, substituting the CPU engine for any that fails.

    Uses the same straggler-backup barrier the CPU shuffle does: a shard is a pure function of
    its descriptor, so a duplicate of a slow one is safe and the barrier keeps whichever copy
    lands first. `on_failure` is what makes a *lost* device local — it recomputes that shard's
    identical partial through the engine rather than failing the gather.
    """
    import ray

    from batcher.carbonite.resilience import gather_with_backups
    from batcher.dist.executors.ray_runtime import engine_config_json, speculation_policy
    from batcher.dist.gpu.tasks import cpu_shard_partial, gpu_shard_partial, gpu_task_options

    cfg_json = engine_config_json()
    gpu_task = ray.remote(**gpu_task_options())(gpu_shard_partial)
    cpu_task = ray.remote(max_retries=int(active_config().distributed.task_max_retries))(
        cpu_shard_partial
    )

    def _launch(i: int):
        return gpu_task.remote(descriptors[i], shard_ops)

    def _on_failure(i: int, _ref, exc):
        if not active_config().distributed.gpu_shard_cpu_fallback:
            raise exc
        note_suppressed("dist", f"gpu shard {i}; recomputing on the CPU engine", exc)
        return ray.get(cpu_task.remote(descriptors[i], shard_ops, cfg_json))

    refs = [_launch(i) for i in range(len(descriptors))]
    results = gather_with_backups(refs, _launch, speculation_policy(), on_failure=_on_failure)
    return [t for t in results if t is not None and t.num_rows]


def _merge(partials: list, ops: list[dict]) -> pa.Table:
    """Fold the shards' results into the answer, then run whatever sat above the reducer.

    Runs on the host through the translator's own kernels, over one row per group (or per
    distinct row, or per top-N entry) per shard — small by construction, which is the whole
    point of reducing before merging. Using the same kernels as the device keeps both halves
    of the algebra in one implementation.
    """
    import pandas as pd
    import pyarrow as pa

    from batcher.core.gpu_plan import DfBackend
    from batcher.core.gpu_plan.execute import run_chain

    be = DfBackend(pd)
    return be.to_arrow(run_chain(pa.concat_tables(partials), ops, be))
