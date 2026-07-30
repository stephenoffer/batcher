"""The single-key group-by fan-out that predates the plan translator.

`core.gpu_plan` translates whole chains and `dist.gpu.aggregate` fans any of them across
devices, which is a strict superset of the shape here — a single-key group-by aggregate
directly over a scan. This path is kept for one reason the general one cannot cover: its kernel
(`core.gpu_transform.gpu_groupby_agg`) falls back to a hand-rolled torch scatter-reduce when
cuDF is not importable, so a GPU fleet with torch and no RAPIDS still gets an accelerated
group-by. The translator's path requires cuDF.

It lives in `dist` because it is Ray scheduling — fan-out, partition descriptors, an autoscale
floor. It sat in the conductor, which is the layer that *sequences* the subsystems rather than
one that schedules tasks itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.dist.gpu.tasks import gpu_task_options

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.io.source import Source

__all__ = ["dispatch_gpu_aggregate", "distributed_gpu_aggregate", "partial_aggs"]


def partial_aggs(aggs: dict[str, tuple[str, str]]) -> dict[str, tuple[str, str]]:
    """The mergeable partial reductions a GPU worker computes per shard. A mean needs both a
    sum and a count partial; sum/count/min/max carry through as themselves."""
    partial: dict[str, tuple[str, str]] = {}
    for alias, (col_name, func) in aggs.items():
        if func == "mean":
            partial[f"{alias}__s"] = (col_name, "sum")
            partial[f"{alias}__n"] = (col_name, "count")
        else:
            partial[f"{alias}__{func}"] = (col_name, func)
    return partial


def _combine_partials(
    partials: list[pa.Table], key: str, aggs: dict[str, tuple[str, str]]
) -> pa.Table:
    """Combine per-shard GPU partials into the final aggregate (the mergeable `combine` step),
    reusing Batcher's own engine so the fold is native and tested."""
    import pyarrow as pa

    import batcher as bt
    from batcher import col

    combined = pa.concat_tables(partials)
    final: dict = {}
    means: dict[str, tuple[str, str]] = {}
    for alias, (_col, func) in aggs.items():
        if func == "sum":
            final[alias] = col(f"{alias}__sum").sum()
        elif func == "count":
            final[alias] = col(f"{alias}__count").sum()
        elif func == "min":
            final[alias] = col(f"{alias}__min").min()
        elif func == "max":
            final[alias] = col(f"{alias}__max").max()
        elif func == "mean":
            final[f"{alias}__st"] = col(f"{alias}__s").sum()
            final[f"{alias}__nt"] = col(f"{alias}__n").sum()
            means[alias] = (f"{alias}__st", f"{alias}__nt")
    ds = bt.from_arrow(combined).group_by(key).agg(**final)
    if means:
        ds = ds.with_columns(**{a: col(s) / col(n) for a, (s, n) in means.items()})
        ds = ds.select(key, *aggs.keys())
    return ds.collect()


def _gpu_partial_task(desc: dict, key: str, reductions: dict):
    """On a GPU worker: read this shard directly from storage, then partial-aggregate it on the
    GPU. Returns a small (one-row-per-group) partial table, or None for an empty shard."""
    import pyarrow as pa

    from batcher.core.gpu_transform import gpu_groupby_agg
    from batcher.dist.executors.partition_io import read_partition_descriptor

    batches = read_partition_descriptor(desc)
    if not batches:
        return None
    return gpu_groupby_agg(pa.Table.from_batches(batches), key, reductions)


def distributed_gpu_aggregate(
    source: Source, key: str, aggs: dict[str, tuple[str, str]], *, sharded: bool
) -> pa.Table | None:
    """GPU aggregate where the GPU WORKER reads its shard directly from storage — no
    whole-table transfer through the driver.

    `sharded=True` (Kyber says the working set exceeds one GPU) fans out `n_gpus x oversubscribe`
    shards across the cluster and combines the mergeable partials; `sharded=False` (fits one GPU)
    runs ONE shard on ONE GPU that reads the whole source itself, so even the single-GPU case
    avoids materializing the source on the driver and shipping it. Returns `None` when the source
    isn't splittable (an in-memory handle) or the cluster has no GPUs → the caller ships the
    in-memory table directly."""
    try:
        import ray

        from batcher.dist.executors.partition_io import _scan_splits
        from batcher.dist.executors.ray_runtime import _ensure_ray, cluster_topology
        from batcher.dist.executors.ray_runtime.scaling import (
            release_autoscale,
            request_autoscale,
        )
    except Exception:
        return None

    if not ray.is_initialized():
        return None
    n_gpus = int(cluster_topology().get("gpus", 0))
    if n_gpus < 1:
        return None
    # Only worth it for a genuinely splittable source (shared-nothing shard reads). An
    # in-memory source has no splits → let the driver-ships-table path handle it.
    from batcher.config import active_config
    from batcher.dist.executors.partition_io import WholeSourceSplit, partition_descriptors

    splits = _scan_splits(source, n_gpus)
    if len(splits) == 1 and isinstance(splits[0], WholeSourceSplit):
        return None

    if sharded:
        # Oversubscribe shards past the GPU count: bound each shard (no single-GPU OOM on a big
        # source), load-balance finely across the cluster, and keep a spot-preempted shard's
        # retry cheap. Ray runs at most `n_gpus` `num_gpus=1` tasks at once, so the rest pipeline.
        factor = max(1, int(active_config().distributed.gpu_shard_oversubscribe))
        n_shards = n_gpus * factor
    else:
        n_shards = 1  # fits one GPU: one worker reads the whole source, no distribution overhead

    # Hold a GPU autoscale floor for the query so a churning spot cluster keeps (or grows) the
    # GPU nodes instead of the autoscaler reclaiming them mid-aggregate; released in `finally`.
    request_autoscale(n_gpus, target_gpus=float(n_gpus))
    try:
        _ensure_ray(n_gpus)
        descs = partition_descriptors(source, n_shards)
        reductions = partial_aggs(aggs)
        task = ray.remote(**gpu_task_options())(_gpu_partial_task)
        partials = ray.get([task.remote(d, key, reductions) for d in descs])
    finally:
        release_autoscale()
    partials = [p for p in partials if p is not None]
    if not partials:
        return None
    return _combine_partials(partials, key, aggs)


def _gpu_aggregate_worker(table, key: str, aggs: dict):
    from batcher.core.gpu_transform import gpu_groupby_agg

    return gpu_groupby_agg(table, key, aggs)


def dispatch_gpu_aggregate(table: pa.Table, key: str, aggs: dict) -> pa.Table:
    """Run the group-by aggregate on a GPU — locally if this process owns one, else on a GPU
    worker (shipping batcher via `worker_runtime_env` so the worker can import the kernel)."""
    from batcher.core.gpu_transform import gpu_available

    if gpu_available():  # a GPU-equipped driver runs it in-process
        return _gpu_aggregate_worker(table, key, aggs)

    import ray

    from batcher.dist.executors.ray_runtime import _ensure_ray

    _ensure_ray(1)
    task = ray.remote(**gpu_task_options())(_gpu_aggregate_worker)
    return ray.get(task.remote(table, key, aggs))
