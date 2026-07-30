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
from batcher.plan.ir_tags import Op

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.io.source import Source

__all__ = ["combine_ops", "dispatch_gpu_aggregate", "distributed_gpu_aggregate", "partial_aggs"]


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


def _col(name: str) -> dict:
    """A column reference in expression IR."""
    return {"e": "col", "name": name}


def combine_ops(key: str, aggs: dict[str, tuple[str, str]]) -> list[dict]:
    """The operator chain that folds per-shard GPU partials into the final aggregate.

    The mergeable `combine` step, written as IR rather than as a `Dataset`. It is the same
    algebra either way, but the IR form is what keeps this module inside its layer: `dist`
    schedules the operators the conductor hands it and must not reach back up through the
    public API to do so.

    A `sum` of sums is a sum and a `count` of counts is *also* a sum — counting the partials
    again would return the number of shards. A `mean` arrives as separate sum and count
    partials and is divided after both have been folded, because the mean of shard means is
    not the mean unless every shard held the same number of rows.

    Args:
        key: The group-by key column.
        aggs: The user's aggregates as `{alias: (column, func)}`.

    Returns:
        A bottom-up operator IR chain, ready for `nest_ops`.
    """
    reductions: list[dict] = []
    means: dict[str, tuple[str, str]] = {}
    for alias, (_col_name, func) in aggs.items():
        if func == "mean":
            total, count = f"{alias}__st", f"{alias}__nt"
            reductions.append({"func": "sum", "alias": total, "input": _col(f"{alias}__s")})
            reductions.append({"func": "sum", "alias": count, "input": _col(f"{alias}__n")})
            means[alias] = (total, count)
        else:
            fold = "sum" if func == "count" else func
            reductions.append({"func": fold, "alias": alias, "input": _col(f"{alias}__{func}")})
    ops: list[dict] = [
        {
            "op": Op.AGGREGATE,
            "group_keys": [{"expr": _col(key), "alias": key}],
            "aggregates": reductions,
        }
    ]
    if means:
        ops.append({"op": Op.PROJECT, "exprs": _mean_projection(key, aggs, means)})
    return ops


def _mean_projection(key: str, aggs: dict, means: dict[str, tuple[str, str]]) -> list[dict]:
    """Divide each folded sum by its folded count, and present the user's own column order."""
    exprs = [{"expr": _col(key), "alias": key}]
    for alias in aggs:
        if alias not in means:
            exprs.append({"expr": _col(alias), "alias": alias})
            continue
        total, count = means[alias]
        # The numerator is cast so an integer column's mean is a mean and not a floor
        # division, matching what `col(a) / col(b)` lowers to on the public API.
        exprs.append(
            {
                "expr": {
                    "e": "binary",
                    "op": "div",
                    "left": {
                        "e": "cast",
                        "input": _col(total),
                        "dtype": "float64",
                        "try_cast": False,
                    },
                    "right": _col(count),
                },
                "alias": alias,
            }
        )
    return exprs


def _combine_partials(
    partials: list[pa.Table], key: str, aggs: dict[str, tuple[str, str]]
) -> pa.Table:
    """Combine per-shard GPU partials into the final aggregate (the mergeable `combine` step),
    running the fold on Batcher's own engine so it is native and tested."""
    import json

    import pyarrow as pa

    from batcher._internal.native import engine
    from batcher.dist.executors.ray_runtime import engine_config_json
    from batcher.plan.distribution import nest_ops

    combined = pa.concat_tables(partials)
    plan = json.dumps(nest_ops(combine_ops(key, aggs)))
    out = engine().execute_plan(plan, [combined.to_batches()], engine_config_json())
    return pa.Table.from_batches(out, schema=out[0].schema) if out else combined


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
