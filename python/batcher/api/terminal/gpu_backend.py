"""The opt-in GPU execution backend for a supported relational shape.

`collect(backend="gpu")` routes a supported plan — currently a single-key group-by aggregate
(sum/count/mean/min/max over numeric columns) directly over a scan — to the GPU, and falls
back to the native CPU engine for everything else (and when no GPU is present). The heavy
compute (the aggregate) runs on a GPU worker via the tested `core.gpu_transform` kernel; the
dispatch ships the driver's batcher with `worker_runtime_env()` so the worker can import it.

This is the "CPU and GPU backends for data transformations" seam: same result, different
*where*. GPU is an accelerator, never a requirement — an unsupported shape or a GPU-less
cluster silently uses the CPU engine, so `backend="gpu"` is always safe to request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.io.source import Source
    from batcher.plan.logical import LogicalPlan

_GPU_AGGS = ("sum", "count", "mean", "min", "max")


def try_gpu_collect(plan: LogicalPlan, sources: list[Source]) -> pa.Table | None:
    """Execute `plan` on the GPU if its shape is supported and a GPU exists, else `None`.

    `None` signals the caller to use the CPU engine — the safe fallback for any unsupported
    shape or a GPU-less cluster."""
    spec = _gpu_agg_spec(plan)
    if spec is None or not _cluster_has_gpu():
        return None
    key_out, key_src, aggs, scan = spec
    import pyarrow as pa

    batches = list(sources[scan.source_id].read())
    if not batches:
        return None
    table = pa.Table.from_batches(batches)
    result = _dispatch_gpu_aggregate(table, key_src, aggs)
    # The kernel names the group column by its source name; rename to the aggregate's alias.
    if key_out != key_src and key_src in result.column_names:
        result = result.rename_columns(
            [key_out if n == key_src else n for n in result.column_names]
        )
    return result


def _gpu_agg_spec(plan: LogicalPlan):
    """`(key_alias, key_col, {alias: (col, func)}, scan)` if `plan` is a GPU-executable
    single-key group-by aggregate directly over a scan, else `None`."""
    from batcher.plan.expr_ir import Col
    from batcher.plan.logical import Aggregate, Scan

    if not isinstance(plan, Aggregate) or not isinstance(plan.input, Scan):
        return None
    if len(plan.group_keys) != 1:
        return None
    gk = plan.group_keys[0]
    if not isinstance(gk.expr, Col):
        return None
    aggs: dict[str, tuple[str, str]] = {}
    for spec in plan.aggregates:
        ae = spec.agg
        if getattr(ae, "func", None) not in _GPU_AGGS or not isinstance(ae.input, Col):
            return None
        aggs[spec.alias] = (ae.input.name, ae.func)
    return gk.alias, gk.expr.name, aggs, plan.input


def _cluster_has_gpu() -> bool:
    """Whether the live cluster (or local process) exposes at least one GPU."""
    try:
        import ray

        if ray.is_initialized():
            from batcher.dist.executors.ray_runtime import cluster_topology

            return cluster_topology().get("gpus", 0) > 0
    except Exception:
        pass
    from batcher.core.gpu_transform import gpu_available

    return gpu_available()


def _gpu_aggregate_worker(table, key: str, aggs: dict):
    from batcher.core.gpu_transform import gpu_groupby_agg

    return gpu_groupby_agg(table, key, aggs)


def _dispatch_gpu_aggregate(table: pa.Table, key: str, aggs: dict) -> pa.Table:
    """Run the group-by aggregate on a GPU — locally if this process owns one, else on a GPU
    worker (shipping batcher via `worker_runtime_env` so the worker can import the kernel)."""
    from batcher.core.gpu_transform import gpu_available

    if gpu_available():  # a GPU-equipped driver runs it in-process
        return _gpu_aggregate_worker(table, key, aggs)

    import ray

    from batcher.dist.executors.ray_runtime import _ensure_ray
    from batcher.dist.executors.ray_runtime.scheduling import worker_runtime_env

    _ensure_ray(1)
    opts: dict = {"num_gpus": 1}
    rt = worker_runtime_env()
    if rt is not None:
        opts["runtime_env"] = rt
    task = ray.remote(**opts)(_gpu_aggregate_worker)
    return ray.get(task.remote(table, key, aggs))
