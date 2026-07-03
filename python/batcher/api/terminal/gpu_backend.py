"""The opt-in GPU execution backend for supported relational shapes.

`collect(backend="gpu")` routes a supported plan to the GPU and falls back to the native CPU
engine for everything else (and when no GPU is present). Two routes:

* A pure single-key group-by aggregate over a scan runs **distributed** — each GPU worker
  partial-aggregates its own shard (cuDF kernel), mergeable combine on the driver — so it scales
  past one GPU's memory (2B rows where single-GPU cuDF/Polars-GPU OOM).
* A linear chain of ops — filter, project / with_columns, multi-key group-by, sort, distinct,
  limit — is translated to a cuDF execution (`core.gpu_plan`) and run on one GPU.

The GPU tasks ship batcher (`worker_runtime_env`) + cuDF (a merged runtime_env) so the worker
runs the tested kernels. This is the "CPU and GPU backends for data transformations" seam: same
result, different *where*. GPU is an accelerator, never a requirement — an unsupported shape,
expression, OOM, or a GPU-less cluster silently uses the CPU engine, so `backend="gpu"` is
always safe to request.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.io.source import Source
    from batcher.plan.logical import LogicalPlan

_GPU_AGGS = ("sum", "count", "mean", "min", "max")


def try_gpu_collect(plan: LogicalPlan, sources: list[Source]) -> pa.Table | None:
    """Execute `plan` on the GPU if its shape is supported and a GPU exists, else `None`.

    `None` signals the caller to use the CPU engine — the safe fallback for any unsupported
    shape or a GPU-less cluster. A splittable source over a multi-GPU cluster runs the
    **distributed** GPU aggregate (each GPU worker reads and partial-aggregates its own shard —
    no whole-table transfer, uses every GPU); otherwise a single-dispatch path ships one table
    to one GPU (fine for small / in-memory sources)."""
    if not _cluster_has_gpu():
        return None
    import pyarrow as pa

    # Fast path: a pure single-key group-by aggregate over a scan runs DISTRIBUTED (each GPU
    # worker partial-aggregates its own shard, mergeable combine) — scales past one GPU's memory.
    spec = _gpu_agg_spec(plan)
    if spec is not None:
        key_out, key_src, aggs, scan = spec
        distributed = _distributed_gpu_aggregate(sources[scan.source_id], key_src, aggs)
        if distributed is not None:
            result = distributed
        else:
            batches = list(sources[scan.source_id].read())
            if not batches:
                return None
            result = _dispatch_gpu_aggregate(pa.Table.from_batches(batches), key_src, aggs)
        if key_out != key_src and key_src in result.column_names:
            result = result.rename_columns(
                [key_out if n == key_src else n for n in result.column_names]
            )
        return result

    # General path: a linear chain of supported ops (filter / project / sort / distinct / limit /
    # multi-key aggregate) is translated to cuDF and run on ONE GPU (single-dispatch). Correct for
    # any data that fits a GPU; OOM / an unsupported expression falls back to the CPU engine.
    from batcher.core.gpu_plan import gpu_join_spec, gpu_plan_ops

    plan_spec = gpu_plan_ops(plan)
    if plan_spec is not None:
        scan, ops = plan_spec
        batches = list(sources[scan.source_id].read())
        if not batches:
            return None
        return _dispatch_cudf_plan(pa.Table.from_batches(batches), ops)

    # A `[supported ops] over Join(scan, scan)` — an equi-join plus a chain — runs on one GPU.
    join_spec = gpu_join_spec(plan)
    if join_spec is not None:
        lscan, rscan, join_ir, ops = join_spec
        lb = list(sources[lscan.source_id].read())
        rb = list(sources[rscan.source_id].read())
        if not lb or not rb:
            return None
        return _dispatch_cudf_join(
            pa.Table.from_batches(lb), pa.Table.from_batches(rb), join_ir, ops
        )
    return None


def _partial_aggs(aggs: dict[str, tuple[str, str]]) -> dict[str, tuple[str, str]]:
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


def _gpu_partial_task(desc: dict, key: str, partial_aggs: dict):
    """On a GPU worker: read this shard directly from storage, then partial-aggregate it on the
    GPU. Returns a small (one-row-per-group) partial table, or None for an empty shard."""
    import pyarrow as pa

    from batcher.core.gpu_transform import gpu_groupby_agg
    from batcher.dist.executors.partition_io import read_partition_descriptor

    batches = read_partition_descriptor(desc)
    if not batches:
        return None
    return gpu_groupby_agg(pa.Table.from_batches(batches), key, partial_aggs)


def _distributed_gpu_aggregate(
    source: Source, key: str, aggs: dict[str, tuple[str, str]]
) -> pa.Table | None:
    """Distributed GPU aggregate: partition the (splittable) source, GPU-partial-aggregate each
    shard on its own GPU worker (no whole-table transfer), then combine the mergeable partials
    on the driver. Returns `None` when the source isn't splittable or the cluster has no GPUs,
    so the caller uses the single-dispatch path."""
    try:
        import ray

        from batcher.dist.executors.partition_io import _scan_splits
        from batcher.dist.executors.ray_runtime import _ensure_ray, cluster_topology
    except Exception:
        return None

    if not ray.is_initialized():
        return None
    n_gpus = int(cluster_topology().get("gpus", 0))
    if n_gpus < 1:
        return None
    # Only worth it for a genuinely splittable source (shared-nothing shard reads). An
    # in-memory source has no splits → let the single-dispatch path handle it.
    from batcher.dist.executors.partition_io import WholeSourceSplit, partition_descriptors

    splits = _scan_splits(source, n_gpus)
    if len(splits) == 1 and isinstance(splits[0], WholeSourceSplit):
        return None

    _ensure_ray(n_gpus)
    descs = partition_descriptors(source, n_gpus)
    partial_aggs = _partial_aggs(aggs)
    opts: dict = {"num_gpus": 1}
    rt = _gpu_task_runtime_env()
    if rt is not None:
        opts["runtime_env"] = rt
    task = ray.remote(**opts)(_gpu_partial_task)
    partials = ray.get([task.remote(d, key, partial_aggs) for d in descs])
    partials = [p for p in partials if p is not None]
    if not partials:
        return None
    return _combine_partials(partials, key, aggs)


def _cudf_plan_worker(table, ops):
    from batcher.core.gpu_plan import execute_cudf_plan

    return execute_cudf_plan(table, ops)


def _cudf_join_worker(left_t, right_t, join_ir, ops):
    from batcher.core.gpu_plan import execute_cudf_join

    return execute_cudf_join(left_t, right_t, join_ir, ops)


def _dispatch_cudf_join(left_t, right_t, join_ir: dict, ops: list[dict]) -> pa.Table | None:
    """Run an equi-join + op chain on one GPU via cuDF (in-process if this process owns a GPU
    with cuDF, else a GPU worker). `None` on any failure → CPU fallback."""
    from batcher.core.gpu_transform import gpu_available

    try:
        if gpu_available():
            with contextlib.suppress(Exception):
                return _cudf_join_worker(left_t, right_t, join_ir, ops)
        import ray

        from batcher.dist.executors.ray_runtime import _ensure_ray

        _ensure_ray(1)
        opts: dict = {"num_gpus": 1}
        rt = _gpu_task_runtime_env()
        if rt is not None:
            opts["runtime_env"] = rt
        return ray.get(
            ray.remote(**opts)(_cudf_join_worker).remote(left_t, right_t, join_ir, ops)
        )
    except Exception:
        return None


def _dispatch_cudf_plan(table: pa.Table, ops: list[dict]) -> pa.Table | None:
    """Run a translated op chain on ONE GPU via cuDF — in-process if this process owns a GPU with
    cuDF, else on a GPU worker (cuDF shipped in the runtime_env). Returns `None` on any failure —
    an unsupported expression, a cuDF-less worker, or a GPU OOM — so the caller uses the CPU
    engine. GPU is an accelerator, never a requirement."""
    from batcher.core.gpu_transform import gpu_available

    try:
        if gpu_available():
            with contextlib.suppress(Exception):
                return _cudf_plan_worker(table, ops)  # GPU-equipped process with cuDF
        import ray

        from batcher.dist.executors.ray_runtime import _ensure_ray

        _ensure_ray(1)
        opts: dict = {"num_gpus": 1}
        rt = _gpu_task_runtime_env()
        if rt is not None:
            opts["runtime_env"] = rt
        return ray.get(ray.remote(**opts)(_cudf_plan_worker).remote(table, ops))
    except Exception:
        return None  # cuDF-less / OOM / unsupported expr -> CPU fallback


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


def _gpu_task_runtime_env() -> dict | None:
    """The runtime_env for a GPU dispatch task: batcher (via `worker_runtime_env`, so the worker
    can import the kernel) plus — when `distributed.gpu_backend_cudf` is on — cuDF (pip) so the
    group-by uses cuDF's fast kernels. numpy is pinned to the cluster version so arrays returned
    from the task unpickle on the driver (cuDF's install otherwise drags numpy to 2.x)."""
    from batcher.config import active_config
    from batcher.dist.executors.ray_runtime.scheduling import worker_runtime_env

    rt = dict(worker_runtime_env() or {})
    if active_config().distributed.gpu_backend_cudf:
        rt["pip"] = ["cudf-cu13==26.6.0", "numpy==1.26.4"]
    return rt or None


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

    _ensure_ray(1)
    opts: dict = {"num_gpus": 1}
    rt = _gpu_task_runtime_env()
    if rt is not None:
        opts["runtime_env"] = rt
    task = ray.remote(**opts)(_gpu_aggregate_worker)
    return ray.get(task.remote(table, key, aggs))
