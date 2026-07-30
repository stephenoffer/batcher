"""The opt-in GPU execution backend for supported relational shapes.

`collect(backend="gpu")` routes a supported plan to the GPU and falls back to the native CPU
engine for everything else (and when no GPU is present). Two routes:

* A pure single-key group-by aggregate over a scan has the GPU **worker read its shard directly
  from storage** (cuDF kernel) — never materializing the source on the driver. Kyber sizes it:
  it fans out one shard per GPU (mergeable combine) when the working set exceeds one GPU — so it
  scales past one GPU's memory (2B rows where single-GPU cuDF/Polars-GPU OOM) — or runs a single
  worker-read shard when it fits one GPU. Only a non-splittable in-memory source is shipped whole.
* A linear chain of ops — filter, project / with_columns, multi-key group-by, sort, distinct,
  limit, window — is translated to a cuDF execution (`core.gpu_plan`) and run on one GPU.

The GPU tasks ship batcher (`worker_runtime_env`) + cuDF (a merged runtime_env) so the worker
runs the tested kernels. This is the "CPU and GPU backends for data transformations" seam: same
result, different *where*. GPU is an accelerator, never a requirement — an unsupported shape,
expression, OOM, or a GPU-less cluster silently uses the CPU engine, so `backend="gpu"` is
always safe to request.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed
from batcher.api.terminal.routing import _ray_already_live

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.io.source import Source
    from batcher.plan.logical import LogicalPlan

_GPU_AGGS = ("sum", "count", "mean", "min", "max")


def try_gpu_collect(
    plan: LogicalPlan, sources: list[Source], hub=None, *, force: bool = True
) -> pa.Table | None:
    """Execute `plan` on the GPU if Kyber's cost policy says it pays and the shape is supported,
    else `None`.

    `None` signals the caller to use the CPU engine — the safe fallback for any unsupported
    shape, a GPU-less cluster, or a plan Kyber routes to the CPU (too small to amortize the GPU
    overhead, or larger than the cluster's GPU memory). `force=True` (an explicit `backend="gpu"`)
    honors the request past the small-input threshold but still respects the memory routing;
    `force=False` (`backend="auto"`) lets Kyber decide fully. When the working set exceeds one
    GPU, the aggregate shards across GPUs (mergeable partials); otherwise a single-dispatch ships
    one table to one GPU."""
    gpu_count = _cluster_gpu_count()
    if gpu_count < 1:
        return None
    from batcher.dist.executors.ray_runtime.accelerators import (
        cluster_accelerator_type,
        cluster_gpu_memory_gb,
    )
    from batcher.kyber.gpu.policy import decide_gpu_backend

    decision = decide_gpu_backend(
        plan,
        sources,
        hub,
        gpu_count=gpu_count,
        force=force,
        gpu_memory_gb=cluster_gpu_memory_gb(),
        accelerator_type=cluster_accelerator_type(),
    )
    if not decision.use_gpu:
        return None
    import pyarrow as pa

    # Fast path: a pure single-key group-by aggregate over a scan runs DISTRIBUTED (each GPU
    # worker partial-aggregates its own shard, mergeable combine) — scales past one GPU's memory.
    spec = _gpu_agg_spec(plan)
    if spec is not None:
        import time

        key_out, key_src, aggs, scan = spec
        # Kyber routes by working-set size: shard across GPUs when it exceeds one GPU, else one
        # GPU. Either way the WORKER reads its shard from storage (no driver materialization); the
        # helper returns None only for a non-splittable in-memory source, which we then ship whole.
        t0 = time.perf_counter()
        result = _distributed_gpu_aggregate(
            sources[scan.source_id], key_src, aggs, sharded=decision.distributed
        )
        if result is None:
            batches = list(sources[scan.source_id].read())
            if not batches:
                return None
            result = _dispatch_gpu_aggregate(pa.Table.from_batches(batches), key_src, aggs)
        # Record this GPU run so Kyber can learn the GPU/CPU crossover (Core measures, Kyber
        # consumes). Keyed on the source's ACTUAL input rows — the same exact x the CPU side
        # records against — so the two fitted lines are directly comparable.
        gpu_ms = (time.perf_counter() - t0) * 1000.0
        _record_gpu_timing(hub, plan, sources, decision.est_rows, gpu_ms)
        if key_out != key_src and key_src in result.column_names:
            result = result.rename_columns(
                [key_out if n == key_src else n for n in result.column_names]
            )
        return result

    # General path: a linear chain of supported ops (filter / project / sort / distinct / limit /
    # multi-key aggregate) is translated to cuDF and run on ONE GPU (single-dispatch). Correct for
    # any data that fits a GPU; OOM / an unsupported expression falls back to the CPU engine.
    from batcher.core.gpu_plan import gpu_join_spec, gpu_plan_ops, gpu_union_spec

    plan_spec = gpu_plan_ops(plan)
    if plan_spec is not None:
        scan, ops = plan_spec
        batches = list(sources[scan.source_id].read())
        if not batches:
            return None
        return _dispatch_cudf_plan(pa.Table.from_batches(batches), ops)

    # A `[ops] over Join(chain, chain)` — an equi/semi/anti join plus the chains pushed below
    # it and above it — runs on one GPU.
    join_spec = gpu_join_spec(plan)
    if join_spec is not None:
        (lscan, lops), (rscan, rops), join_ir, ops = join_spec
        lb = list(sources[lscan.source_id].read())
        rb = list(sources[rscan.source_id].read())
        if not lb or not rb:
            return None
        return _dispatch_cudf_join(
            pa.Table.from_batches(lb), pa.Table.from_batches(rb), lops, rops, join_ir, ops
        )

    # A `[ops] over Union(chains)` — concat (+ optional dedup) + a chain — on one GPU.
    union_spec = gpu_union_spec(plan)
    if union_spec is not None:
        inputs, distinct, ops = union_spec
        read = [(list(sources[sc.source_id].read()), iops) for sc, iops in inputs]
        tables = [(pa.Table.from_batches(b), iops) for b, iops in read if b]
        if not tables:
            return None
        return _dispatch_cudf_union([t for t, _ in tables], [o for _, o in tables], distinct, ops)
    return None


def _agg_input_rows(plan, sources, fallback: int = 0) -> int:
    """The ACTUAL input row count for an aggregate-over-scan (the scan source's footer count) —
    the exact x-coordinate for the crossover learner, identical for the same source across GPU
    and CPU runs so the two fitted lines are directly comparable. An estimate drifts cold→warm
    and would pollute the fit; the footer count does not. `fallback` (the estimate) is used only
    when the source can't report an exact count."""
    try:
        spec = _gpu_agg_spec(plan)
        if spec is not None:
            rc = sources[spec[3].source_id].row_count()
            if rc:
                return int(rc)
    except Exception as exc:
        note_suppressed("api", "read exact rows for GPU sizing", exc)
    return fallback


def _record_gpu_timing(hub, plan, sources, est_rows: int, wall_ms: float) -> None:
    """Feed one GPU aggregate run's (actual input rows, wall time) to Kyber's crossover learner.
    Best-effort — a missing hub or unknown size is silently skipped; never breaks the query."""
    rows = _agg_input_rows(plan, sources, fallback=est_rows)
    if hub is None or rows <= 0:
        return
    from batcher.dist.executors.ray_runtime.accelerators import cluster_accelerator_type
    from batcher.kyber.gpu import record_backend_timing

    # Tagged with the device that produced it, so an H100 fleet's timings never join a T4's line.
    record_backend_timing(hub, "gpu", rows, wall_ms, cluster_accelerator_type())


def record_cpu_crossover(plan, sources, hub, wall_ms: float) -> None:
    """Record a CPU group-by run for Kyber's GPU/CPU crossover learner (Core measures, Kyber
    consumes). Best-effort and tightly gated: only a single-key aggregate over a scan, and only
    when the cluster actually has a GPU (else the crossover is irrelevant and this pays nothing —
    no estimator call). Never raises into the query. Lives here, next to the GPU-side recorder, so
    both halves of the crossover feed the same learner from one place."""
    try:
        if hub is None or _gpu_agg_spec(plan) is None or _cluster_gpu_count() < 1:
            return
        from batcher.kyber.gpu import record_backend_timing
        from batcher.kyber.gpu.policy import _estimate

        rows = _agg_input_rows(plan, sources, fallback=int(_estimate(plan, sources, hub)[0] or 0))
        if rows:
            record_backend_timing(hub, "cpu", rows, wall_ms)
    except Exception as exc:  # pragma: no cover - learning must never break a query
        note_suppressed("api", "record the GPU/CPU crossover point", exc)
        return


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
        partial_aggs = _partial_aggs(aggs)
        opts = _gpu_task_opts()
        task = ray.remote(**opts)(_gpu_partial_task)
        partials = ray.get([task.remote(d, key, partial_aggs) for d in descs])
    finally:
        release_autoscale()
    partials = [p for p in partials if p is not None]
    if not partials:
        return None
    return _combine_partials(partials, key, aggs)


def _cudf_plan_worker(table, ops):
    from batcher.core.gpu_plan import execute_cudf_plan

    return execute_cudf_plan(table, ops)


def _cudf_join_worker(left_t, right_t, left_ops, right_ops, join_ir, ops):
    from batcher.core.gpu_plan import execute_cudf_join

    return execute_cudf_join(left_t, right_t, left_ops, right_ops, join_ir, ops)


def _cudf_union_worker(tables, input_ops, distinct, ops):
    from batcher.core.gpu_plan import execute_cudf_union

    return execute_cudf_union(tables, input_ops, distinct, ops)


def _dispatch_cudf_union(
    tables: list, input_ops: list[list[dict]], distinct: bool, ops: list[dict]
) -> pa.Table | None:
    """Run a union (+ op chain) on one GPU via cuDF; `None` on failure → CPU fallback."""
    return _dispatch_on_gpu(_cudf_union_worker, tables, input_ops, distinct, ops)


def _dispatch_cudf_join(
    left_t, right_t, left_ops: list[dict], right_ops: list[dict], join_ir: dict, ops: list[dict]
) -> pa.Table | None:
    """Run a join + op chain on one GPU via cuDF (in-process if this process owns a GPU with
    cuDF, else a GPU worker). `None` on any failure → CPU fallback."""
    return _dispatch_on_gpu(_cudf_join_worker, left_t, right_t, left_ops, right_ops, join_ir, ops)


def _dispatch_cudf_plan(table: pa.Table, ops: list[dict]) -> pa.Table | None:
    """Run a translated op chain on ONE GPU via cuDF; `None` on failure → CPU fallback."""
    return _dispatch_on_gpu(_cudf_plan_worker, table, ops)


def _dispatch_on_gpu(worker, *args) -> pa.Table | None:
    """Run `worker(*args)` on a GPU — in-process when this process owns one with cuDF, else on
    a GPU worker (cuDF shipped in the runtime_env).

    Returns `None` on any failure — an untranslatable expression, a cuDF-less worker, or a GPU
    OOM — so the caller uses the CPU engine. GPU is an accelerator, never a requirement, and
    every dispatch shape (chain, join, union) needs exactly this behavior, so they share it
    rather than restating it three times with three chances to drift."""
    from batcher.core.gpu_transform import gpu_available

    try:
        if gpu_available():
            with contextlib.suppress(Exception):
                return worker(*args)  # GPU-equipped process with cuDF
        import ray

        from batcher.dist.executors.ray_runtime import _ensure_ray

        _ensure_ray(1)
        return ray.get(ray.remote(**_gpu_task_opts())(worker).remote(*args))
    except Exception:
        return None


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


def _gpu_task_opts() -> dict:
    """Ray remote-options for a GPU dispatch task: one GPU, the batcher+cuDF runtime_env, and a
    spot-preemption retry budget.

    `max_retries` reruns a task whose GPU worker/node was lost (spot reclamation) on surviving
    capacity — so a large GPU query on a churning spot cluster self-heals instead of one lost
    shard collapsing it to the single-node CPU fallback. `retry_exceptions` is deliberately left
    off: a deterministic application error (a GPU OOM, an unsupported expression) must fall back
    to the CPU engine immediately, not after N pointless retries."""
    from batcher.config import active_config

    opts: dict = {"num_gpus": 1, "max_retries": int(active_config().distributed.task_max_retries)}
    rt = _gpu_task_runtime_env()
    if rt is not None:
        opts["runtime_env"] = rt
    return opts


def _cluster_gpu_count() -> int:
    """Live GPU count across the cluster (or 1 for a GPU-equipped local process), else 0.

    The count — not just presence — so Kyber's policy can size the cluster's aggregate GPU
    memory and pick single-device vs sharded execution."""
    # Gated exactly as the `distributed="auto"` probe is, and for the same reason: this runs
    # on every terminal op, and `import ray` costs ~0.44 s the first time to answer a question
    # `sys.modules` already settles — a cluster cannot be initialized in a process that has
    # not imported Ray. See `routing._ray_already_live` for the full argument.
    if _ray_already_live():
        try:
            import ray

            if ray.is_initialized():
                from batcher.dist.executors.ray_runtime import cluster_topology

                return int(cluster_topology().get("gpus", 0))
        except Exception as exc:
            note_suppressed("api", "read cluster GPU topology", exc)
    from batcher.core.gpu_transform import gpu_available

    return 1 if gpu_available() else 0


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
    opts = _gpu_task_opts()
    task = ray.remote(**opts)(_gpu_aggregate_worker)
    return ray.get(task.remote(table, key, aggs))
