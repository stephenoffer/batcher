"""The opt-in GPU execution backend for supported relational shapes.

`collect(backend="gpu")` routes a supported plan to the GPU and falls back to the native CPU
engine for everything else (and when no GPU is present). Two routes:

* A pure single-key group-by aggregate over a scan has the GPU **worker read its shard directly
  from storage** (cuDF kernel) — never materializing the source on the driver. Kyber sizes it:
  it fans out one shard per GPU (mergeable combine) when the working set exceeds one GPU — so it
  scales past one GPU's memory (2B rows where single-GPU cuDF/Polars-GPU OOM) — or runs a single
  worker-read shard when it fits one GPU. Only a non-splittable in-memory source is shipped whole.
* A linear chain of ops — filter, project / with_columns, group-by aggregate, sort, distinct,
  limit, window, join, union — is translated to a cuDF execution (`core.gpu_plan`). A chain
  ending in a **mergeable** aggregate fans out across every GPU (`dist.gpu`), each device
  reducing the shard it reads itself; anything else runs as a single dispatch on one GPU.

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
from batcher.dist.gpu import gpu_task_options
from batcher.dist.gpu.groupby import dispatch_gpu_aggregate, distributed_gpu_aggregate

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

    import time

    # The TRANSLATED path is tried first, because it is a strict superset of what the legacy
    # group-by kernel below covers: more reductions, chains above and below the reducer,
    # per-shard recovery, and shard sizing from what the plan wants. Trying the legacy path
    # first, as this used to, meant the single most common GPU shape — a one-key group-by over
    # a scan — never reached any of it.
    t0 = time.perf_counter()
    try:
        result = _translated(plan, sources, gpu_count, decision)
        if result is None:
            result = _legacy_groupby(plan, sources, decision)
    except Exception as exc:
        # `backend="gpu"` is documented as always safe: an unsupported shape, a lost device or
        # a kernel that cannot handle this data uses the CPU engine and returns the same rows.
        # Nothing enforced that. A raise from here reached the caller, which has no handler,
        # so a query the GPU could not run *failed* instead of running — the legacy kernel
        # raised a bare `TypeError` on a string group key, which is an ordinary column.
        note_suppressed("api", "run this plan on the GPU; using the CPU engine", exc)
        return None
    if result is None:
        return None
    # Record this GPU run so Kyber can learn the GPU/CPU crossover (Core measures, Kyber
    # consumes). Keyed on the source's ACTUAL input rows — the same exact x the CPU side records
    # against — so the two fitted lines are directly comparable.
    _record_gpu_timing(hub, plan, sources, decision.est_rows, (time.perf_counter() - t0) * 1000.0)
    return result


def _translated(plan: LogicalPlan, sources: list[Source], gpu_count: int, decision):
    """Run `plan` through the plan translator, or `None` when it does not apply.

    Three attempts in descending order of ambition, each a fallback for the one before:

    1. **Fan out.** Every chain whose shape divides — a mergeable reducer to fold, or a
       row-local chain to reassemble — runs a shard per device, each device reading its own
       shard from storage.
    2. **One worker, reading for itself.** Reached when the fan-out declined or failed, so this
       is a retry as a single shard rather than the usual path. It still keeps the source off
       the driver, which is the point: staging a large relation on the driver to send it to a
       GPU is the wrong end of the machine, and the driver is routinely the smallest node.
    3. **Ship the table.** Only for an in-memory source, whose rows are on the driver by
       construction, or a process that owns a device but has no Ray to schedule with.
    """
    import pyarrow as pa

    from batcher.core.gpu_plan import gpu_join_spec, gpu_plan_ops, gpu_union_spec
    from batcher.dist.gpu import gpu_chain_on_worker, gpu_join_on_worker, gpu_union_on_worker

    plan_spec = gpu_plan_ops(plan)
    if plan_spec is not None:
        scan, ops = plan_spec
        source = sources[scan.source_id]
        fanned = _try_sharded_aggregate(source, ops, gpu_count, decision)
        if fanned is not None:
            return fanned
        on_worker = gpu_chain_on_worker(source, ops)
        if on_worker is not None:
            return on_worker
        batches = list(source.read())  # in-memory source: the rows are on the driver already
        if not batches:
            return None
        return _dispatch_cudf_plan(pa.Table.from_batches(batches), ops)

    # A `[ops] over Join(chain, chain)`. A join the planner marked `broadcast` splits its probe
    # side across every device, each reading the whole build side itself; anything else runs on
    # one GPU, which reads both sides itself.
    join_spec = gpu_join_spec(plan)
    if join_spec is not None:
        (lscan, lops), (rscan, rops), join_ir, ops = join_spec
        left, right = sources[lscan.source_id], sources[rscan.source_id]
        fanned = _try_sharded_join(left, right, lops, rops, join_ir, ops, gpu_count, decision)
        if fanned is not None:
            return fanned
        on_worker = gpu_join_on_worker(left, right, lops, rops, join_ir, ops)
        if on_worker is not None:
            return on_worker
        lb, rb = list(left.read()), list(right.read())
        if not lb or not rb:
            return None
        return _dispatch_cudf_join(
            pa.Table.from_batches(lb), pa.Table.from_batches(rb), lops, rops, join_ir, ops
        )

    # A `[ops] over Union(chains)` — concat (+ optional dedup) + a chain. A `UNION ALL` shards
    # each of its inputs across every device; anything else runs on one GPU, which reads every
    # input itself.
    union_spec = gpu_union_spec(plan)
    if union_spec is not None:
        inputs, distinct, ops = union_spec
        usources = [sources[sc.source_id] for sc, _ in inputs]
        input_ops = [o for _, o in inputs]
        fanned = _try_sharded_union(usources, input_ops, distinct, ops, gpu_count, decision)
        if fanned is not None:
            return fanned
        on_worker = gpu_union_on_worker(usources, input_ops, distinct, ops)
        if on_worker is not None:
            return on_worker
        read = [(list(sources[sc.source_id].read()), iops) for sc, iops in inputs]
        tables = [(pa.Table.from_batches(b), iops) for b, iops in read if b]
        if not tables:
            return None
        return _dispatch_cudf_union([t for t, _ in tables], [o for _, o in tables], distinct, ops)
    return None


def _legacy_groupby(plan: LogicalPlan, sources: list[Source], decision):
    """The single-key group-by fan-out, for a fleet whose workers have torch but not cuDF.

    Reached only when the translator declined or could not run — which on a normal cluster is
    never, since cuDF ships with the task's runtime_env. Its kernel falls back to a torch
    scatter-reduce, so it is the difference between an accelerated group-by and none at all on
    a RAPIDS-less fleet.
    """
    import pyarrow as pa

    spec = _gpu_agg_spec(plan)
    if spec is None:
        return None
    key_out, key_src, aggs, scan = spec
    result = distributed_gpu_aggregate(
        sources[scan.source_id], key_src, aggs, sharded=decision.distributed
    )
    if result is None:
        batches = list(sources[scan.source_id].read())
        if not batches:
            return None
        result = dispatch_gpu_aggregate(pa.Table.from_batches(batches), key_src, aggs)
    if key_out != key_src and key_src in result.column_names:
        return result.rename_columns([key_out if n == key_src else n for n in result.column_names])
    return result


def _with_gpu_capacity(gpu_count: int, decision, run):
    """Run a fan-out with the cluster grown to the devices the plan wants, then released.

    Asks the autoscaler for `decision.desired_gpus` — enough to hold the working set in one
    wave — rather than for the devices the cluster already has. Asking for what is already
    there pins the floor against reclamation and can never grow the cluster, so a query that
    could use thirty-two devices ran on the four it happened to find. It then waits (bounded,
    and a no-op on a fixed cluster) before `run` sizes its shards, since sizing against the
    pre-scale topology is how a query asks for capacity and then declines to use it.

    `run` receives the device count the wait actually produced. Any failure inside is a
    `None` — every fan-out has a slower path behind it, and none of them change the answer.
    """
    from batcher.dist.executors.ray_runtime.scaling import (
        await_autoscale,
        release_autoscale,
        request_autoscale,
    )

    wanted = max(gpu_count, int(decision.desired_gpus))
    request_autoscale(gpu_count, target_gpus=float(wanted))
    try:
        # One core per device task is the floor a GPU stage needs; the GPU target is the
        # binding one. A zero CPU target would make the wait a no-op, since it reads as
        # "this query wants nothing".
        await_autoscale(wanted, target_gpus=float(wanted))
        return run(_cluster_gpu_count())
    except Exception as exc:
        note_suppressed("api", "fan a GPU stage out across devices", exc)
        return None
    finally:
        release_autoscale()


def _try_sharded_aggregate(source: Source, ops: list[dict], gpu_count: int, decision):
    """Fan a chain with a mergeable reducer out across the cluster's GPUs, or `None`.

    `None` means the fan-out does not apply — the chain has no shardable split, the source is
    not splittable, or the cluster is unreadable — and the caller then uses the single-device
    dispatch. Every failure mode is a slower path, never a different answer: the fan-out is the
    mergeable decomposition of the same chain."""
    from batcher.dist.gpu import sharded_gpu_aggregate

    return _with_gpu_capacity(
        gpu_count,
        decision,
        lambda live: sharded_gpu_aggregate(
            source, ops, gpu_count=live, sharded=decision.distributed
        ),
    )


def _try_sharded_join(left, right, lops, rops, join_ir, ops, gpu_count: int, decision):
    """Fan a broadcast-safe join out across the cluster's GPUs, or `None`."""
    from batcher.dist.gpu import sharded_gpu_join

    return _with_gpu_capacity(
        gpu_count,
        decision,
        lambda live: sharded_gpu_join(
            left,
            right,
            lops,
            rops,
            join_ir,
            ops,
            gpu_count=live,
            sharded=decision.distributed,
        ),
    )


def _try_sharded_union(usources, input_ops, distinct, ops, gpu_count: int, decision):
    """Fan a `UNION ALL` out across the cluster's GPUs, or `None`.

    A deduplicating union declines inside the fan-out rather than here, so the rule about why
    lives beside the algebra that cannot honour it."""
    from batcher.dist.gpu import sharded_gpu_union

    return _with_gpu_capacity(
        gpu_count,
        decision,
        lambda live: sharded_gpu_union(
            usources,
            input_ops,
            distinct,
            ops,
            gpu_count=live,
            sharded=decision.distributed,
        ),
    )


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
    from batcher.kyber.gpu.adaptive import record_device_throughput, shape_key

    # Tagged with the device that produced it, so an H100 fleet's timings never join a T4's line,
    # and with the query's shape, so a wide transfer-bound projection does not average against a
    # narrow group-by on the same board.
    device = cluster_accelerator_type()
    record_backend_timing(hub, "gpu", rows, wall_ms, device, shape_key(plan))
    # The same run as a throughput rather than a point on a line. It is what a fan-out divides
    # its shards by, and it is learnable from GPU runs alone — the crossover fit needs CPU
    # samples this fleet may never produce.
    record_device_throughput(hub, device, rows, wall_ms / 1000.0)


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
        from batcher.kyber.gpu.adaptive import shape_key
        from batcher.kyber.gpu.policy import _estimate

        rows = _agg_input_rows(plan, sources, fallback=int(_estimate(plan, sources, hub)[0] or 0))
        if rows:
            # The same shape key as the GPU half. Both lines of a crossover have to come from
            # the same rung of the ladder, so recording one shaped and the other pooled would
            # leave the shaped bucket permanently unusable.
            record_backend_timing(hub, "cpu", rows, wall_ms, None, shape_key(plan))
    except Exception as exc:  # pragma: no cover - learning must never break a query
        note_suppressed("api", "record the GPU/CPU crossover point", exc)
        return


def _on_device():
    """Configure this worker's device allocator before it computes.

    Runs inside the task body rather than at submission: only the process that was handed a
    device knows how much of it is free, and the pool it builds has to live in that process.
    Idempotent, so a worker reused across tasks keeps the pool the first one paid for.
    """
    from batcher.carbonite.accel import prepare_device_memory

    prepare_device_memory()


def _cudf_plan_worker(table, ops):
    from batcher.core.gpu_plan import execute_cudf_plan

    _on_device()
    return execute_cudf_plan(table, ops)


def _cudf_join_worker(left_t, right_t, left_ops, right_ops, join_ir, ops):
    from batcher.core.gpu_plan import execute_cudf_join

    _on_device()
    return execute_cudf_join(left_t, right_t, left_ops, right_ops, join_ir, ops)


def _cudf_union_worker(tables, input_ops, distinct, ops):
    from batcher.core.gpu_plan import execute_cudf_union

    _on_device()
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
        return ray.get(ray.remote(**gpu_task_options())(worker).remote(*args))
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
