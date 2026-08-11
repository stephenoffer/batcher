"""Decide whether a plan runs on the GPU, run it, and record what that cost.

The entry point `api.terminal.core` calls, and the one place the fallback contract is stated:
an unsupported shape, a device out of memory, a lost worker or a cluster with no GPU all return
`None`, and the caller uses the CPU engine. `backend="gpu"` is therefore always safe to ask for.

It optimizes the plan before translating it. The plan arriving here is the one the user built,
and translating *that* meant a multi-way join read and joined its inputs unfiltered — device
intermediates the CPU engine never materializes — and a sixteen-column fact table crossed the
host link to answer a four-column query. Both are ordinary optimizer output the GPU path simply
never saw.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.io.source import Source
    from batcher.plan.logical import LogicalPlan

from batcher.api.terminal.gpu_backend.failure import note_gpu_failure
from batcher.api.terminal.gpu_backend.fanout import _cluster_gpu_count
from batcher.api.terminal.gpu_backend.translate import (
    _gpu_agg_spec,
    _legacy_groupby,
    _translated,
)


def try_gpu_collect(
    plan: LogicalPlan,
    sources: list[Source],
    hub=None,
    *,
    force: bool = True,
    columns: list[str] | None = None,
) -> pa.Table | None:
    """Execute `plan` on the GPU if Kyber's cost policy says it pays and the shape is supported,
    else `None`.

    `columns` is the terminal op's requested output columns, needed only to build the CPU
    engine's context under `distributed.gpu_shadow_verify`; it is otherwise unused.

    `None` signals the caller to use the CPU engine — the safe fallback for any unsupported
    shape, a GPU-less cluster, or a plan Kyber routes to the CPU (too small to amortize the GPU
    overhead, or larger than the cluster's GPU memory). `force=True` (an explicit `backend="gpu"`)
    honors the request past the small-input threshold but still respects the memory routing;
    `force=False` (`backend="auto"`) lets Kyber decide fully. When the working set exceeds one
    GPU, the aggregate shards across GPUs (mergeable partials); otherwise a single-dispatch ships
    one table to one GPU."""
    gpu_count = _cluster_gpu_count()
    if gpu_count < 1:
        if force:
            _note_no_visible_device()
        return None
    from batcher.dist.executors.ray_runtime.accelerators import (
        cluster_accelerator_type,
        cluster_gpu_memory_gb,
    )
    from batcher.kyber.gpu.policy import decide_gpu_backend

    # The plan reaching here is the one the user built, before any rewriting. Translating *that*
    # was the largest single cost on this path and the least visible: without predicate pushdown
    # a multi-way join reads and joins its inputs unfiltered, so the device holds intermediates
    # the CPU engine never materializes and runs out of memory doing it; without projection
    # pruning a sixteen-column fact table crosses the host link whole to answer a four-column
    # query. Both are ordinary optimizer output, and the GPU backend simply never saw it.
    #
    # Memoized on the same key the CPU path uses, so on the fallback route this costs one cache
    # hit rather than a second optimization.
    raw_plan = plan
    plan = _optimized(plan, sources, hub)

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
        note_gpu_failure("run this plan on the GPU; using the CPU engine", exc)
        return None
    if result is None:
        return None
    # Stopped before verification of either kind: the recorded figure has to be what the device
    # path costs, not what checking it costs, or the learned GPU/CPU crossover moves the moment
    # an operator switches a diagnostic on.
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    from batcher.api.terminal.gpu_backend.verify import enforce_schema_contract

    # The engine declares this plan's column types without running it, so every device result
    # can be held against them for the price of a field-list walk. Unconditional, unlike
    # shadow-verify below: every defect this tier has shipped was a column-type defect with
    # correct values, on a device, invisible to a pandas CI replay — and this is the only check
    # that sees that class without a GPU lane and without doubling the work. A result that
    # disagrees is refused, and the CPU engine answers the query.
    checked = enforce_schema_contract(result, plan)
    if checked is None:
        return None
    result = checked
    from batcher.config import active_config

    if active_config().distributed.gpu_shadow_verify:
        from batcher.api.terminal.gpu_backend.verify import shadow_verify

        result = shadow_verify(plan, sources, columns or [], result)
    # Record this GPU run so Kyber can learn the GPU/CPU crossover (Core measures, Kyber
    # consumes). Keyed on the source's ACTUAL input rows — the same exact x the CPU side records
    # against — so the two fitted lines are directly comparable. Against the *raw* plan for
    # exactly that reason: the CPU recorder is handed the raw one, and an optimized plan can
    # match `_gpu_agg_spec` where the raw one does not (or the reverse), which would put the two
    # backends' timings on two different x-axes.
    _record_gpu_timing(hub, raw_plan, sources, decision.est_rows, elapsed_ms)
    return result


#: Whether the "no device for an explicit GPU request" warning has already been given. Once per
#: process: the condition is a property of the cluster, not of the query, so repeating it on
#: every terminal op would be noise around a fact that cannot change mid-run.
_NOTED_NO_DEVICE = False


def _note_no_visible_device() -> None:
    """Say why an explicit `backend="gpu"` found no device, once per process.

    `backend="gpu"` is documented as always safe, and it is — the CPU engine returns the same
    rows. What it was not is *audible*: on a cluster whose driver is CPU-only, every explicit
    GPU request fell back silently and the user's only signal was the running time.

    And the reason is almost never "there are no GPUs". The device count is read from the live
    Ray topology, and only when Ray is already initialized in this process — a `sys.modules`
    gate that exists so an ordinary small query does not pay a 0.44 s `import ray` to be told
    it is single-node. A CPU-only head node with four GPU workers therefore reports zero
    devices until something initializes Ray, which a plain `collect()` never does. That is an
    actionable difference and it is what this message carries.
    """
    global _NOTED_NO_DEVICE
    if _NOTED_NO_DEVICE:
        return
    _NOTED_NO_DEVICE = True

    import logging

    from batcher._internal.logging import get_logger, log_kv
    from batcher.api.terminal.routing import _ray_already_live

    detail = (
        "the cluster reports no GPU"
        if _ray_already_live()
        else (
            "no GPU is visible from this process: the cluster's devices are read from the live "
            "Ray topology, and Ray is not initialized here. Run with distributed=True, or "
            "initialize Ray first, to reach the cluster's GPUs"
        )
    )
    # Warned rather than noted: this is a decline, not a defect, so it is not routed through
    # `note_gpu_failure` — but it is a decline of something the caller asked for *explicitly*,
    # and the only signal it otherwise leaves is the running time.
    log_kv(
        get_logger("api"),
        logging.WARNING,
        'backend="gpu" was requested and the CPU engine ran this query instead',
        reason=detail,
    )


def _optimized(plan: LogicalPlan, sources: list[Source], hub):
    """`plan` as Kyber would rewrite it, or `plan` itself when the optimizer cannot run.

    Failing back to the raw plan rather than to the CPU engine: an unoptimized plan the device
    can run is still a correct answer and usually still a faster one, and the shapes that make
    the optimizer decline are not the shapes the device is being asked about.
    """
    from batcher import kyber

    try:
        return kyber.optimize_logical(plan, sources=sources, hub=hub)
    except Exception as exc:
        note_suppressed("api", "optimize the plan before translating it for the GPU", exc)
        return plan


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
    # The same run, as a throughput rather than as a point on a line. It is what a fan-out
    # divides its shards by, and it is learnable from GPU runs alone — the crossover fit needs
    # CPU samples this fleet may never produce.
    #
    # Divided by the devices that shared the work, because what is measured here is a whole
    # query: `rows` is the source's total and `wall_ms` is the collect. On an eight-device
    # fan-out that is eight boards' throughput recorded under one board's name, and the
    # consumer deals shards *between* device models on the assumption it is per device — so
    # two models measured at different fan-out widths are compared as though the wider one
    # were the faster part.
    record_device_throughput(hub, device, rows, wall_ms / 1000.0, devices=_cluster_gpu_count())


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
