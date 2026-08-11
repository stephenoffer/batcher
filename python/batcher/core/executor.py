"""The Core local executor.

Drives a `PhysicalPlan` to completion against in-memory input relations by
handing the lowered IR to the native engine, then transcribes the native
engine's per-operator `ExecMetrics` into `OperatorFeedback` for the MetadataHub.
Core *measures* — it does not optimize: it faithfully reports what the data plane
observed (rows in/out, time, peak bytes, spill, backend), keyed by the same
pre-order operator id Kyber assigns in `annotate_ops`. The morsel scheduler and
JIT tier-up replace the single `execute_plan_metered` call without changing this
interface. Re-optimization is **not** one of them: the stage-boundary loop lives
in `api/adaptive/`, in the control plane, because it re-runs Kyber between
stages. There is no `bc-adapt` crate and the Rust side has no adaptivity of its
own; earlier revisions of this docstring said otherwise.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pyarrow as pa

from batcher._internal.hardware import fingerprint
from batcher._internal.hardware.cgroup import cgroup_throttled_ratio
from batcher._internal.hardware.cpu import cpu_thermal_events
from batcher._internal.mathx import safe_div
from batcher._internal.native import engine
from batcher.config import active_config
from batcher.core.runtime import current_query_id
from batcher.plan.feedback import FeedbackSink, OperatorFeedback, cpu_utilization
from batcher.plan.ids import OpId
from batcher.plan.physical import PhysicalOp, PhysicalPlan

__all__ = ["LocalExecutor", "execute_local", "execute_local_metered", "record_exec_metrics"]


def record_exec_metrics(
    sink: FeedbackSink | None,
    metrics_json: str,
    batch_size: int,
    planned: Sequence[PhysicalOp] = (),
) -> None:
    """Transcribe a native `ExecMetrics` document into per-operator `OperatorFeedback`.

    The one place the engine's measured runtime facts (rows in/out, time, peak
    bytes, spill, backend) become feedback — shared by the single-node executor and
    the distributed workers (which run sub-plans and ship their metrics back to the
    driver). Calibration buckets by operator `kind`, so the sub-plan-local `op_id`s a
    distributed worker reports need no global correlation — such a caller passes no
    `planned` ops and its rows carry no signature. Best-effort: a malformed or empty
    document drops silently rather than failing the query.
    """
    if sink is None:
        return
    try:
        ops = json.loads(metrics_json).get("ops", [])
    except (ValueError, TypeError):
        return
    _record_op_feedback(sink, ops, batch_size, planned)


_tracing_started = False


def _ensure_native_tracing(native) -> None:
    """Install the Rust data-plane tracing bridge once, at the configured level.

    Core is the layer allowed to touch the native engine, so it (not the neutral logging
    module) calls `init_tracing` — keeping `_native` out of `_internal` and the
    layer-independence contract intact. A no-op after the first call and when logging is
    unconfigured or the engine predates the bridge.
    """
    global _tracing_started
    if _tracing_started:
        return
    from batcher._internal.logging import native_tracing_settings

    settings = native_tracing_settings()
    init = getattr(native, "init_tracing", None)
    if settings is not None and init is not None:
        import contextlib

        with contextlib.suppress(Exception):  # tracing init must never break a query
            init(*settings)
        _tracing_started = True


def _record_op_feedback(
    sink: FeedbackSink,
    ops: list[dict],
    batch_size: int,
    planned: Sequence[PhysicalOp] = (),
) -> None:
    """Build and record one `OperatorFeedback` per already-parsed `ExecMetrics` op.

    `planned` are Kyber's annotated operators for the plan that produced `ops`, indexed
    by `op_id` (both are the plan's pre-order walk). When supplied, each feedback row
    carries the operator's stable `signature` and the *uncorrected* row estimate, which
    is what closes Kyber's cardinality-correction loop. It is omitted by the distributed
    workers, whose `op_id`s address their own sub-plan and cannot be correlated with the
    driver's tree — they still feed the per-`kind` cost calibration.
    """
    # The machine these measurements describe. A worker stamps its *own* fingerprint into the
    # document before shipping it, so a heterogeneous cluster's rows stay attributed to the
    # node that produced them; falling back to this process's fingerprint is right for every
    # single-node run and for a worker whose document predates the stamp.
    local_fingerprint = fingerprint()
    # How hard this process's CPU quota was biting while the work ran. A property of the
    # cgroup rather than of any one operator, so it is read once and stamped on every row of
    # the batch — the same shape as the fingerprint above, and for the same reason: it
    # describes the machine the measurements were taken on, not the operator that took them.
    # `None` (no cgroup, or unreadable counters) records `0.0`, which is "no evidence".
    local_throttled = cgroup_throttled_ratio() or 0.0
    # And whether the silicon clamped *itself* while the work ran. Read once per batch, and
    # a *delta* since the previous batch, so it describes this run rather than the machine's
    # whole uptime. Always `0` on a virtualized host, which does not expose the counters.
    local_thermal = cpu_thermal_events()
    for op in ops:
        rows_in = op.get("rows_in", 0)
        rows_out = op.get("rows_out", 0)
        op_id = int(op.get("op_id", 0))
        annotated = planned[op_id] if 0 <= op_id < len(planned) else None
        sink.record(
            OperatorFeedback(
                op_id=OpId(op_id),
                kind=op.get("kind", ""),
                n_actual=int(rows_out),
                t_op_ms=op.get("elapsed_ns", 0) / 1e6,
                m_peak_bytes=int(op.get("peak_bytes", 0)),
                selectivity=safe_div(rows_out, rows_in, 1.0),
                batch_size=batch_size,
                backend=op.get("backend", "interp"),
                algorithm="spill" if op.get("spilled") else "",
                spill_bytes=int(op.get("spill_bytes", 0) or 0),
                peak_rss_bytes=int(op.get("peak_rss_bytes", 0) or 0),
                cpu_utilization=cpu_utilization(
                    op.get("cpu_ns", 0),
                    op.get("elapsed_ns", 0),
                    op.get("threads", 1),
                    op.get("wall_span_ns", 0),
                ),
                threads=int(op.get("threads", 0) or 0),
                n_input=int(rows_in),
                n_build=int(op.get("rows_build", 0) or 0),
                result_bytes=int(op.get("result_bytes", op.get("peak_bytes", 0)) or 0),
                signature=_signature_of(annotated),
                n_estimated=_raw_estimate_of(annotated),
                expr_factor=annotated.properties.expr_factor if annotated else 1.0,
                # The engine flattens its hardware counters into the same document, so they
                # read as ordinary keys. `or 0` covers both an older engine that omits the key
                # and a platform that reports null for a counter it cannot read.
                minor_faults=int(op.get("minor_faults", 0) or 0),
                major_faults=int(op.get("major_faults", 0) or 0),
                vol_ctx_switches=int(op.get("vol_ctx_switches", 0) or 0),
                invol_ctx_switches=int(op.get("invol_ctx_switches", 0) or 0),
                io_read_bytes=int(op.get("io_read_bytes", 0) or 0),
                io_write_bytes=int(op.get("io_write_bytes", 0) or 0),
                hw_fingerprint=str(op.get("hw_fingerprint", "") or local_fingerprint),
                # A distributed worker stamps its own reading into the document before it
                # travels, exactly as it does the fingerprint, so a driver recording on its
                # behalf attributes the throttling to the node that suffered it.
                cpu_throttled_ratio=float(op.get("cpu_throttled_ratio") or local_throttled),
                cpu_thermal_events=int(op.get("cpu_thermal_events") or local_thermal),
            )
        )


def _signature_of(op: PhysicalOp | None) -> str:
    """The annotated operator's stable signature, or `""` when it is unavailable."""
    return op.properties.signature if op is not None else ""


def _raw_estimate_of(op: PhysicalOp | None) -> float:
    """The pre-correction row estimate, or `0.0` when unavailable/not a number."""
    if op is None:
        return 0.0
    raw = op.properties.est_rows_raw
    return float(raw) if raw == raw and raw > 0 else 0.0  # NaN-safe


class LocalExecutor:
    """Executes physical plans in-process via the native engine."""

    def __init__(self, feedback: FeedbackSink | None = None) -> None:
        self._feedback = feedback

    def execute(
        self,
        plan: PhysicalPlan,
        sources: list[list[pa.RecordBatch]],
    ) -> list[pa.RecordBatch]:
        # Import the native submodule directly (not `from batcher import _native`),
        # so Core never routes through the package root — keeping it independent of
        # the api/kyber/carbonite layers per the import contract.
        _native = engine()
        cfg = active_config()
        # Ship Kyber's per-operator spill budgets alongside the plan so the engine
        # budgets each stateful operator individually (not one global cap for all).
        engine_cfg = cfg.engine_config_json_with(
            plan.op_budgets(),
            prefer_materializing_aggregate=plan.prefer_materializing_aggregate,
        )
        # Collect per-operator metrics only when there is a sink to consume them;
        # the plain entry point avoids the (tiny) JSON serialization otherwise.
        _ensure_native_tracing(_native)
        # The id makes this execution cancellable. `""` outside a `query_scope` — which is
        # every non-terminal-op caller — and the engine then registers nothing and polls
        # nothing, so an uncancellable path costs exactly what it did before.
        query_id = current_query_id() or None
        if self._feedback is None:
            return _native.execute_plan(plan.to_json(), sources, engine_cfg, query_id)

        out, metrics_json = _native.execute_plan_metered(
            plan.to_json(), sources, engine_cfg, query_id
        )
        record_exec_metrics(self._feedback, metrics_json, cfg.execution.morsel_rows, plan.ops)
        return out


def execute_local(
    plan: PhysicalPlan,
    sources: list[list[pa.RecordBatch]],
    feedback: FeedbackSink | None = None,
) -> list[pa.RecordBatch]:
    """Convenience wrapper around `LocalExecutor.execute`."""
    return LocalExecutor(feedback).execute(plan, sources)


def execute_local_metered(
    plan: PhysicalPlan,
    sources: list[list[pa.RecordBatch]],
    feedback: FeedbackSink | None = None,
) -> tuple[list[pa.RecordBatch], list[dict], dict]:
    """Execute and return ``(batches, ops, usage)``.

    `ops` is *this run's* raw per-operator `ExecMetrics` (one dict per operator: ``op_id``,
    ``kind``, ``rows_in``, ``rows_out``, ``elapsed_ns``, ``peak_bytes``, ``spilled``,
    ``backend``, ``cpu_ns``, ``threads``). `usage` is the document's ``query`` block — what
    the whole execution cost the machine (CPU, peak RSS, faults, real block-device bytes).

    The two are separate because they hold on different tiers. Per-operator hardware
    counters are only sound where each operator owns an exclusive wall interval, which the
    streaming executor — the default — does not give them, so it reports zeros. The
    whole-execution reading is measured at the FFI boundary and holds everywhere, which
    makes it the one that answers "what did this cost" on the common path.

    Core *measures*; this is the same metered native call the feedback loop uses,
    surfaced directly so the control plane can report measured per-operator stats
    (`Dataset.stats()` / `explain(analyze=True)`). When `feedback` is supplied the
    same ops are also recorded into the sink, so a profiled run still feeds the
    learning loop. A malformed/empty metrics document yields an empty `ops` list and an
    empty `usage` rather than raising.
    """
    _native = engine()
    # The same tracing bridge `LocalExecutor.execute` installs. Without it a session whose
    # queries are *all* profiled (`stats()` / `explain(analyze=True)`) — which is exactly
    # when data-plane traces are wanted — never got any, because this second entry point to
    # the same native call silently skipped the install.
    _ensure_native_tracing(_native)
    cfg = active_config()
    out, metrics_json = _native.execute_plan_metered(
        plan.to_json(),
        sources,
        # The routing hint travels with the budgets, exactly as `LocalExecutor.execute`
        # ships it. Dropping it here silently disarmed the whole optimization on the
        # ordinary path: `collect()` reaches the engine through *this* entry point, so
        # a plan Kyber had marked as a materializing aggregate was executed as a
        # streaming one anyway — and only a run that took the unmetered wrapper ever
        # saw the faster route.
        cfg.engine_config_json_with(
            plan.op_budgets(),
            prefer_materializing_aggregate=plan.prefer_materializing_aggregate,
        ),
        current_query_id() or None,
    )
    try:
        doc = json.loads(metrics_json)
        ops = doc.get("ops", [])
        usage = doc.get("query") or {}
    except (ValueError, TypeError):
        ops, usage = [], {}
    if feedback is not None and ops:
        _record_op_feedback(feedback, ops, cfg.execution.morsel_rows, plan.ops)
    return out, ops, usage
