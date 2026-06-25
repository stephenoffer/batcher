"""The Core local executor.

Drives a `PhysicalPlan` to completion against in-memory input relations by
handing the lowered IR to the native engine, then transcribes the native
engine's per-operator `ExecMetrics` into `OperatorFeedback` for the MetadataHub.
Core *measures* — it does not optimize: it faithfully reports what the data plane
observed (rows in/out, time, peak bytes, spill, backend), keyed by the same
pre-order operator id Kyber assigns in `_annotate_ops`. The morsel scheduler, JIT
tier-up, and the `bc-adapt` re-optimization loop replace the single
`execute_plan_metered` call without changing this interface.
"""

from __future__ import annotations

import json

import pyarrow as pa

from batcher.config import active_config
from batcher.plan.feedback import FeedbackSink, OperatorFeedback
from batcher.plan.ids import OpId
from batcher.plan.physical import PhysicalPlan

__all__ = ["LocalExecutor", "execute_local", "execute_local_metered", "record_exec_metrics"]


def record_exec_metrics(sink: FeedbackSink | None, metrics_json: str, batch_size: int) -> None:
    """Transcribe a native `ExecMetrics` document into per-operator `OperatorFeedback`.

    The one place the engine's measured runtime facts (rows in/out, time, peak
    bytes, spill, backend) become feedback — shared by the single-node executor and
    the distributed workers (which run sub-plans and ship their metrics back to the
    driver). Calibration buckets by operator `kind`, so the sub-plan-local `op_id`s a
    distributed worker reports need no global correlation. Best-effort: a malformed
    or empty document drops silently rather than failing the query.
    """
    if sink is None:
        return
    try:
        ops = json.loads(metrics_json).get("ops", [])
    except (ValueError, TypeError):
        return
    for op in ops:
        rows_in = op.get("rows_in", 0)
        rows_out = op.get("rows_out", 0)
        sink.record(
            OperatorFeedback(
                op_id=OpId(int(op.get("op_id", 0))),
                kind=op.get("kind", ""),
                n_actual=int(rows_out),
                t_op_ms=op.get("elapsed_ns", 0) / 1e6,
                m_peak_bytes=int(op.get("peak_bytes", 0)),
                selectivity=(rows_out / rows_in) if rows_in else 1.0,
                batch_size=batch_size,
                backend=op.get("backend", "interp"),
                algorithm="spill" if op.get("spilled") else "",
                cpu_utilization=_cpu_utilization(
                    op.get("cpu_ns", 0), op.get("elapsed_ns", 0), op.get("threads", 1)
                ),
            )
        )


def _cpu_utilization(cpu_ns: float, elapsed_ns: float, threads: int) -> float:
    """Mean fraction of allocated cores kept busy, clamped to [0, 1].

    `cpu_ns` is CPU-time summed across all worker threads during the operator;
    dividing by ``elapsed_ns x threads`` (the engine's *actual* live thread count,
    not a guessed host core count — which is wrong under a cgroup CPU quota) gives
    the per-core busy fraction. 0.0 when the engine reported no CPU time (older
    build), no wall time, or no thread count.
    """
    if cpu_ns <= 0 or elapsed_ns <= 0 or threads <= 0:
        return 0.0
    return min(1.0, cpu_ns / (elapsed_ns * threads))


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
        import batcher._native as _native

        cfg = active_config()
        # Ship Kyber's per-operator spill budgets alongside the plan so the engine
        # budgets each stateful operator individually (not one global cap for all).
        engine_cfg = cfg.engine_config_json_with(plan.op_budgets())
        # Collect per-operator metrics only when there is a sink to consume them;
        # the plain entry point avoids the (tiny) JSON serialization otherwise.
        if self._feedback is None:
            return _native.execute_plan(plan.to_json(), sources, engine_cfg)

        out, metrics_json = _native.execute_plan_metered(plan.to_json(), sources, engine_cfg)
        record_exec_metrics(self._feedback, metrics_json, cfg.execution.morsel_rows)
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
) -> tuple[list[pa.RecordBatch], list[dict]]:
    """Execute and return ``(batches, ops)`` where `ops` is *this run's* raw
    per-operator `ExecMetrics` (one dict per operator: ``op_id``, ``kind``,
    ``rows_in``, ``rows_out``, ``elapsed_ns``, ``peak_bytes``, ``spilled``,
    ``backend``).

    Core *measures*; this is the same metered native call the feedback loop uses,
    surfaced directly so the control plane can report measured per-operator stats
    (`Dataset.stats()`). A malformed/empty metrics document yields an empty `ops`
    list rather than raising.
    """
    import batcher._native as _native

    cfg = active_config()
    out, metrics_json = _native.execute_plan_metered(
        plan.to_json(), sources, cfg.engine_config_json_with(plan.op_budgets())
    )
    try:
        ops = json.loads(metrics_json).get("ops", [])
    except (ValueError, TypeError):
        ops = []
    return out, ops
