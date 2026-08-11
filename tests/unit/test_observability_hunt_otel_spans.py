"""Observability hunt: OpenTelemetry span emission must carry per-operator detail on
BOTH the single-node and the distributed path.

Regression: `_emit` iterated only `profile.ops` (the driver tree). On a distributed run
those operators are unmeasured — the measured facts live in `profile.worker_ops` (the
map sub-plan) — so a distributed query traced as a bare `batcher.query` span with no
operator spans at all, though `render()`/`stats()` both surface the worker ops.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("opentelemetry.sdk")


def _exporter():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    # A provider is set process-wide once; add our exporter to whatever is active.
    if not hasattr(provider, "add_span_processor"):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


def _emit(collector):
    from batcher.api.terminal.otel import emit_query_spans
    from batcher.config import active_config, set_config

    prev = active_config()
    set_config(
        prev.replace(observability=prev.observability.__class__(otel_traces=True, event_log=False))
    )
    try:
        profile = collector.to_profile(total_ms=10.0, rows=1, query_id="q")
        emit_query_spans(profile)
    finally:
        set_config(prev)


def test_single_node_emits_query_and_operator_spans():
    from batcher.plan.profile import ProfileCollector

    exporter = _exporter()
    c = ProfileCollector()
    c.optimized_ir = {"op": "filter", "input": {"op": "scan", "source_id": 0}}
    c.metric_ops = [
        {"op_id": 0, "kind": "filter", "rows_in": 3, "rows_out": 1, "elapsed_ns": 5000},
        {"op_id": 1, "kind": "scan", "rows_in": 3, "rows_out": 3, "elapsed_ns": 2000},
    ]
    _emit(c)
    names = [s.name for s in exporter.get_finished_spans()]
    assert "batcher.query" in names
    assert names.count("batcher.op.filter") == 1
    assert names.count("batcher.op.scan") == 1


def test_distributed_emits_worker_operator_spans():
    """The fix: a distributed query's measured worker ops must become operator spans."""
    from batcher.plan.profile import ProfileCollector

    exporter = _exporter()
    c = ProfileCollector()
    c.optimized_ir = {"op": "aggregate", "input": {"op": "scan", "source_id": 0}}
    c.distributed = True
    # One whole `ExecMetrics` document per worker, not a bare op-list: the document also
    # carries the `query` block, which is where a worker's share of the CPU, memory and
    # disk cost lives and is unrecoverable once the driver has dropped the rest of it.
    c.worker_metrics = [
        {
            "ops": [
                {
                    "op_id": 0,
                    "kind": "aggregate",
                    "rows_in": 100,
                    "rows_out": 5,
                    "elapsed_ns": 9000,
                    "result_bytes": 256,
                    "backend": "interp",
                }
            ],
            "query": {"wall_ns": 9_000, "cpu_ns": 36_000},
        }
    ]
    _emit(c)
    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    # Before the fix: only "batcher.query" — the measured aggregate was dropped.
    assert "batcher.op.aggregate" in names, f"worker op span missing; got {names}"
    worker = next(s for s in spans if s.name == "batcher.op.aggregate")
    assert worker.attributes.get("batcher.op.scope") == "worker"
    assert worker.attributes.get("batcher.op.rows_out") == 5
    # And the worker's share of what the run cost the machine reaches the query span, so a
    # tracing backend can filter on it (`batcher.cores_busy < 2` finds a query that failed
    # to parallelize). Without it a distributed trace shows where the time went and never
    # whether the cluster had the cores to spend.
    query_span = next(s for s in spans if s.name == "batcher.query")
    assert query_span.attributes.get("batcher.cpu_ms") == 0.036
    assert query_span.attributes.get("batcher.cores_busy") == 4.0
