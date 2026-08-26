"""Batcher's OpenTelemetry spans must be usable as spans, not just as attribute bags.

Two defects, both of which made the trace backend the least useful of the four
observability surfaces:

* Emission happens *after* the query has run, so a span opened and closed at emit time
  lasted as long as it took to set a dozen attributes. Every Batcher span arrived as a
  zero-width tick, and a waterfall — the reason to emit spans at all — showed nothing.
* A query that raised produced no span. The one class of run an operator most wants to
  find in a trace backend was the only class that was never in it, and a latency
  histogram built from these spans silently excluded every timeout.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("opentelemetry.sdk")

from batcher.plan.profile import OpProfile, QueryProfile  # noqa: E402


@pytest.fixture
def spans():
    """An in-memory exporter, with OTel emission switched on for the test's duration."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from batcher.config import active_config, set_config

    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    if not hasattr(provider, "add_span_processor"):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    prev = active_config()
    set_config(
        prev.replace(observability=prev.observability.__class__(otel_traces=True, event_log=False))
    )
    try:
        yield exporter
    finally:
        set_config(prev)


def _ms(span) -> float:
    return (span.end_time - span.start_time) / 1e6


def _profile() -> QueryProfile:
    ops = (
        OpProfile(op_id=0, kind="aggregate", depth=0, measured=True, rows_out=10, elapsed_ms=200.0),
        OpProfile(op_id=1, kind="scan", depth=1, measured=True, rows_out=100, elapsed_ms=50.0),
    )
    return QueryProfile(ops=ops, total_ms=400.0, rows=10, measured=True, query_id="q1")


def test_the_query_span_covers_the_querys_real_interval(spans):
    from batcher.api.terminal.otel import emit_query_spans

    emit_query_spans(_profile())
    query = next(s for s in spans.get_finished_spans() if s.name == "batcher.query")
    assert _ms(query) == pytest.approx(400.0, abs=1.0)


def test_each_operator_span_carries_its_measured_duration(spans):
    from batcher.api.terminal.otel import emit_query_spans

    emit_query_spans(_profile())
    by_name = {s.name: s for s in spans.get_finished_spans()}
    assert _ms(by_name["batcher.op.aggregate"]) == pytest.approx(200.0, abs=1.0)
    assert _ms(by_name["batcher.op.scan"]) == pytest.approx(50.0, abs=1.0)


def test_operator_spans_start_with_the_query_rather_than_being_laid_out_end_to_end(spans):
    """The profile records a duration per operator and no start offset.

    Laying them out sequentially would look more like a waterfall and would be a
    fabrication — and on the streaming tier, where operators genuinely interleave, it
    would be a wrong one.
    """
    from batcher.api.terminal.otel import emit_query_spans

    emit_query_spans(_profile())
    finished = {s.name: s for s in spans.get_finished_spans()}
    starts = {finished[name].start_time for name in ("batcher.op.aggregate", "batcher.op.scan")}
    assert len(starts) == 1
    assert starts.pop() == finished["batcher.query"].start_time


def test_operator_spans_are_children_of_the_query_span(spans):
    from batcher.api.terminal.otel import emit_query_spans

    emit_query_spans(_profile())
    finished = {s.name: s for s in spans.get_finished_spans()}
    query_id = finished["batcher.query"].context.span_id
    for name in ("batcher.op.aggregate", "batcher.op.scan"):
        assert finished[name].parent.span_id == query_id


def test_a_failed_query_produces_an_error_span_with_the_exception_on_it(spans):
    from batcher.api.terminal.otel import emit_failure_span

    emit_failure_span("q-fail", 12.5, ValueError("unknown column 'nope'"))
    span = next(s for s in spans.get_finished_spans() if s.name == "batcher.query")
    assert span.attributes["batcher.ok"] is False
    assert span.attributes["batcher.query_id"] == "q-fail"
    assert _ms(span) == pytest.approx(12.5, abs=1.0)
    assert span.status.status_code.name == "ERROR"
    assert any(event.name == "exception" for event in span.events)


def test_report_failure_routes_a_raised_query_into_the_trace(spans):
    """The wiring, not just the emitter: `report_failure` is the one funnel every failed
    terminal op passes through, and it used to publish to the bus and stop there."""
    from batcher.api.terminal.event_log import report_failure

    report_failure("q-wired", total_ms=5.0, exc=RuntimeError("boom"))
    names = [s.name for s in spans.get_finished_spans()]
    assert "batcher.query" in names


def test_emission_is_a_no_op_when_the_feature_is_off():
    from batcher.api.terminal.otel import emit_failure_span, emit_query_spans, otel_enabled

    assert otel_enabled() is False
    emit_query_spans(_profile())
    emit_failure_span("q", 1.0, ValueError("x"))
