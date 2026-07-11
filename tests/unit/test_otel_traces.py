"""The engine emits its execution profile as OpenTelemetry spans, when configured.

Batcher produces spans into the host's globally-configured tracer; the enterprise owns the
exporter. These tests stand in an in-memory exporter (what an OTLP setup would be) and
assert: a query span with per-operator child spans carrying the measured facts, nothing
when disabled, and correctness-neutrality (a broken tracer never fails the query).
"""

from __future__ import annotations

import dataclasses

import pytest

import batcher as bt
from batcher.config import active_config, config_context

pytestmark = pytest.mark.unit

pytest.importorskip("opentelemetry.sdk", reason="OpenTelemetry SDK not installed")

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)


@pytest.fixture(scope="module", autouse=True)
def _provider():
    """An in-memory exporter attached to the live provider.

    OTel allows the global provider to be set only once per process, so if another test (or
    Ray) already set one, attach our span processor to *it* rather than fighting the
    one-shot setter. Either way our exporter ends up on the provider `get_tracer` resolves.
    """
    exporter = InMemorySpanExporter()
    current = trace.get_tracer_provider()
    if hasattr(current, "add_span_processor"):
        current.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture
def exporter(_provider):
    _provider.clear()
    return _provider


def _traced_config(enabled: bool):
    cfg = active_config()
    obs = dataclasses.replace(cfg.observability, otel_traces=enabled, event_log=False)
    return dataclasses.replace(cfg, observability=obs)


def _run_a_grouped_query():
    return (
        bt.from_pydict({"g": ["a", "a", "b"], "v": [1, 2, 3]})
        .group_by("g")
        .agg(s=bt.col("v").sum())
        .to_pydict()
    )


def test_a_query_emits_a_query_span_with_operator_children(exporter):
    with config_context(_traced_config(True)):
        _run_a_grouped_query()
    names = {s.name for s in exporter.get_finished_spans()}
    assert "batcher.query" in names
    assert "batcher.op.aggregate" in names
    assert "batcher.op.scan" in names


def test_the_query_span_carries_the_top_line_facts(exporter):
    with config_context(_traced_config(True)):
        _run_a_grouped_query()
    query = next(s for s in exporter.get_finished_spans() if s.name == "batcher.query")
    attrs = dict(query.attributes)
    assert attrs["batcher.rows"] == 2  # two groups
    assert attrs["batcher.distributed"] is False
    assert attrs["batcher.spilled"] is False
    assert attrs["batcher.bottleneck.kind"] == "aggregate"


def test_operator_spans_carry_measured_row_counts(exporter):
    with config_context(_traced_config(True)):
        _run_a_grouped_query()
    ops = {
        s.name: dict(s.attributes)
        for s in exporter.get_finished_spans()
        if s.name.startswith("batcher.op.")
    }
    assert ops["batcher.op.scan"]["batcher.op.rows_out"] == 3
    assert ops["batcher.op.aggregate"]["batcher.op.rows_out"] == 2


def test_no_spans_are_emitted_when_disabled(exporter):
    with config_context(_traced_config(False)):
        _run_a_grouped_query()
    assert exporter.get_finished_spans() == ()


def test_the_query_result_is_unaffected_by_tracing(exporter):
    with config_context(_traced_config(True)):
        traced = _run_a_grouped_query()
    with config_context(_traced_config(False)):
        untraced = _run_a_grouped_query()
    assert traced == untraced


def test_a_broken_exporter_never_fails_the_query(_provider, monkeypatch):
    """Telemetry is best-effort: an exporter that raises must not break execution."""
    from batcher.api.terminal import otel

    def boom(_profile):
        raise RuntimeError("exporter down")

    monkeypatch.setattr(otel, "_emit", boom)
    with config_context(_traced_config(True)):
        assert _run_a_grouped_query() == {"g": ["a", "b"], "s": [3, 3]}
