"""Emit a query's execution profile as OpenTelemetry spans.

An enterprise runs its own observability stack (an OTLP collector, a tracing backend);
Batcher's job is to *produce* spans, not to own an exporter. This module emits one span
per query, with a child span per operator carrying the same measured facts the event log
records — rows, time, memory, spill, backend — so the query shows up in the same traces
as the rest of the platform, correlated by trace context the host app already propagates.

It uses only the OpenTelemetry **API**, whose global no-op tracer makes this free when no
SDK/provider is installed. So there is no hard dependency: without `opentelemetry` the
import guard makes the emit a no-op, and with it but *unconfigured* the API's no-op tracer
drops the spans. Spans flow only once the host has configured a provider + exporter — the
enterprise's decision, not the engine's.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from batcher.plan.profile import QueryProfile

__all__ = ["emit_failure_span", "emit_query_spans", "otel_enabled"]

_INSTRUMENTATION = "batcher"
# Whether `opentelemetry` is importable — resolved once (it cannot change mid-process). The
# *tracer itself* is fetched fresh per query, not cached, so a provider the host configures
# after the first query is still honored (a cached tracer would bind to the old provider).
_OTEL_AVAILABLE: bool | None = None


def _tracer() -> object | None:
    """A fresh OTel tracer for this emit, or None if `opentelemetry` is not installed."""
    global _OTEL_AVAILABLE
    if _OTEL_AVAILABLE is False:
        return None
    try:
        from opentelemetry import trace

        _OTEL_AVAILABLE = True
        return trace.get_tracer(_INSTRUMENTATION)
    except ImportError:
        _OTEL_AVAILABLE = False
        return None


def otel_enabled() -> bool:
    """Whether OTel span emission is on (config flag) *and* `opentelemetry` is available."""
    from batcher.config import active_config

    return active_config().observability.otel_traces and _tracer() is not None


def emit_query_spans(profile: QueryProfile) -> None:
    """Emit `profile` as one query span with a child span per operator.

    Best-effort and correctness-neutral: any error (a misbehaving exporter, a missing
    provider) is swallowed so observability never breaks a query. A no-op when disabled or
    when `opentelemetry` is absent.

    Args:
        profile: The measured profile of a query that has finished.

    Returns:
        None.
    """
    if not otel_enabled():
        return
    tracer = _tracer()
    try:
        _emit(tracer, profile)
    except Exception:  # pragma: no cover - telemetry must never fail a query
        from batcher._internal.logging import get_logger

        get_logger("api").debug("otel span emit failed", exc_info=True)


def _emit(tracer: object, profile: QueryProfile) -> None:
    # A query span carrying the top-line facts, then one child span per operator so the
    # bottleneck and any spill are visible in a trace waterfall exactly as in `explain`.
    #
    # **The spans are given explicit timestamps.** Emission happens *after* the query has
    # run, so a span opened and closed here lasts as long as it takes to set a dozen
    # attributes — microseconds — and every Batcher span arrived in the backend as a
    # zero-width tick. A waterfall is the reason to emit spans at all, and it showed
    # nothing. The query span is placed over its real interval, and each operator span is
    # given its real measured duration.
    #
    # What is *not* reconstructed is where each operator sat inside that interval: the
    # profile records a duration per operator and no start offset, so every operator span
    # begins at the query's start. Laying them out end to end would look more like a
    # waterfall and would be a fabrication — and on the streaming tier, where operators
    # genuinely interleave, it would be a wrong one.
    end_ns = time.time_ns()
    start_ns = end_ns - _to_ns(profile.total_ms)
    with tracer.start_as_current_span(  # type: ignore[attr-defined]
        "batcher.query", start_time=start_ns, end_on_exit=False
    ) as query_span:
        query_span.set_attribute("batcher.query_id", profile.query_id)
        query_span.set_attribute("batcher.rows", profile.rows)
        query_span.set_attribute("batcher.total_ms", profile.total_ms)
        query_span.set_attribute("batcher.distributed", profile.distributed)
        query_span.set_attribute("batcher.spilled", profile.spilled)
        bottleneck = profile.bottleneck
        if bottleneck is not None:
            query_span.set_attribute("batcher.bottleneck.kind", bottleneck.kind)
            query_span.set_attribute("batcher.bottleneck.op_id", bottleneck.op_id)
        _set_usage(query_span, profile.usage)
        for op in profile.ops:
            if op.measured:
                _emit_op(tracer, op, start_ns=start_ns)
        # On the distributed path the driver tree is unmeasured — the measured per-operator
        # facts live in the worker map sub-plan (a separate op-id space). Emit those as child
        # spans too, or a distributed query (the one whose operators matter most) would trace
        # as a bare query span with no operator detail, unlike `render()` / `stats()` which
        # both surface the worker ops.
        for op in profile.worker_ops:
            _emit_op(tracer, op, scope="worker", start_ns=start_ns)
    query_span.end(end_ns)


def _to_ns(ms: float) -> int:
    """Milliseconds as whole nanoseconds, never negative."""
    return max(0, int(ms * 1_000_000))


def _set_usage(span: object, usage) -> None:
    """Attach what the run cost the machine to the query span.

    A trace waterfall shows *where* the time went and cannot show whether the box had the
    cores to spend — the same plan reads identically at 1 core busy and at 30. These are the
    attributes that separate the two, and a tracing backend can filter on them directly
    (``batcher.cores_busy < 2`` finds every query that failed to parallelize).

    Skipped entirely when the platform reported nothing, because a span carrying
    ``cpu_ms = 0`` asserts the query used no CPU, which is a different claim from
    "unmeasured" and the one a reader will act on.
    """
    if not usage.measured:
        return
    span.set_attribute("batcher.cpu_ms", usage.cpu_ms)  # type: ignore[attr-defined]
    span.set_attribute("batcher.execution_ms", usage.wall_ms)  # type: ignore[attr-defined]
    span.set_attribute("batcher.cores_busy", usage.cores_busy)  # type: ignore[attr-defined]
    span.set_attribute("batcher.peak_rss_bytes", usage.peak_rss_bytes)  # type: ignore[attr-defined]
    span.set_attribute("batcher.major_faults", usage.major_faults)  # type: ignore[attr-defined]
    span.set_attribute("batcher.io_read_bytes", usage.io_read_bytes)  # type: ignore[attr-defined]
    span.set_attribute("batcher.io_write_bytes", usage.io_write_bytes)  # type: ignore[attr-defined]


def _emit_op(tracer: object, op, *, scope: str = "driver", start_ns: int = 0) -> None:
    """One operator's measured facts as a child span, over its measured duration.

    `scope` distinguishes the driver-tree operators from the distributed map sub-plan's
    worker operators (a separate op-id space), so a trace consumer can tell them apart.
    """
    with tracer.start_as_current_span(  # type: ignore[attr-defined]
        f"batcher.op.{op.kind}", start_time=start_ns, end_on_exit=False
    ) as span:
        span.set_attribute("batcher.op.id", op.op_id)
        span.set_attribute("batcher.op.kind", op.kind)
        span.set_attribute("batcher.op.scope", scope)
        span.set_attribute("batcher.op.rows_in", op.rows_in)
        span.set_attribute("batcher.op.rows_out", op.rows_out)
        span.set_attribute("batcher.op.elapsed_ms", op.elapsed_ms)
        span.set_attribute("batcher.op.result_bytes", op.result_bytes)
        span.set_attribute("batcher.op.spilled", op.spilled)
        # The magnitude, not just the fact: a 1 GB spill and a 100 GB one are the same
        # boolean and very different incidents.
        if op.spill_bytes:
            span.set_attribute("batcher.op.spill_bytes", op.spill_bytes)
        if op.cpu_ms:
            span.set_attribute("batcher.op.cpu_ms", op.cpu_ms)
            span.set_attribute("batcher.op.threads", op.threads)
        if op.backend:
            span.set_attribute("batcher.op.backend", op.backend)
    span.end(start_ns + _to_ns(op.elapsed_ms))


def emit_failure_span(query_id: str, total_ms: float, exc: BaseException) -> None:
    """Emit a span for a query that raised, with the exception recorded on it.

    Without this a failing query produced no span at all, so the one class of run an
    operator most wants to find in a trace backend was the one class that was never there —
    and a latency histogram built from these spans silently excluded every timeout.

    Best-effort and correctness-neutral: any error here is swallowed, and the original
    exception is neither caught nor altered by the caller.

    Args:
        query_id: The id the event log assigned.
        total_ms: Wall time from execution start to the failure.
        exc: The exception being raised.

    Returns:
        None.
    """
    if not otel_enabled():
        return
    tracer = _tracer()
    try:
        from opentelemetry.trace import Status, StatusCode

        end_ns = time.time_ns()
        span = tracer.start_span(  # type: ignore[attr-defined]
            "batcher.query", start_time=end_ns - _to_ns(total_ms)
        )
        span.set_attribute("batcher.query_id", query_id)
        span.set_attribute("batcher.total_ms", total_ms)
        span.set_attribute("batcher.ok", False)
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
        span.end(end_ns)
    except Exception:  # pragma: no cover - telemetry must never fail a query
        from batcher._internal.logging import get_logger

        get_logger("api").debug("otel failure span emit failed", exc_info=True)
