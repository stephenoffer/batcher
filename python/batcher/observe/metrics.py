"""Process-wide counters and timings, as a plain dict.

The event bus carries everything the engine does, but subscribing to it means learning its
kinds, its field names, and its threading rules. Most people exporting metrics want none of
that: they want a flat dict of numbers to hand to Prometheus, OpenTelemetry, StatsD, or a
log line, once a minute, forever.

That is this module. A single bus sink folds every query into a fixed set of counters and
histograms, `metrics_snapshot` returns them as nested plain data, and `prometheus_text`
renders the same numbers in the Prometheus text exposition format. The sink costs a few
integer adds per event and holds a bounded amount of state regardless of how many queries
run.

Collection starts on the first snapshot, or on an explicit `start_metrics()`. It is not
always-on because attaching *any* bus sink signals to the engine that per-query profiles
are being consumed, which makes it assemble one on every query — a cost a process that
exports no metrics should not pay.

Counters are cumulative from that point, like every counter-based metrics system, so a
monitoring backend can compute rates by differencing. `reset_metrics` exists for tests and
for a long-lived service that would rather report per-interval numbers itself.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from batcher._internal import events

__all__ = ["metrics_snapshot", "prometheus_text", "reset_metrics", "start_metrics"]

# Duration buckets in milliseconds, Prometheus-style cumulative histogram boundaries.
# Chosen to straddle Batcher's stated range: sub-millisecond planning through multi-minute
# distributed jobs, roughly one bucket per half order of magnitude.
_BUCKETS_MS: tuple[float, ...] = (1, 5, 10, 50, 100, 500, 1_000, 5_000, 30_000, 300_000)


class _Collector:
    """Folds bus events into counters. One instance per process, guarded by one lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Zero every counter and restart the uptime clock."""
        with self._lock:
            self._started = time.time()
            self.queries_total = 0
            self.queries_failed = 0
            self.rows_out_total = 0
            self.rows_scanned_total = 0
            self.bytes_scanned_total = 0
            self.spills_total = 0
            self.query_ms_total = 0.0
            self.query_ms_max = 0.0
            self._buckets: dict[float, int] = dict.fromkeys(_BUCKETS_MS, 0)
            self._op_count: dict[str, int] = defaultdict(int)
            self._op_ms: dict[str, float] = defaultdict(float)
            self._op_rows: dict[str, int] = defaultdict(int)
            self._log_counts: dict[str, int] = defaultdict(int)
            # Distributed / inference counters. These are cumulative like the rest, so a
            # scraper differences them for rates; the live per-stage view is the separate
            # `InferenceProgress` store. Cardinality is bounded by hardware (GPU devices) and
            # by the plan (skip reasons), never by run length, so a 12-hour job holds these
            # flat.
            self.partitions_done_total = 0
            self.skipped_total = 0
            self.infer_batches_total = 0
            self.infer_rows_total = 0
            self.infer_latency_ms_total = 0.0
            self.infer_blocked_ms_total = 0.0
            self._skipped_by_reason: dict[str, int] = defaultdict(int)
            self._gpu: dict[str, dict[str, float]] = {}
            self.gpu_util_pct_max = 0.0

    def handle(self, event: events.Event) -> None:
        """Fold one event in. Hot path: a dict lookup and a few adds, no allocation."""
        kind = event.kind
        fields = event.fields
        with self._lock:
            if kind == events.QUERY_END:
                self._end_query(fields)
            elif kind == events.STAGE_END:
                name = event.name or "unknown"
                self._op_count[name] += 1
                self._op_ms[name] += float(fields.get("elapsed_ms", 0.0))
                self._op_rows[name] += int(fields.get("rows_out", 0))
                if fields.get("spilled"):
                    self.spills_total += 1
            elif kind == events.PROGRESS:
                self.rows_scanned_total += int(fields.get("rows", 0))
                self.bytes_scanned_total += int(fields.get("bytes", 0))
            elif kind == events.LOG:
                self._log_counts[str(fields.get("level", "INFO"))] += 1
            elif kind == events.PARTITION:
                self.partitions_done_total += 1
            elif kind == events.INFER:
                self.infer_batches_total += 1
                self.infer_rows_total += int(fields.get("rows", 0))
                self.infer_latency_ms_total += float(fields.get("latency_ms", 0.0))
                self.infer_blocked_ms_total += float(fields.get("blocked_ms", 0.0))
            elif kind == events.SKIPPED:
                count = int(fields.get("count", 0))
                self.skipped_total += count
                reason = str(fields.get("reason", "read_error"))
                # Fold an unseen reason into "other" once the map is full, so a per-row unique
                # reason cannot grow it without bound over a long run.
                if reason not in self._skipped_by_reason and len(self._skipped_by_reason) >= 64:
                    reason = "other"
                self._skipped_by_reason[reason] += count
            elif kind == events.GPU:
                self._record_gpu(fields)

    def _record_gpu(self, fields: dict[str, Any]) -> None:
        """Fold one GPU sample in as a per-device gauge. Assumes the lock is held."""
        device = str(fields.get("device", fields.get("actor", "gpu0")))
        util = float(fields.get("util_pct", 0.0))
        self.gpu_util_pct_max = max(self.gpu_util_pct_max, util)
        self._gpu[device] = {
            "util_pct": util,
            "mem_used_bytes": float(fields.get("mem_used_bytes", 0)),
            "mem_total_bytes": float(fields.get("mem_total_bytes", 0)),
        }

    def _end_query(self, fields: dict[str, Any]) -> None:
        """Record a finished query. Assumes the lock is held."""
        self.queries_total += 1
        if not fields.get("ok", True):
            self.queries_failed += 1
        self.rows_out_total += int(fields.get("rows", 0))
        elapsed = float(fields.get("total_ms", 0.0))
        self.query_ms_total += elapsed
        self.query_ms_max = max(self.query_ms_max, elapsed)
        for edge in _BUCKETS_MS:
            if elapsed <= edge:
                self._buckets[edge] += 1

    def snapshot(self) -> dict[str, Any]:
        """A consistent, deep-copied view of every counter. Assumes nothing about callers."""
        with self._lock:
            ok = self.queries_total - self.queries_failed
            return {
                "uptime_seconds": time.time() - self._started,
                "queries": {
                    "total": self.queries_total,
                    "succeeded": ok,
                    "failed": self.queries_failed,
                    "duration_ms_total": self.query_ms_total,
                    "duration_ms_max": self.query_ms_max,
                    "duration_ms_mean": (
                        self.query_ms_total / self.queries_total if self.queries_total else 0.0
                    ),
                    "duration_ms_buckets": dict(self._buckets),
                },
                "rows": {
                    "scanned_total": self.rows_scanned_total,
                    "out_total": self.rows_out_total,
                },
                "bytes": {"scanned_total": self.bytes_scanned_total},
                "spills": {"total": self.spills_total},
                "operators": {
                    name: {
                        "count": count,
                        "elapsed_ms_total": self._op_ms[name],
                        "rows_out_total": self._op_rows[name],
                    }
                    for name, count in sorted(self._op_count.items())
                },
                "logs": dict(sorted(self._log_counts.items())),
                "partitions": {"done_total": self.partitions_done_total},
                "skipped": {
                    "total": self.skipped_total,
                    "by_reason": dict(sorted(self._skipped_by_reason.items())),
                },
                "inference": {
                    "batches_total": self.infer_batches_total,
                    "rows_total": self.infer_rows_total,
                    "latency_ms_total": self.infer_latency_ms_total,
                    "latency_ms_mean": (
                        self.infer_latency_ms_total / self.infer_batches_total
                        if self.infer_batches_total
                        else 0.0
                    ),
                    "blocked_ms_total": self.infer_blocked_ms_total,
                },
                "gpu": {
                    "util_pct_max": self.gpu_util_pct_max,
                    "devices": {device: dict(stats) for device, stats in sorted(self._gpu.items())},
                },
            }


_collector = _Collector()
_detach: Callable[[], None] | None = None
_attach_lock = threading.Lock()


def start_metrics() -> None:
    """Begin collecting engine counters, so later snapshots include everything since.

    Counting is opt-in rather than always-on, and deliberately so. Attaching any sink to
    the event bus tells the engine that something is consuming per-query profiles, which
    makes it assemble one on every query — a real cost on the small-query path that a
    process exporting no metrics should not pay.

    `metrics_snapshot` and `prometheus_text` call this for you, so the usual path needs
    nothing. Call it explicitly at startup when you want the first scrape to include the
    queries that ran before it. Idempotent and cheap on the repeat path.

    Examples:
        .. doctest::

            >>> from batcher.observe import metrics_snapshot, start_metrics
            >>> start_metrics()
            >>> metrics_snapshot()["queries"]["total"] >= 0
            True

    Returns:
        None.
    """
    global _detach
    if _detach is not None:
        return
    with _attach_lock:
        if _detach is None:
            _detach = events.subscribe(_collector.handle)


def metrics_snapshot() -> dict[str, Any]:
    """Every engine counter and timing as a nested dict of plain numbers.

    The dependency-free metrics export: no Batcher types, no event-bus knowledge, nothing
    that needs closing. Counters are cumulative since collection started, so a scrape loop
    differences successive snapshots to get rates. The first call starts collection, so
    call `start_metrics` at startup if the first scrape should cover earlier queries.

    The top-level keys are ``uptime_seconds``, ``queries`` (counts plus a duration
    histogram), ``rows``, ``bytes``, ``spills``, ``operators`` (per operator kind), ``logs``
    (records per level), and — for a distributed or batch-inference job — ``partitions``,
    ``skipped`` (dropped rows under ``on_read_error="skip"``), ``inference`` (batches, rows,
    and latency), and ``gpu`` (peak plus per-device utilization and VRAM).

    Examples:
        .. doctest::

            >>> from batcher.observe import metrics_snapshot
            >>> snap = metrics_snapshot()
            >>> sorted(snap)  # doctest: +NORMALIZE_WHITESPACE
            ['bytes', 'gpu', 'inference', 'logs', 'operators', 'partitions', 'queries',
             'rows', 'skipped', 'spills', 'uptime_seconds']
            >>> snap["queries"]["total"] >= 0
            True

    Returns:
        A nested dict of counters, safe to `json.dumps` and to mutate.
    """
    start_metrics()
    return _collector.snapshot()


def prometheus_text() -> str:
    """The same counters rendered in the Prometheus text exposition format.

    Serve this from your application's existing ``/metrics`` endpoint and Batcher's numbers
    join whatever you already scrape. Batcher owns no HTTP server for this and pulls in no
    client library; the string is built directly, so there is nothing to configure.

    Every series is prefixed ``batcher_``, counters carry the conventional ``_total``
    suffix, and query duration is exported as a real histogram with ``_bucket``, ``_sum``,
    and ``_count`` series.

    Examples:
        .. doctest::

            >>> from batcher.observe import prometheus_text
            >>> "batcher_queries_total" in prometheus_text()
            True

    Returns:
        The metrics as Prometheus exposition text, ending in a newline.
    """
    snap = metrics_snapshot()
    out: list[str] = []

    def counter(name: str, value: float, help_text: str, unit: str = "counter") -> None:
        out.append(f"# HELP batcher_{name} {help_text}")
        out.append(f"# TYPE batcher_{name} {unit}")
        out.append(f"batcher_{name} {value}")

    counter("uptime_seconds", snap["uptime_seconds"], "Seconds since metrics started", "gauge")
    counter("queries_total", snap["queries"]["total"], "Queries executed")
    counter("queries_failed_total", snap["queries"]["failed"], "Queries that raised")
    counter("rows_scanned_total", snap["rows"]["scanned_total"], "Rows read from sources")
    counter("rows_out_total", snap["rows"]["out_total"], "Rows returned to callers")
    counter("bytes_scanned_total", snap["bytes"]["scanned_total"], "Bytes read from sources")
    counter("spills_total", snap["spills"]["total"], "Operator spills to disk")

    out.append("# HELP batcher_query_duration_ms Query wall time in milliseconds")
    out.append("# TYPE batcher_query_duration_ms histogram")
    cumulative = 0
    for edge, count in snap["queries"]["duration_ms_buckets"].items():
        cumulative = max(cumulative, count)
        out.append(f'batcher_query_duration_ms_bucket{{le="{edge}"}} {cumulative}')
    out.append(f'batcher_query_duration_ms_bucket{{le="+Inf"}} {snap["queries"]["total"]}')
    out.append(f"batcher_query_duration_ms_sum {snap['queries']['duration_ms_total']}")
    out.append(f"batcher_query_duration_ms_count {snap['queries']['total']}")

    if snap["operators"]:
        out.append("# HELP batcher_operator_elapsed_ms_total Operator wall time by kind")
        out.append("# TYPE batcher_operator_elapsed_ms_total counter")
        for name, stats in snap["operators"].items():
            label = f'{{kind="{name}"}}'
            out.append(f"batcher_operator_elapsed_ms_total{label} {stats['elapsed_ms_total']}")

    counter(
        "partitions_done_total", snap["partitions"]["done_total"], "Distributed partitions done"
    )
    counter("skipped_total", snap["skipped"]["total"], "Rows dropped under on_read_error=skip")
    counter("inference_batches_total", snap["inference"]["batches_total"], "Inference batches run")
    counter("inference_rows_total", snap["inference"]["rows_total"], "Rows through inference")
    counter(
        "inference_latency_ms_total",
        snap["inference"]["latency_ms_total"],
        "Cumulative inference batch latency",
    )
    counter(
        "inference_blocked_ms_total",
        snap["inference"]["blocked_ms_total"],
        "Cumulative worker time blocked on input",
    )
    gpu_devices = snap["gpu"]["devices"]
    if gpu_devices:
        out.append("# HELP batcher_gpu_utilization_percent Current GPU utilization by device")
        out.append("# TYPE batcher_gpu_utilization_percent gauge")
        for device, stats in gpu_devices.items():
            out.append(f'batcher_gpu_utilization_percent{{device="{device}"}} {stats["util_pct"]}')
        out.append("# HELP batcher_gpu_memory_used_bytes Current GPU memory in use by device")
        out.append("# TYPE batcher_gpu_memory_used_bytes gauge")
        for device, stats in gpu_devices.items():
            out.append(
                f'batcher_gpu_memory_used_bytes{{device="{device}"}} {stats["mem_used_bytes"]}'
            )
    return "\n".join(out) + "\n"


def reset_metrics() -> None:
    """Zero every counter and restart the uptime clock.

    For tests, and for a service that reports per-interval numbers rather than letting a
    backend difference cumulative ones. Does not detach the collector, so counting resumes
    immediately from zero.

    Examples:
        .. doctest::

            >>> from batcher.observe import metrics_snapshot, reset_metrics
            >>> reset_metrics()
            >>> metrics_snapshot()["queries"]["total"]
            0

    Returns:
        None.
    """
    _collector.reset()
