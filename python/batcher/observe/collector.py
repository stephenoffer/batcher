"""The counter store behind ``observe.metrics`` — one process-wide event-bus subscriber.

Split out of :mod:`batcher.observe.metrics`, which is the public façade (``start_metrics``,
``metrics_snapshot``, ``prometheus_text``, ``reset_metrics``). This module owns the *state*:
the histogram boundaries, the per-family counter objects, and the lock-protected mutation
each engine event performs. Neutral, like the rest of ``observe`` — it reads the event bus
and never the engine.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any

from batcher._internal import events
from batcher.observe.accelerators.diagnosis import window_snapshot
from batcher.observe.counters import (
    ResourceGauges,
    StreamCounters,
    WorkCounters,
    WriteCounters,
)
from batcher.observe.node_metrics import node_conditions

# Duration buckets in milliseconds, Prometheus-style cumulative histogram boundaries.
# Chosen to straddle Batcher's stated range: sub-millisecond planning through multi-minute
# distributed jobs, roughly one bucket per half order of magnitude.
_BUCKETS_MS: tuple[float, ...] = (1, 5, 10, 50, 100, 500, 1_000, 5_000, 30_000, 300_000)


class _Collector:
    """Folds bus events into counters. One instance per process, guarded by one lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Own their own locks and their own reset, so they are built once and cleared by
        # `reset` rather than replaced — a `metrics_snapshot()` racing a `reset_metrics()`
        # then reads an empty collector, never a half-built one.
        self.work = WorkCounters()
        self.resources = ResourceGauges()
        self.streams = StreamCounters()
        self.writes = WriteCounters()
        self.reset()

    def reset(self) -> None:
        """Zero every counter and restart the uptime clock."""
        self.work.reset()
        self.resources.reset()
        self.streams.reset()
        self.writes.reset()
        with self._lock:
            self._started = time.time()
            self.queries_total = 0
            self.queries_failed = 0
            self.queries_active = 0
            self.rows_out_total = 0
            self.stream_rows_total = 0
            self.stream_bytes_total = 0
            self.query_ms_total = 0.0
            self.query_ms_max = 0.0
            self._buckets: dict[float, int] = dict.fromkeys(_BUCKETS_MS, 0)
            self._errors: dict[str, int] = defaultdict(int)
            self._log_counts: dict[str, int] = defaultdict(int)
            # Distributed / inference counters. These are cumulative like the rest, so a
            # scraper differences them for rates; the live per-stage view is the separate
            # `InferenceProgress` store. Cardinality is bounded by hardware (GPU devices) and
            # by the plan (skip reasons), never by run length, so a 12-hour job holds these
            # flat.
            self.partitions_done_total = 0
            self.skipped_total = 0
            self.malformed_rows_total = 0
            self._malformed_by_source: dict[str, int] = defaultdict(int)
            # Fault tolerance. These were invisible before: the engine recovers from
            # worker loss transparently, so a query that survived two deaths and one
            # that was simply slow looked identical from outside.
            self._recovery_events: dict[str, int] = defaultdict(int)
            self.infer_batches_total = 0
            self.infer_rows_total = 0
            self.infer_latency_ms_total = 0.0
            self.infer_blocked_ms_total = 0.0
            self._skipped_by_reason: dict[str, int] = defaultdict(int)
            # Data-quality contracts. Keyed by constraint name, which a contract fixes at
            # authoring time, so the map is bounded by the contract rather than by run
            # length — with the same overflow fold the skip reasons use, because a
            # `check(name=...)` built from a row value would otherwise grow it forever.
            self.dq_checks_total = 0
            self.dq_failed_total = 0
            self.dq_violations_total = 0
            self._dq_by_constraint: dict[str, dict[str, int]] = {}
            self._gpu: dict[str, dict[str, float]] = {}
            self.gpu_util_pct_max = 0.0

    def handle(self, event: events.Event) -> None:
        """Fold one event in. Hot path: a dict lookup and a few adds, no allocation."""
        kind = event.kind
        fields = event.fields
        # Delegated before the lock: each owns its own, and taking this one around them
        # would serialize every operator of every concurrent query behind one mutex for no
        # reason — nothing here reads their state.
        if kind == events.STAGE_END:
            self.work.record_stage(event.name or "unknown", fields)
            return
        if kind == events.RESOURCE:
            group, stats = event.name or "unknown", fields.get("stats")
            self.resources.record(group, stats)
            if group == "spill":
                # The one gauge reading that is also a counter increment: a spill store is
                # created per out-of-core phase and torn down with it, so its lifetime
                # `bytes_written` is that phase's whole contribution to the process's spill
                # volume. Gauges alone would only ever show the last phase's.
                self.work.record_spill_store(stats)
            return
        if kind == events.STREAM:
            self.streams.record(event.name or "stream", fields)
            return
        if kind == events.WRITE:
            self.writes.record(event.name or "unknown", fields)
            return
        with self._lock:
            if kind == events.QUERY_END:
                self._end_query(fields)
            elif kind == events.QUERY_START:
                self.queries_active += 1
            elif kind == events.PROGRESS:
                self.stream_rows_total += int(fields.get("rows", 0))
                self.stream_bytes_total += int(fields.get("bytes", 0))
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
            elif kind == events.MALFORMED:
                # Rows the job threw away: a record a reader could not parse, or a row a
                # `map_batches` UDF raised on and `max_errored_rows` allowed to be dropped.
                # Kept apart from `skipped_total`, which counts whole *inputs* — one total
                # over both units answers neither "how many files went missing" nor "how
                # many rows". `by_source` separates the reader from the UDF.
                self._record_malformed(fields)
            elif kind == events.DQ:
                self._record_dq(event.name or str(fields.get("constraint", "unknown")), fields)
            elif kind == events.GPU:
                self._record_gpu(fields)
            elif kind == events.RECOVERY:
                # Keyed by the `event` discriminator, so one series distinguishes a
                # recompute from a speculative backup from a retired replica. Bounded by
                # `RECOVERY_EVENTS`, so it cannot grow.
                self._recovery_events[str(fields.get("event", "unknown"))] += 1

    def _record_malformed(self, fields: dict[str, Any]) -> None:
        """Fold one dropped row in, attributed to what dropped it. Assumes the lock is held."""
        count = int(fields.get("count", 0))
        self.malformed_rows_total += count
        source = str(fields.get("source", "unknown"))
        # Bounded the way `_skipped_by_reason` is. `source` is a format name or a stage kind
        # today, but a caller publishing a per-row unique value must not be able to grow the
        # map without limit over a long run.
        if source not in self._malformed_by_source and len(self._malformed_by_source) >= 64:
            source = "other"
        self._malformed_by_source[source] += count

    def _record_dq(self, constraint: str, fields: dict[str, Any]) -> None:
        """Fold one constraint result in. Assumes the lock is held."""
        violations = int(fields.get("violations", 0))
        passed = bool(fields.get("ok", True))
        self.dq_checks_total += 1
        self.dq_violations_total += violations
        if not passed:
            self.dq_failed_total += 1
        if constraint not in self._dq_by_constraint and len(self._dq_by_constraint) >= 256:
            constraint = "other"
        entry = self._dq_by_constraint.setdefault(
            constraint, {"checks": 0, "failed": 0, "violations": 0}
        )
        entry["checks"] += 1
        entry["violations"] += violations
        if not passed:
            entry["failed"] += 1

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
        # What the run cost the machine. `WorkCounters` owns its own lock, and nothing here
        # reads its state, so the nesting is one-directional and cannot deadlock.
        self.work.record_query(fields.get("usage"))
        self.queries_total += 1
        # Clamped at zero: collection can start mid-query, and a `QUERY_END` whose
        # `QUERY_START` predates the collector would otherwise drive the live gauge
        # negative and keep it there for the rest of the process.
        self.queries_active = max(0, self.queries_active - 1)
        if not fields.get("ok", True):
            self.queries_failed += 1
            # The exception type, not the message: the message embeds row counts, column
            # names and predicate literals, so keying on it would grow this map without
            # bound and leak query content into a metrics label. `report_failure` publishes
            # `error` as ``"TypeName: message"``.
            error = str(fields.get("error", "")).split(":", 1)[0].strip() or "unknown"
            if error not in self._errors and len(self._errors) >= 64:
                error = "other"
            self._errors[error] += 1
        self.rows_out_total += int(fields.get("rows", 0))
        elapsed = float(fields.get("total_ms", 0.0))
        self.query_ms_total += elapsed
        self.query_ms_max = max(self.query_ms_max, elapsed)
        for edge in _BUCKETS_MS:
            if elapsed <= edge:
                self._buckets[edge] += 1

    def snapshot(self) -> dict[str, Any]:
        """A consistent, deep-copied view of every counter. Assumes nothing about callers."""
        operators = self.work.operators()
        totals = self.work.totals()
        rows_scanned, bytes_scanned = self.work.scanned()
        resources = self.resources.snapshot()
        streaming = self.streams.snapshot()
        writes = self.writes.snapshot()
        with self._lock:
            ok = self.queries_total - self.queries_failed
            return {
                "uptime_seconds": time.time() - self._started,
                "queries": {
                    "total": self.queries_total,
                    "active": self.queries_active,
                    "succeeded": ok,
                    "failed": self.queries_failed,
                    "failed_by_error": dict(sorted(self._errors.items())),
                    "duration_ms_total": self.query_ms_total,
                    "duration_ms_max": self.query_ms_max,
                    "duration_ms_mean": (
                        self.query_ms_total / self.queries_total if self.queries_total else 0.0
                    ),
                    # String keys, because the snapshot is served as JSON and JSON object
                    # keys are strings: an int-keyed dict came back from `json.loads` with
                    # string keys, so a consumer that read its own snapshot back saw a
                    # different shape than the one it was handed.
                    "duration_ms_buckets": {str(edge): n for edge, n in self._buckets.items()},
                },
                "rows": {
                    "scanned_total": rows_scanned,
                    "out_total": self.rows_out_total,
                    "streamed_total": self.stream_rows_total,
                },
                "bytes": {
                    "scanned_total": bytes_scanned,
                    "streamed_total": self.stream_bytes_total,
                },
                "spills": {
                    "total": self.work.spills_total,
                    "bytes_total": totals["spill_bytes"],
                    "out_of_core_phases_total": totals["out_of_core_phases"],
                },
                "cpu": {
                    "time_ms_total": totals["cpu_ms"],
                    "execution_ms_total": totals["wall_ms"],
                    "cores_busy": totals["cores_busy"],
                    "threads_max": totals["threads_max"],
                    "involuntary_context_switches_total": totals["invol_ctx_switches"],
                    "voluntary_context_switches_total": totals["vol_ctx_switches"],
                },
                "memory": {
                    "peak_rss_bytes_max": totals["peak_rss_bytes_max"],
                    "minor_faults_total": totals["minor_faults"],
                    "major_faults_total": totals["major_faults"],
                },
                "io": {
                    "read_bytes_total": totals["io_read_bytes"],
                    "write_bytes_total": totals["io_write_bytes"],
                },
                "backends": totals["backends"],
                "operators": operators,
                "resources": resources,
                "streaming": streaming,
                "writes": writes,
                "logs": dict(sorted(self._log_counts.items())),
                "partitions": {"done_total": self.partitions_done_total},
                "recovery": dict(self._recovery_events),
                "data_quality": {
                    "checks_total": self.dq_checks_total,
                    "failed_total": self.dq_failed_total,
                    "violations_total": self.dq_violations_total,
                    "by_constraint": {
                        name: dict(stats) for name, stats in sorted(self._dq_by_constraint.items())
                    },
                },
                "skipped": {
                    "total": self.skipped_total,
                    "by_reason": dict(sorted(self._skipped_by_reason.items())),
                    "malformed_rows_total": self.malformed_rows_total,
                    "malformed_rows_by_source": dict(sorted(self._malformed_by_source.items())),
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
                    # The sampled *window*, which the per-device gauges above cannot express:
                    # they are instantaneous by design, and a consumer stitching a series out
                    # of repeated snapshots still cannot tell a steadily half-fed device from
                    # one alternating between saturated and idle. Empty and flagged unsampled
                    # unless sampling was turned on, so nothing here invents a quiet fleet.
                    "window": window_snapshot(),
                },
                "node": node_conditions(),
            }
