"""Streaming counters — the one workload that runs for weeks, made scrapeable.

A continuous query already produces a full Spark-parity `StreamingQueryProgress` per
micro-batch: input and output rows, the per-phase duration breakdown, how far behind the
trigger cadence it is, and each stateful operator's retained rows and bytes. All of it was
delivered only to a `StreamingQueryListener` the user had to write and register, and to
`query.last_progress`. Neither reaches a metrics backend.

That left the workload with the longest life of anything the engine runs as the one a
scrape loop could not see: no lag gauge, no state-store growth series, no throughput. And
those are precisely the streaming failure modes that develop over hours rather than
announcing themselves — a query that falls a little further behind every batch, or a state
store that grows because a watermark never advances, both look healthy in any single
reading and obvious in a chart.

Counters and gauges live side by side here on purpose. Rows and batches accumulate; lag,
throughput, and state size are levels. A lag figure without the throughput that produced it
cannot say whether a query is recovering or drowning, so splitting them across two places
would only mean every consumer had to join them back.
"""

from __future__ import annotations

import threading
from typing import Any

from batcher.observe.counters._series import as_number, escape_label

__all__ = ["StreamCounters"]

#: Queries tracked at once. Bounded by how many streams a process runs, which is small;
#: the cap exists so a caller naming each restart differently cannot grow the map forever.
_MAX_QUERIES = 32

#: Fields summed across micro-batches — the counters a backend differences for a rate.
_SUMMED = ("input_rows", "output_rows", "duration_ms")

#: Fields one query's entry may hold. The engine's own phase set is fixed at four, so this
#: only binds a caller that derives a phase name from data — and that caller would otherwise
#: grow one query's dict by a key per micro-batch, forever.
_MAX_FIELDS = 32

#: Fields carried as the latest reading — levels, not accumulations.
_LATEST = (
    "batch_id",
    "behind_by_ms",
    "input_rows_per_second",
    "processed_rows_per_second",
    "state_rows",
    "state_bytes",
)

#: Per-query Prometheus series, as ``(field, metric name, type, help)``.
_SERIES = (
    ("batches", "batches_total", "counter", "Micro-batches completed"),
    ("input_rows", "input_rows_total", "counter", "Rows read by a streaming query"),
    ("output_rows", "output_rows_total", "counter", "Rows a streaming query produced"),
    ("duration_ms", "duration_ms_total", "counter", "Milliseconds spent in micro-batches"),
    (
        "behind_by_ms",
        "behind_by_ms",
        "gauge",
        "How much longer the last micro-batch took than its trigger cadence",
    ),
    (
        "input_rows_per_second",
        "input_rows_per_second",
        "gauge",
        "Rows read per second in the last micro-batch",
    ),
    (
        "processed_rows_per_second",
        "processed_rows_per_second",
        "gauge",
        "Rows produced per second in the last micro-batch",
    ),
    ("state_rows", "state_rows", "gauge", "Rows the stateful operators currently retain"),
    ("state_bytes", "state_bytes", "gauge", "Bytes the retained streaming state holds"),
)


class StreamCounters:
    """Per-query streaming totals and levels, folded from `STREAM` events.

    One instance per collector, with its own lock. Keyed by query name, because a process
    running an ingest stream and an enrichment stream needs to know which one fell behind.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queries: dict[str, dict[str, float]] = {}

    def reset(self) -> None:
        """Forget every query's counters."""
        with self._lock:
            self._queries.clear()

    def record(self, name: str, fields: dict[str, Any]) -> None:
        """Fold one completed micro-batch in.

        Args:
            name: The streaming query's name.
            fields: The `STREAM` event's payload.
        """
        with self._lock:
            if name not in self._queries and len(self._queries) >= _MAX_QUERIES:
                name = "other"
            entry = self._queries.setdefault(name, {"batches": 0.0})
            entry["batches"] += 1
            for field in _SUMMED:
                entry[field] = entry.get(field, 0.0) + as_number(fields.get(field))
            for field in _LATEST:
                entry[field] = as_number(fields.get(field))
            # The per-phase breakdown is summed too, so "is the query slow or is the
            # *checkpoint* slow" is answerable from the export — the first thing to rule out
            # when a stream falls behind, and the two have opposite remedies.
            #
            # Bounded, because this is the one workload that runs for weeks: the engine's
            # own phase set is fixed at four, but a per-batch-derived phase name would
            # otherwise add a key per micro-batch and grow this dict without limit over an
            # uptime measured in days. A new phase past the cap is dropped rather than
            # folded into an "other" bucket — summing unrelated phases into one duration
            # would produce a number that means nothing.
            for key, value in fields.items():
                if not (key.startswith("duration_") and key.endswith("_ms")):
                    continue
                if key in _SUMMED or (key not in entry and len(entry) >= _MAX_FIELDS):
                    continue
                entry[key] = entry.get(key, 0.0) + as_number(value)

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Every tracked query's counters, keyed by name and sorted."""
        with self._lock:
            return {name: dict(entry) for name, entry in sorted(self._queries.items())}

    def render(self) -> list[str]:
        """The counters as Prometheus exposition lines, labelled by query.

        Returns:
            A list of lines, empty until a streaming query has completed a micro-batch.
            Empty rather than zeroed, because a process that runs no stream should not
            publish a streaming series at all — unlike the always-on process counters, a
            zero here would be indistinguishable from a stopped query.
        """
        queries = self.snapshot()
        if not queries:
            return []
        out: list[str] = []
        for field, metric, kind, help_text in _SERIES:
            name = f"batcher_streaming_{metric}"
            out.append(f"# HELP {name} {help_text}")
            out.append(f"# TYPE {name} {kind}")
            for query, entry in queries.items():
                out.append(f'{name}{{query="{escape_label(query)}"}} {entry.get(field, 0)}')
        return out
