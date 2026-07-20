"""The bounded in-memory record of recent engine activity — the UI's data model.

One bus sink that folds the flat `Event` stream into the shape a human asks about: a
list of queries, each with its operators, its live progress, its decisions, and the log
lines emitted while it ran. The web UI serves this; nothing else in the engine reads it.

Bounded on purpose. A long-lived process (a notebook, a service) would otherwise grow a
list forever, so queries and logs live in `deque(maxlen=...)` ring buffers and the oldest
simply fall off. This is a *debugging* window, not an archive — the durable artifact is
the per-query JSON event log on disk (`api/terminal/event_log.py`), which this store
deliberately does not duplicate.

Thread-safe because the bus is: morsel progress arrives from engine threads while an HTTP
handler thread is serializing a snapshot. Every mutation and every read takes `_lock`, and
reads return deep-enough copies that a caller can never observe a half-updated query.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from batcher._internal import events
from batcher.observe.analytics import (
    compare_runs,
    failure_groups,
    health_report,
    operator_rollup,
    percentiles,
    pipeline_report,
    throughput_series,
)
from batcher.observe.dag import build_dag
from batcher.observe.insights import derive_insights

__all__ = ["ActivityStore", "QueryRecord", "StageRecord"]

#: How many finished queries and log lines the ring buffers retain.
DEFAULT_MAX_QUERIES = 100
DEFAULT_MAX_LOGS = 2000


@dataclass(slots=True)
class StageRecord:
    """One operator's live state — the planned estimate plus what has happened so far."""

    op_id: int
    kind: str
    est_rows: float | None = None
    rows_out: int = 0
    elapsed_ms: float = 0.0
    spilled: bool = False
    done: bool = False
    started_ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_id": self.op_id,
            "kind": self.kind,
            "est_rows": self.est_rows,
            "rows_out": self.rows_out,
            "elapsed_ms": self.elapsed_ms,
            "spilled": self.spilled,
            "done": self.done,
        }


@dataclass(slots=True)
class QueryRecord:
    """One query's whole story: what it planned, what it did, and what it said.

    `status` is ``"running"`` until a `QUERY_END` event arrives, then ``"ok"`` or
    ``"error"``. `profile` is the full `QueryProfile` dict once finished — the same
    document the on-disk event log holds, so the UI's detail view and the archived
    artifact can never disagree.
    """

    query_id: str
    label: str = ""
    #: The plan-shape signature grouping repeated runs into one pipeline; "" if unsigned.
    signature: str = ""
    status: str = "running"
    started_wall: float = 0.0
    started_ts: float = 0.0
    total_ms: float = 0.0
    rows: int = 0
    rows_seen: int = 0
    bytes_seen: int = 0
    error: str = ""
    stages: dict[int, StageRecord] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    profile: dict[str, Any] | None = None

    def to_dict(
        self,
        *,
        detail: bool = False,
        baseline: dict[str, Any] | None = None,
        siblings: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """The query as JSON. `detail` adds the per-stage, decision, and profile payload.

        The list view fetches every query on a poll interval, so it takes only the summary
        fields; the detail view fetches one query and can afford the full document.

        `baseline` carries how this run compares to the rest of its pipeline. A duration
        means little alone — 40 ms is fine or alarming depending entirely on what this same
        query usually takes — so the comparison travels with the run rather than leaving the
        reader to find the pipeline and do the arithmetic.
        """
        summary = {
            "query_id": self.query_id,
            "label": self.label,
            "signature": self.signature,
            "status": self.status,
            "started_wall": self.started_wall,
            "total_ms": self.total_ms,
            "rows": self.rows,
            "rows_seen": self.rows_seen,
            "bytes_seen": self.bytes_seen,
            "n_stages": len(self.stages),
            "n_done": sum(1 for s in self.stages.values() if s.done),
            "error": self.error,
        }
        if not detail:
            return summary
        return {
            **summary,
            "stages": [s.to_dict() for s in sorted(self.stages.values(), key=lambda s: s.op_id)],
            "decisions": list(self.decisions),
            "profile": self.profile,
            # Derived on read, not on ingest: both are pure functions of the profile, and
            # computing them while the query was still running would spend engine time on a
            # view nobody may open.
            "dag": build_dag(
                (self.profile or {}).get("optimized_ir"), (self.profile or {}).get("ops", [])
            ),
            "insights": derive_insights(self.profile),
            "baseline": baseline,
            # The pipeline's other runs, so the run page can offer prev/next and a compare
            # picker without a second request — and without the caller having to know that
            # "the same pipeline" means "the same signature".
            "siblings": siblings or [],
        }


class ActivityStore:
    """A bus sink holding the last N queries and log lines, and serving them as JSON.

    Attach with `attach` (which returns a detach callable) — the store does not subscribe
    itself on construction, so a test can build one and feed it events directly.
    """

    def __init__(
        self,
        *,
        max_queries: int = DEFAULT_MAX_QUERIES,
        max_logs: int = DEFAULT_MAX_LOGS,
    ) -> None:
        self._lock = threading.Lock()
        self._queries: deque[QueryRecord] = deque(maxlen=max_queries)
        self._by_id: dict[str, QueryRecord] = {}
        self._logs: deque[dict[str, Any]] = deque(maxlen=max_logs)
        # Total lines ever appended (not `len(self._logs)`, which stops growing once the
        # ring is full). This is the cursor space `logs` reports against, so a poller can
        # tell "nothing new" apart from "everything I had was evicted".
        self._log_seq = 0

    # --- lifecycle ----------------------------------------------------------

    def attach(self) -> Callable[[], None]:
        """Subscribe this store to the event bus; returns the detach callable."""
        return events.subscribe(self.handle)

    # --- ingest -------------------------------------------------------------

    def handle(self, event: events.Event) -> None:
        """Fold one bus event into the store. This is the sink handed to `subscribe`."""
        if event.kind == events.LOG:
            self._add_log(event)
            return
        with self._lock:
            if event.kind == events.QUERY_START:
                self._start_query(event)
                return
            record = self._by_id.get(event.query_id)
            if record is None:
                # An event for a query that has already aged out of the ring buffer (or
                # one emitted before its QUERY_START, which the bus does not order). Drop
                # it rather than resurrecting a partial record the UI would show as a
                # ghost query with no plan and no start time.
                return
            self._apply(record, event)

    def _start_query(self, event: events.Event) -> None:
        record = QueryRecord(
            query_id=event.query_id,
            label=str(event.fields.get("label") or event.name or event.query_id),
            signature=str(event.fields.get("signature") or ""),
            started_wall=event.wall,
            started_ts=event.ts,
        )
        if len(self._queries) == self._queries.maxlen:
            # `deque` is about to evict the oldest; drop its index entry too or `_by_id`
            # grows without bound and keeps every evicted record alive.
            self._by_id.pop(self._queries[0].query_id, None)
        self._queries.append(record)
        self._by_id[record.query_id] = record

    def _apply(self, record: QueryRecord, event: events.Event) -> None:
        """Dispatch a non-start, non-log event onto its query record."""
        fields = event.fields
        if event.kind == events.STAGE_START:
            op_id = int(fields.get("op_id", 0))
            record.stages[op_id] = StageRecord(
                op_id=op_id,
                kind=event.name,
                est_rows=fields.get("est_rows"),
                started_ts=event.ts,
            )
        elif event.kind == events.STAGE_END:
            stage = record.stages.get(int(fields.get("op_id", 0)))
            if stage is not None:
                stage.rows_out = int(fields.get("rows_out", stage.rows_out))
                stage.elapsed_ms = float(fields.get("elapsed_ms", 0.0))
                stage.spilled = bool(fields.get("spilled", False))
                stage.done = True
        elif event.kind == events.PROGRESS:
            record.rows_seen += int(fields.get("rows", 0))
            record.bytes_seen += int(fields.get("bytes", 0))
        elif event.kind == events.DECISION:
            record.decisions.append(dict(fields))
        elif event.kind == events.QUERY_END:
            record.status = "ok" if fields.get("ok", True) else "error"
            record.total_ms = float(fields.get("total_ms", 0.0))
            record.rows = int(fields.get("rows", 0))
            record.error = str(fields.get("error", ""))
            profile = fields.get("profile")
            if isinstance(profile, dict):
                record.profile = profile

    def _add_log(self, event: events.Event) -> None:
        with self._lock:
            self._logs.append(
                {
                    "wall": event.wall,
                    "level": event.fields.get("level", "INFO"),
                    "logger": event.name,
                    "message": event.fields.get("message", ""),
                    "fields": event.fields.get("fields") or {},
                    "query_id": event.query_id,
                }
            )
            self._log_seq += 1

    # --- read ---------------------------------------------------------------

    def queries(self) -> list[dict[str, Any]]:
        """Every retained query as a summary dict, newest first."""
        with self._lock:
            return [q.to_dict() for q in reversed(self._queries)]

    def query(self, query_id: str) -> dict[str, Any] | None:
        """The full detail document for one query, or `None` if it has aged out."""
        with self._lock:
            record = self._by_id.get(query_id)
            if record is None:
                return None
            peers = [
                q.total_ms
                for q in self._queries
                if q.signature
                and q.signature == record.signature
                and q.status == "ok"
                and q.query_id != query_id
            ]
            baseline = None
            if peers:
                typical = median(peers)
                baseline = {
                    "runs": len(peers),
                    "median_ms": typical,
                    "ratio": (record.total_ms / typical) if typical > 0 else None,
                    "fastest_ms": min(peers),
                    "slowest_ms": max(peers),
                }
            siblings = [
                {
                    "query_id": q.query_id,
                    "started_wall": q.started_wall,
                    "total_ms": q.total_ms,
                    "status": q.status,
                    "rows": q.rows,
                }
                for q in self._queries
                if q.signature and q.signature == record.signature
            ]
            return record.to_dict(detail=True, baseline=baseline, siblings=siblings)

    def logs(self, *, since: int = 0, limit: int = 200) -> dict[str, Any]:
        """Log lines from cursor `since`, plus the cursor to pass on the next poll.

        The cursor counts lines *ever* appended, not an index into the ring buffer, so a
        poller that falls behind eviction resumes at the oldest surviving line instead of
        silently replaying or skipping.
        """
        with self._lock:
            # The oldest surviving line has cursor `_log_seq - len(_logs)`; clamp a
            # stale `since` up to it so an evicted range is skipped, not replayed.
            oldest = self._log_seq - len(self._logs)
            start = max(0, min(since, self._log_seq) - oldest)
            lines = list(self._logs)[start : start + limit]
            cursor = oldest + start + len(lines)
        return {"lines": lines, "cursor": cursor}

    def pipelines(self) -> list[dict[str, Any]]:
        """Retained queries grouped into pipelines by plan shape, busiest first.

        A *pipeline* is every run of the same plan shape — the unit a person actually thinks
        in ("my nightly rollup"), as opposed to a query id, which is one execution of it.
        Grouping by `plan_signature` means two runs over different data land together while a
        structurally different query does not, and it is the same identity Kyber's learned
        stats key on, so the dashboard and the optimizer agree on what "the same query" means.

        The per-pipeline trend (`recent_ms`, oldest → newest) is what makes this worth
        grouping: a single run tells you a duration, a series tells you whether the pipeline
        is degrading.

        Returns:
            One dict per pipeline, ordered by total time spent.
        """
        with self._lock:
            queries = list(self._queries)
        groups: dict[str, list[QueryRecord]] = {}
        for query in queries:
            # An unsigned query is its own pipeline rather than joining a shared "" bucket,
            # which would lump every unsignable plan into one meaningless group.
            groups.setdefault(query.signature or f"~{query.query_id}", []).append(query)

        out: list[dict[str, Any]] = []
        for signature, runs in groups.items():
            done = [r for r in runs if r.status != "running"]
            durations = [r.total_ms for r in done]
            out.append(
                {
                    "signature": signature,
                    "label": runs[-1].label,
                    "runs": len(runs),
                    "n_running": sum(1 for r in runs if r.status == "running"),
                    "n_failed": sum(1 for r in runs if r.status == "error"),
                    "total_ms": sum(durations),
                    "median_ms": median(durations) if durations else 0.0,
                    "min_ms": min(durations, default=0.0),
                    "max_ms": max(durations, default=0.0),
                    "total_rows": sum(r.rows for r in done),
                    "last_wall": max((r.started_wall for r in runs), default=0.0),
                    "last_status": runs[-1].status,
                    "recent_ms": durations[-20:],
                    "percentiles": percentiles(durations),
                    "rows_per_run": [r.rows for r in done][-20:],
                    "slowest_id": max(done, key=lambda r: r.total_ms).query_id if done else "",
                    "fastest_id": min(done, key=lambda r: r.total_ms).query_id if done else "",
                    "query_ids": [r.query_id for r in runs][-50:],
                }
            )
        out.sort(key=lambda p: p["total_ms"], reverse=True)
        return out

    def details(self) -> list[dict[str, Any]]:
        """Full documents for every retained run, for the cross-run analyses.

        Built on demand rather than kept alongside the records: the DAG and insights are
        derived, so storing them too would be a second copy that can fall out of date with
        the profile it came from.
        """
        with self._lock:
            ids = [q.query_id for q in self._queries]
        return [doc for doc in (self.query(qid) for qid in ids) if doc]

    def timeseries(self) -> dict[str, Any]:
        """Throughput and run counts bucketed across the session, for the overview chart."""
        return throughput_series(self.queries())

    def operators(self) -> list[dict[str, Any]]:
        """Session-wide totals per operator kind, costliest first."""
        return operator_rollup(self.details())

    def compare(self, a_id: str, b_id: str) -> dict[str, Any]:
        """A step-by-step diff of two runs, matched by `op_id`."""
        return compare_runs(self.query(a_id), self.query(b_id))

    def failures(self) -> list[dict[str, Any]]:
        """Failed runs grouped by error message, most frequent first."""
        return failure_groups(self.queries())

    def pipeline(self, signature: str) -> dict[str, Any]:
        """Cross-run analysis for one pipeline: its steady bottleneck and recurring findings."""
        return pipeline_report(signature, self.details())

    def health(self, system: dict[str, Any]) -> dict[str, Any]:
        """The engine's current verdict and the checks behind it."""
        return health_report(self.queries(), self.details(), system)

    def summary(self) -> dict[str, Any]:
        """Engine-level counters for the UI header: totals, throughput, failures."""
        with self._lock:
            queries = list(self._queries)
        done = [q for q in queries if q.status != "running"]
        total_ms = sum(q.total_ms for q in done)
        total_rows = sum(q.rows for q in done)
        spill_bytes = sum(
            int(op.get("spill_bytes", 0))
            for q in queries
            for op in ((q.profile or {}).get("ops") or [])
        )
        return {
            "n_queries": len(queries),
            "n_running": sum(1 for q in queries if q.status == "running"),
            "spill_bytes": spill_bytes,
            "percentiles": percentiles([q.total_ms for q in done]),
            "n_failed": sum(1 for q in queries if q.status == "error"),
            "n_pipelines": len({q.signature or f"~{q.query_id}" for q in queries}),
            "total_rows": total_rows,
            "total_ms": total_ms,
            "rows_per_sec": (total_rows / (total_ms / 1000.0)) if total_ms > 0 else 0.0,
            "median_ms": median([q.total_ms for q in done]) if done else 0.0,
        }
