"""Emit a query's column-level lineage as an OpenLineage run event.

`governance.lineage.column_lineage` already computes, for every output column, the source
columns its values derive from. Until now that answer was computed for `Dataset.lineage()`
and otherwise discarded, so the governance artifact most enterprises fund a separate tool
to produce was being thrown away once per query. This module emits it.

# Why there is no `openlineage-python` dependency

The same division as `otel.py` — Batcher produces events, the platform consumes them —
reached by a different route. OpenLineage's HTTP transport is a POST of the event JSON to
``{url}/api/v1/lineage``, and that wire contract is accepted by every backend an
enterprise actually runs: Marquez, DataHub, OpenMetadata, Atlas. The client library's
Python dataclasses, by contrast, have moved between modules across its major versions
(`openlineage.client.run`, then `openlineage.client.event_v2`), so binding to them would
mean version-sniffing a dependency the host also installs for its own integrations.

Emitting the document directly costs one stdlib POST, keeps the facets inspectable in a
test with nothing installed, and cannot conflict with the host's own client. A backend
reachable only over Kafka is reached by pointing `openlineage_url` at a proxy.

# The emit never delays a query

Events are handed to a bounded queue drained by one daemon thread. A lineage backend that
is down, slow to resolve, or saturated costs a dropped event and a debug log line, never
query latency and never an exception — the query already computed the right answer, and
telemetry does not get to change that.

# One event per job, not one per worker

Lineage is a property of the *job*, not of a partition: every worker in a distributed run
executes the same plan over different rows, so emitting per worker would produce N
identical events differing only in row counts. The emit therefore happens once, on the
driver, in the control plane — and the event *describes* the distributed run rather than
ignoring it. The Batcher run facet carries `distributed`, the worker-operator count, and
the measured usage summed across workers, so a distributed query is distinguishable in the
lineage backend from the same plan run on one node.

This is the one place that distinction matters: the emit is not per-row work, it is one
HTTP request per query issued after the query has finished.

# Run identity

OpenLineage requires a run id that is a UUID; Batcher's `query_id` is not one. The run id
is derived from the query id with a fixed-namespace UUID5, so the START event emitted
before execution and the COMPLETE event emitted after it agree without threading state
between them, and a re-emitted event is idempotent in the backend.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from batcher.io import Source
    from batcher.plan.logical import LogicalPlan
    from batcher.plan.profile import QueryProfile

__all__ = [
    "emit_run_complete",
    "emit_run_failure",
    "emit_run_start",
    "openlineage_enabled",
]

_PRODUCER = "https://github.com/stephenoffer/batcher"
_COLUMN_LINEAGE_SCHEMA = "https://openlineage.io/spec/facets/1-0-1/ColumnLineageDatasetFacet.json"
# A fixed namespace so the query-id → run-id mapping is stable across processes and across
# the START/COMPLETE pair. Any constant UUID works; this one is arbitrary and permanent.
_RUN_NAMESPACE = uuid.UUID("6f1b0d5e-6a3a-5c7f-9b1e-2a0c9d4e8f31")

# One daemon thread drains a bounded queue. Bounded because an unreachable backend must
# cost a fixed amount of memory rather than an unbounded one, and dropping the oldest
# events is the right failure for telemetry: a lineage backend that missed an hour wants
# the most recent hour, not the first minute of the outage.
_QUEUE_DEPTH = 256
_queue: queue.Queue[_Delivery] | None = None
_worker: threading.Thread | None = None
_worker_lock = threading.Lock()
_dropped = 0


class _Delivery(NamedTuple):
    """One event, plus the destination resolved at the moment it was produced.

    The destination travels *with* the event rather than being read on the drain thread.
    The configuration is scoped — `config_context` is routinely exited before a background
    thread gets to run, and the ambient config on a worker thread is not the one the query
    ran under — so reading it late silently posted to nowhere. Resolving here also keeps
    the secret lookup for the API key on the thread that already has the query's context.
    """

    url: str
    api_key: str
    timeout_s: float
    event: dict[str, Any]


def openlineage_enabled() -> bool:
    """Whether lineage emission is on and an endpoint is configured.

    Returns:
        True when a run event would actually be produced.
    """
    obs = _observability()
    return bool(obs.openlineage and _endpoint(obs))


def _observability() -> Any:
    """The active observability configuration."""
    from batcher.config import active_config

    return active_config().observability


def _endpoint(obs: Any) -> str:
    """The lineage receiver's base URL — the config field, else the standard env var.

    ``OPENLINEAGE_URL`` is read as a fallback because it is the variable every other
    OpenLineage integration in a platform already sets, and asking an operator to set a
    second one that means the same thing is how the two drift apart.
    """
    return (obs.openlineage_url or os.environ.get("OPENLINEAGE_URL", "")).rstrip("/")


def _api_key(obs: Any) -> str:
    """The bearer token for the receiver, resolving a ``env:``/``file:``/``cmd:`` reference."""
    from batcher.io.credentials import resolve_secret

    raw = obs.openlineage_api_key or os.environ.get("OPENLINEAGE_API_KEY", "")
    if not raw:
        return ""
    return resolve_secret(raw, what="OpenLineage API key") or ""


def _ensure_worker() -> queue.Queue[_Delivery]:
    """The emit queue, starting the drain thread on first use."""
    global _queue, _worker
    with _worker_lock:
        if _queue is None:
            _queue = queue.Queue(maxsize=_QUEUE_DEPTH)
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(
                target=_drain, args=(_queue,), name="batcher-openlineage", daemon=True
            )
            _worker.start()
        return _queue


def _drain(q: queue.Queue[_Delivery]) -> None:
    """Post every queued delivery, forever, swallowing each failure independently."""
    while True:
        delivery = q.get()
        try:
            _post(delivery)
        except Exception:  # pragma: no cover - telemetry must never escalate
            from batcher._internal.logging import get_logger

            get_logger("api").debug("openlineage post failed", exc_info=True)
        finally:
            q.task_done()


def _post(delivery: _Delivery) -> None:
    """POST one event to its receiver over OpenLineage's HTTP transport."""
    import urllib.request

    body = json.dumps(delivery.event).encode()
    request = urllib.request.Request(f"{delivery.url}/api/v1/lineage", data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    if delivery.api_key:
        request.add_header("Authorization", f"Bearer {delivery.api_key}")
    with urllib.request.urlopen(request, timeout=delivery.timeout_s):
        pass


def _emit(event: dict[str, Any]) -> None:
    """Hand one event to the drain thread, dropping it if the queue is full.

    Best-effort and correctness-neutral: a saturated queue means the backend is not
    keeping up, and the right response is to lose the event rather than to block the
    thread that just finished a query.
    """
    global _dropped
    obs = _observability()
    url = _endpoint(obs)
    if not url:
        return
    delivery = _Delivery(url, _api_key(obs), obs.openlineage_timeout_s, event)
    try:
        _ensure_worker().put_nowait(delivery)
    except queue.Full:
        _dropped += 1
        from batcher._internal.logging import get_logger

        get_logger("api").debug("openlineage queue full; dropped %d events", _dropped)


def _run_id(query_id: str) -> str:
    """The UUID this query's run is known by, derived stably from `query_id`."""
    return str(uuid.uuid5(_RUN_NAMESPACE, query_id or "unknown"))


def _now() -> str:
    """The current instant as the ISO-8601 string OpenLineage requires."""
    return datetime.now(UTC).isoformat()


def _namespace() -> str:
    """The OpenLineage namespace this deployment's jobs are recorded under."""
    return _observability().openlineage_namespace or "batcher"


def _source_names(sources: list[Source] | None) -> list[str]:
    """The table identifier for each bound source, in scan order.

    Uses the same `table_name` resolution `Dataset.lineage()` renders origins with, so the
    names in an emitted event and the names a user sees locally are the same names.
    """
    from batcher.api.security import table_name

    return [table_name(s) or f"<source {i}>" for i, s in enumerate(sources or [])]


def _column_lineage_facet(plan: LogicalPlan, tables: list[str]) -> dict[str, Any] | None:
    """The standard ColumnLineage facet for `plan`, or None when it has nothing to say.

    Over-approximates exactly as `column_lineage` does — an opaque `map_batches` makes
    every output column derive from every input column — because for a governance answer a
    false positive costs a review and a false negative costs a breach.
    """
    from batcher.governance import column_lineage

    lineage = column_lineage(plan, tables)
    if not lineage:
        return None
    namespace = _namespace()
    fields = {
        alias: {
            "inputFields": [
                {"namespace": namespace, "name": table, "field": column}
                for table, column in sorted(origins)
            ]
        }
        for alias, origins in lineage.items()
        if origins
    }
    if not fields:
        return None
    return {
        "_producer": _PRODUCER,
        "_schemaURL": _COLUMN_LINEAGE_SCHEMA,
        "fields": fields,
    }


def _batcher_run_facet(profile: QueryProfile | None) -> dict[str, Any]:
    """What the run cost and how it was executed, as a Batcher-specific run facet.

    This is where the distributed shape of the run is recorded. A lineage backend cannot
    otherwise tell a plan that ran on one node from the same plan that ran on a hundred,
    and those are very different runs to audit.
    """
    facet: dict[str, Any] = {"_producer": _PRODUCER, "_schemaURL": _PRODUCER}
    if profile is None:
        return facet
    facet.update(
        {
            "queryId": profile.query_id,
            "distributed": profile.distributed,
            "rows": profile.rows,
            "totalMs": profile.total_ms,
            "spilled": any(op.spilled for op in profile.ops),
            # Populated only on the distributed path, and the single clearest signal that
            # this run was a cluster run rather than a local one.
            "workerOperators": len(profile.worker_ops),
        }
    )
    usage = profile.usage
    if usage.measured:
        facet.update(
            {
                "cpuMs": usage.cpu_ms,
                "coresBusy": usage.cores_busy,
                "peakRssBytes": usage.peak_rss_bytes,
            }
        )
    return facet


def emit_run_start(query_id: str, plan: LogicalPlan, sources: list[Source] | None) -> None:
    """Emit the START event that opens this query's lineage run.

    Declares the inputs and the column lineage before execution, so a query that later
    fails still leaves a record of what it was going to read. A no-op when disabled.

    Args:
        query_id: The id the event log assigned to this query.
        plan: The logical plan about to execute.
        sources: The plan's bound sources, in scan order.

    Returns:
        None.
    """
    if not openlineage_enabled():
        return
    try:
        _emit(_build(query_id, "START", plan=plan, sources=sources, profile=None))
    except Exception:  # pragma: no cover - telemetry must never fail a query
        from batcher._internal.logging import get_logger

        get_logger("api").debug("openlineage start emit failed", exc_info=True)


def emit_run_complete(
    profile: QueryProfile, plan: LogicalPlan, sources: list[Source] | None
) -> None:
    """Emit the COMPLETE event carrying the measured facts of a finished run.

    Args:
        profile: The measured profile of the query that has finished.
        plan: The logical plan that executed.
        sources: The plan's bound sources, in scan order.

    Returns:
        None.
    """
    if not openlineage_enabled():
        return
    try:
        _emit(_build(profile.query_id, "COMPLETE", plan=plan, sources=sources, profile=profile))
    except Exception:  # pragma: no cover - telemetry must never fail a query
        from batcher._internal.logging import get_logger

        get_logger("api").debug("openlineage complete emit failed", exc_info=True)


def emit_run_failure(query_id: str, exc: BaseException) -> None:
    """Emit the FAIL event for a query that raised, with the error recorded on the run.

    The inputs were already declared by the START event, so this event carries the error
    facet and nothing else. Without it a failed query would leave an open run in the
    lineage backend forever, which is worse than no record at all.

    Args:
        query_id: The id the event log assigned, or an empty string if none was announced.
        exc: The exception that ended the query.

    Returns:
        None.
    """
    if not openlineage_enabled():
        return
    try:
        event = _build(query_id, "FAIL", plan=None, sources=None, profile=None)
        event["run"]["facets"]["errorMessage"] = {
            "_producer": _PRODUCER,
            "_schemaURL": ("https://openlineage.io/spec/facets/1-0-0/ErrorMessageRunFacet.json"),
            "message": f"{type(exc).__name__}: {exc}",
            "programmingLanguage": "PYTHON",
        }
        _emit(event)
    except Exception:  # pragma: no cover - telemetry must never fail a query
        from batcher._internal.logging import get_logger

        get_logger("api").debug("openlineage failure emit failed", exc_info=True)


def _build(
    query_id: str,
    event_type: str,
    *,
    plan: LogicalPlan | None,
    sources: list[Source] | None,
    profile: QueryProfile | None,
) -> dict[str, Any]:
    """Assemble one OpenLineage RunEvent document.

    Kept as plain data rather than the client's dataclasses so the shape is inspectable in
    a test without the optional dependency installed, which is the only way the facet
    contents are checkable in CI.
    """
    namespace = _namespace()
    tables = _source_names(sources)
    event: dict[str, Any] = {
        "eventType": event_type,
        "eventTime": _now(),
        "producer": _PRODUCER,
        "schemaURL": "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/RunEvent",
        "run": {
            "runId": _run_id(query_id),
            "facets": {"batcher": _batcher_run_facet(profile)},
        },
        "job": {
            "namespace": namespace,
            "name": _job_name(plan),
            "facets": {},
        },
        "inputs": [{"namespace": namespace, "name": table} for table in tables],
        "outputs": [],
    }
    if plan is not None and tables:
        facet = _column_lineage_facet(plan, tables)
        if facet is not None:
            # The standard facet belongs on an *output* dataset. A `collect()` has no
            # persisted output, so the lineage is attached to an explicitly-named symbolic
            # output rather than dropped: a governance backend that asks "where does
            # customers.ssn flow" must get an answer for an ad-hoc read too, and a run with
            # inputs and no outputs answers nothing.
            event["outputs"] = [
                {
                    "namespace": namespace,
                    "name": _job_name(plan),
                    "facets": {"columnLineage": facet},
                }
            ]
    return event


def _job_name(plan: LogicalPlan | None) -> str:
    """A stable job name for `plan` — the same signature the learned-stats loop keys on.

    Two runs of the same pipeline over different data share a name, which is what makes a
    lineage backend's job history meaningful; a structurally different query gets its own.
    """
    if plan is None:
        return "batcher.query"
    try:
        from batcher.kyber.signature import plan_signature

        signature = plan_signature(plan)
    except Exception:  # pragma: no cover - an unsignable plan must still emit
        return "batcher.query"
    return f"batcher.query.{signature}" if signature else "batcher.query"
