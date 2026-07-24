"""The read-only JSON API — one table mapping a path to the store call behind it.

A table rather than an if-chain because the routes are data: the handler iterates it, the
`/api` index route publishes it, and adding an endpoint is one entry instead of an edit in
three places. Every handler is a pure read; nothing here mutates the engine or the store,
which is what makes it safe to serve a dashboard beside a running query.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from batcher.observe.metrics import metrics_snapshot
from batcher.observe.store import ActivityStore
from batcher.observe.system import system_snapshot

__all__ = ["ROUTES", "Route", "resolve"]

#: A route handler: the store and the parsed query string in, a JSON-encodable payload out.
Route = Callable[[ActivityStore, dict[str, list[str]]], Any]


def _str_param(query: dict[str, list[str]], name: str) -> str:
    """Read one string query parameter, or `""` when absent."""
    values = query.get(name)
    return values[0] if values else ""


def _int_param(query: dict[str, list[str]], name: str, default: int = 0) -> int:
    """Read one integer query parameter, falling back to `default` on absent/garbage."""
    values = query.get(name)
    if not values:
        return default
    try:
        return int(values[0])
    except ValueError:
        return default


#: Every JSON route, with the one-line description the `/api` index publishes. Keep the
#: description honest: it is the only documentation a person poking at the API will find.
ROUTES: dict[str, tuple[Route, str]] = {
    "/api/summary": (
        lambda store, _q: store.summary(),
        "Engine-level counters: runs, pipelines, throughput, failures.",
    ),
    "/api/queries": (
        lambda store, _q: {"queries": store.queries()},
        "Every retained run as a summary, newest first.",
    ),
    "/api/pipelines": (
        lambda store, _q: {"pipelines": store.pipelines()},
        "Runs grouped by plan shape, busiest first.",
    ),
    "/api/system": (
        lambda _store, _q: system_snapshot(),
        "The machine, the engine build, and the sizing config every timing is relative to.",
    ),
    "/api/timeseries": (
        lambda store, _q: store.timeseries(),
        "Throughput and run counts bucketed across the session.",
    ),
    "/api/operators": (
        lambda store, _q: {"operators": store.operators()},
        "Session-wide totals per operator kind, costliest first.",
    ),
    "/api/failures": (
        lambda store, _q: {"groups": store.failures()},
        "Failed runs grouped by error message.",
    ),
    "/api/pipeline": (
        lambda store, q: store.pipeline(_str_param(q, "signature")),
        "Cross-run analysis for one pipeline. Takes ?signature=.",
    ),
    "/api/health": (
        lambda store, _q: store.health(system_snapshot()),
        "The engine's current verdict and the checks behind it.",
    ),
    "/api/compare": (
        lambda store, q: store.compare(_str_param(q, "a"), _str_param(q, "b")),
        "A step-by-step diff of two runs. Takes ?a= and ?b=.",
    ),
    "/api/logs": (
        lambda store, q: store.logs(since=_int_param(q, "since")),
        "Log lines from a cursor. Takes ?since=.",
    ),
    "/api/live": (
        lambda store, q: store.live(_str_param(q, "id") or None) or {},
        "Distributed and accelerator telemetry: partitions, GPU load, actor pool, drops.",
    ),
    "/api/metrics": (
        lambda _store, _q: metrics_snapshot(),
        "Cumulative process counters — the same numbers as the Prometheus export.",
    ),
}


def resolve(route: str) -> Route | None:
    """The handler for a path, or `None` when the path is not an API route.

    Args:
        route: The request path, already stripped of its query string and trailing slash.

    Returns:
        The route handler, or None.
    """
    entry = ROUTES.get(route)
    return entry[0] if entry else None


def index() -> dict[str, Any]:
    """The API's own directory — what a person gets for poking at `/api`."""
    return {
        "routes": [
            {"path": path, "description": description}
            for path, (_handler, description) in sorted(ROUTES.items())
        ],
        "dynamic": [
            {"path": "/api/query/<id>", "description": "The full document for one run."},
            {"path": "/metrics", "description": "Prometheus text exposition of the counters."},
        ],
        "write": [
            {
                "path": "/api/pipeline/meta",
                "method": "POST",
                "description": "Name or annotate a pipeline. Body: "
                "{pipeline_id, name?, note?}. The only write; touches names, not the engine.",
            },
        ],
    }
