"""Grouping across runs — cost by operator kind, and failures by cause.

Both answer "what does this session look like in aggregate", which is a different question
from any single run and the reason the session view exists at all.
"""

from __future__ import annotations

from typing import Any

__all__ = ["failure_groups", "operator_rollup"]


def operator_rollup(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-operator-kind totals across every profiled run, costliest first.

    The session-level answer to "what is this workload actually made of". One slow join in
    one run is a run-level finding; joins being 60% of everything the engine did all session
    is a different, larger fact, and it is invisible from any single run's page.

    Args:
        details: Full run documents (those carrying a `dag`).

    Returns:
        One dict per operator kind, sorted by total elapsed time descending.
    """
    acc: dict[str, dict[str, Any]] = {}
    for run in details:
        for node in (run.get("dag") or {}).get("nodes", []):
            if not node.get("measured"):
                continue
            kind = str(node.get("kind", "?"))
            entry = acc.setdefault(
                kind,
                {
                    "kind": kind,
                    "runs": 0,
                    "total_ms": 0.0,
                    "rows_out": 0,
                    "spilled": 0,
                    "spill_bytes": 0,
                    "max_ms": 0.0,
                    "slowest_run": "",
                    "example_runs": [],
                },
            )
            elapsed = float(node.get("elapsed_ms", 0.0))
            entry["runs"] += 1
            entry["total_ms"] += elapsed
            entry["rows_out"] += int(node.get("rows_out", 0))
            if elapsed > entry["max_ms"]:
                entry["max_ms"] = elapsed
                # Carry the worst offender so a session-level row can be clicked through to
                # the run that produced it. A rollup you cannot drill into is a dead end.
                entry["slowest_run"] = run.get("query_id", "")
            if run.get("query_id") and run["query_id"] not in entry["example_runs"]:
                entry["example_runs"].append(run["query_id"])
                del entry["example_runs"][8:]
            if node.get("spilled"):
                entry["spilled"] += 1
                entry["spill_bytes"] += int(node.get("spill_bytes", 0))
    rollup = sorted(acc.values(), key=lambda e: e["total_ms"], reverse=True)
    grand = sum(e["total_ms"] for e in rollup) or 1.0
    for entry in rollup:
        entry["share"] = entry["total_ms"] / grand
        entry["mean_ms"] = entry["total_ms"] / max(entry["runs"], 1)
    return rollup


def failure_groups(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Failures grouped by their error message, most frequent first.

    Twenty failures are usually one bug, and a list of twenty identical rows hides that.
    Grouping turns "lots of red" into "one cause, twenty times", which is the difference
    between a wall of noise and a thing to go fix.

    Args:
        queries: Run summaries.

    Returns:
        One entry per distinct error, with the runs that produced it.
    """
    groups: dict[str, dict[str, Any]] = {}
    for query in queries:
        if query.get("status") != "error":
            continue
        # Group on the exception type and message, not the full text: a run id or a row
        # count embedded in the tail would split one cause into many groups.
        message = str(query.get("error") or "Unknown error")
        key = message.split(" at ")[0][:160]
        entry = groups.setdefault(
            key, {"error": key, "count": 0, "runs": [], "first_wall": None, "last_wall": None}
        )
        entry["count"] += 1
        if len(entry["runs"]) < 12:
            entry["runs"].append(query.get("query_id", ""))
        wall = float(query.get("started_wall") or 0.0)
        entry["first_wall"] = (
            wall if entry["first_wall"] is None else min(entry["first_wall"], wall)
        )
        entry["last_wall"] = wall if entry["last_wall"] is None else max(entry["last_wall"], wall)
    return sorted(groups.values(), key=lambda g: g["count"], reverse=True)
