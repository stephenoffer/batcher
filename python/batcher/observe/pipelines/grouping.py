"""Folding the flat run list into pipelines — the shape the pipelines page is built from.

A *pipeline* is every run of one plan shape, grouped by the plan signature Kyber keys
learned stats on. This turns the store's list of runs into one entry per pipeline, each
carrying the cross-run numbers a single run cannot show (the trend, the percentiles), its
durable name and note from the registry, and a compact plan graph — the visual fingerprint
the page draws so one pipeline is tellable from another before its name is read.

Kept here, beside the registry and identity, rather than in the store: the store owns the
ring buffer of runs; this owns what "a pipeline" means across them. It takes the run records
duck-typed, so it never imports the store and the two do not form a cycle.
"""

from __future__ import annotations

from statistics import median
from typing import Any

from batcher.observe.analytics import percentiles
from batcher.observe.dag import plan_shape
from batcher.observe.pipelines.registry import PipelineRegistry

__all__ = ["group_pipelines"]


def group_pipelines(records: list[Any], registry: PipelineRegistry) -> list[dict[str, Any]]:
    """Group run records into pipelines, busiest first, enriched from the registry.

    Args:
        records: The retained run records (anything with the `QueryRecord` attributes).
        registry: The pipeline registry, read for names and stamped with first-seen times.

    Returns:
        One dict per pipeline, ordered by total time spent, each carrying its stable id,
        its name/note, a plan-shape thumbnail, and the cross-run statistics.
    """
    groups: dict[str, list[Any]] = {}
    for record in records:
        # An unsigned query is its own pipeline rather than joining a shared "" bucket,
        # which would lump every unsignable plan into one meaningless group.
        groups.setdefault(record.signature or f"~{record.query_id}", []).append(record)

    names = registry.all()
    out: list[dict[str, Any]] = []
    for pipeline_id, runs in groups.items():
        first_wall = min((r.started_wall for r in runs), default=0.0)
        # Stamp first-seen now, so a pipeline someone later names already has a creation
        # time. A no-op after the first sighting of each id.
        registry.seen(pipeline_id, first_wall)
        out.append(_summary(pipeline_id, runs, names.get(pipeline_id), first_wall))
    out.sort(key=lambda p: p["total_ms"], reverse=True)
    return out


def _summary(pipeline_id: str, runs: list[Any], meta: Any, first_wall: float) -> dict[str, Any]:
    """One pipeline's summary dict."""
    done = [r for r in runs if r.status != "running"]
    durations = [r.total_ms for r in done]
    return {
        # `signature` stays for backward compatibility; `pipeline_id` is the same value under
        # the name the rest of the system now uses for identity.
        "signature": pipeline_id,
        "pipeline_id": pipeline_id,
        # The custom name a person gave it, or "" — the UI falls back to a name generated
        # from the plan shape, so an unnamed pipeline is never nameless.
        "name": meta.name if meta else "",
        "note": meta.note if meta else "",
        "label": runs[-1].label,
        # The plan as a compact graph: the visual fingerprint the pipelines list draws so one
        # pipeline is tellable from another without opening it.
        "plan_shape": _shape_for(done),
        "runs": len(runs),
        "n_running": sum(1 for r in runs if r.status == "running"),
        "n_failed": sum(1 for r in runs if r.status == "error"),
        "total_ms": sum(durations),
        "median_ms": median(durations) if durations else 0.0,
        "min_ms": min(durations, default=0.0),
        "max_ms": max(durations, default=0.0),
        "total_rows": sum(r.rows for r in done),
        "first_wall": first_wall,
        "last_wall": max((r.started_wall for r in runs), default=0.0),
        "last_status": runs[-1].status,
        "recent_ms": durations[-20:],
        "percentiles": percentiles(durations),
        "rows_per_run": [r.rows for r in done][-20:],
        "slowest_id": max(done, key=lambda r: r.total_ms).query_id if done else "",
        "fastest_id": min(done, key=lambda r: r.total_ms).query_id if done else "",
        "query_ids": [r.query_id for r in runs][-50:],
    }


def _shape_for(done: list[Any]) -> dict[str, Any]:
    """The plan-shape thumbnail for a pipeline, from its newest run that has a plan.

    Newest-first: a pipeline's plan can change between runs (a different optimizer decision
    on different data), and the thumbnail should show what it looks like now. Falls back
    through older runs, then to an empty shape a renderer draws as "no plan".
    """
    for record in reversed(done):
        optimized_ir = (record.profile or {}).get("optimized_ir")
        if optimized_ir:
            return plan_shape(optimized_ir)
    return plan_shape(None)
