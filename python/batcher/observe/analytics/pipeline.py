"""The per-pipeline report — its runs, its steps, its plan graph, and its run grid.

This is the page a reader lands on after choosing a pipeline, so it assembles several views
of the same runs: a step rollup, the plan as a graph, and the runs-by-step matrix.
"""

from __future__ import annotations

from statistics import median
from typing import Any

from batcher.observe.analytics.series import percentiles

__all__ = ["pipeline_report"]


def pipeline_report(signature: str, details: list[dict[str, Any]]) -> dict[str, Any]:
    """What is true of a pipeline across *all* its runs, not just the latest one.

    A run's insights answer "what happened that time". These answer "what does this query
    always do" — the step that is reliably the bottleneck, the findings that recur, and
    whether a problem is chronic or was a one-off. A recurring warning deserves a fix; the
    same warning seen once is often just a cold cache.

    Args:
        signature: The pipeline's plan signature.
        details: Full run documents; those from other pipelines are ignored.

    Returns:
        ``{"runs": int, "steps": [...], "recurring": [...], "spill_runs": int}``.
    """
    mine = [d for d in details if d.get("signature") == signature]
    if not mine:
        return {"runs": 0, "steps": [], "recurring": [], "spill_runs": 0}

    steps: dict[int, dict[str, Any]] = {}
    for run in mine:
        for node in (run.get("dag") or {}).get("nodes", []):
            if not node.get("measured"):
                continue
            entry = steps.setdefault(
                int(node["op_id"]),
                {
                    "op_id": int(node["op_id"]),
                    "kind": node.get("kind", "?"),
                    "detail": node.get("detail", ""),
                    "samples": 0,
                    "total_ms": 0.0,
                    "max_ms": 0.0,
                    "critical_runs": 0,
                },
            )
            entry["samples"] += 1
            entry["total_ms"] += float(node.get("elapsed_ms", 0.0))
            entry["max_ms"] = max(entry["max_ms"], float(node.get("elapsed_ms", 0.0)))
            if node.get("on_critical_path"):
                entry["critical_runs"] += 1
    for entry in steps.values():
        entry["mean_ms"] = entry["total_ms"] / max(entry["samples"], 1)
        # How often this step is on the critical path. A step that is always on it is where
        # tuning pays every run; one that is sometimes on it moves with the data.
        entry["critical_share"] = entry["critical_runs"] / max(entry["samples"], 1)

    counts: dict[str, dict[str, Any]] = {}
    for run in mine:
        for insight in run.get("insights", []):
            entry = counts.setdefault(
                insight["rule"],
                {
                    "rule": insight["rule"],
                    "title": insight["title"],
                    "severity": insight["severity"],
                    "action": insight["action"],
                    "count": 0,
                },
            )
            entry["count"] += 1
    recurring = sorted(counts.values(), key=lambda r: r["count"], reverse=True)
    for entry in recurring:
        entry["share"] = entry["count"] / len(mine)
        entry["chronic"] = entry["count"] >= max(2, len(mine) // 2)

    return {
        "runs": len(mine),
        "steps": sorted(steps.values(), key=lambda e: e["mean_ms"], reverse=True),
        "recurring": recurring,
        "spill_runs": sum(
            1 for d in mine if any(n.get("spilled") for n in (d.get("dag") or {}).get("nodes", []))
        ),
        "dag": _pipeline_dag(mine, steps),
        "grid": _run_grid(mine),
    }


def _pipeline_dag(runs: list[dict[str, Any]], steps: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """The pipeline's plan graph, with each node carrying *typical* rather than one-off stats.

    A pipeline is defined by its plan shape, so every run draws the same graph — which means
    the graph can be shown once, annotated with what each step usually costs. That is the
    view a run-level DAG cannot give: `hash_join` taking 80 ms in the run you happen to have
    open says nothing about whether 80 ms is normal for it.

    Layout and edges are taken verbatim from a representative run so the pipeline graph and
    every run graph are the same picture; only the numbers on the nodes differ.
    """
    representative = next((r for r in reversed(runs) if (r.get("dag") or {}).get("nodes")), None)
    if not representative:
        return {"nodes": [], "edges": [], "width": 0, "depth": 0, "critical_path": []}
    dag = dict(representative["dag"])
    per_run = {}
    for run in runs:
        for node in (run.get("dag") or {}).get("nodes", []):
            if node.get("measured"):
                per_run.setdefault(int(node["op_id"]), []).append(
                    float(node.get("elapsed_ms", 0.0))
                )

    nodes = []
    for node in dag.get("nodes", []):
        op_id = int(node["op_id"])
        samples = per_run.get(op_id, [])
        stats = steps.get(op_id, {})
        merged = dict(node)
        merged.update(
            samples=len(samples),
            mean_ms=stats.get("mean_ms", 0.0),
            max_ms=stats.get("max_ms", 0.0),
            critical_share=stats.get("critical_share", 0.0),
            percentiles=percentiles(samples),
            # The node's headline number becomes the typical cost, so the drawing's colour
            # ramp encodes "what this step usually costs" rather than one arbitrary run.
            elapsed_ms=stats.get("mean_ms", 0.0),
        )
        nodes.append(merged)
    dag["nodes"] = nodes
    return dag


def _when(run: dict[str, Any]) -> float:
    """A run's start time, for ordering. `0.0` when unknown, which sorts it first."""
    return float(run.get("started_wall") or 0.0)


def _run_grid(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """A runs x steps matrix — the shape Airflow's grid view popularised, and for the reason.

    A tree or a graph shows one run's structure; it cannot show the same step across twenty
    runs. The matrix can, and that is where "this step got slower last Tuesday" is visible at
    a glance. Each cell carries its duration *and* its ratio to that step's own median, so a
    slow cell stands out regardless of whether the step takes microseconds or minutes.
    """
    # Oldest → newest, the direction a trend is read in. Sorted on the timestamp rather than
    # reversing the input: the caller's order is not part of this function's contract, and
    # assuming it silently drew every column in reverse.
    ordered = sorted(runs, key=_when)
    labels: dict[int, str] = {}
    for run in ordered:
        for node in (run.get("dag") or {}).get("nodes", []):
            labels.setdefault(int(node["op_id"]), str(node.get("kind", "?")))
    if not labels:
        return {"steps": [], "runs": [], "cells": []}

    medians = {}
    for op_id in labels:
        samples = [
            float(n.get("elapsed_ms", 0.0))
            for r in ordered
            for n in (r.get("dag") or {}).get("nodes", [])
            if int(n.get("op_id", -1)) == op_id and n.get("measured")
        ]
        medians[op_id] = median(samples) if samples else 0.0

    cells = []
    for column, run in enumerate(ordered):
        by_id = {int(n["op_id"]): n for n in (run.get("dag") or {}).get("nodes", [])}
        for op_id in labels:
            node = by_id.get(op_id)
            elapsed = float(node.get("elapsed_ms", 0.0)) if node else None
            typical = medians[op_id]
            cells.append(
                {
                    "run": column,
                    "op_id": op_id,
                    "elapsed_ms": elapsed,
                    # None rather than 1.0 when there is nothing to compare against, so the
                    # UI can render "no data" instead of implying a perfectly typical cell.
                    "ratio": (elapsed / typical) if (elapsed is not None and typical > 0) else None,
                    "spilled": bool(node.get("spilled")) if node else False,
                }
            )
    return {
        "steps": [
            {"op_id": op_id, "kind": kind, "median_ms": medians[op_id]}
            for op_id, kind in sorted(labels.items())
        ],
        "runs": [
            {
                "query_id": r.get("query_id", ""),
                "started_wall": r.get("started_wall", 0.0),
                "total_ms": r.get("total_ms", 0.0),
                "status": r.get("status", "ok"),
            }
            for r in ordered
        ],
        "cells": cells,
    }
