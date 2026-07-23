"""Worker-side metering — the seam that closes the Core→Kyber loop on the distributed path.

`Core measures, Kyber decides` is a *contract*, not a single-node convenience: the cost
coefficients Kyber calibrates and the memory model Carbonite fits are meant to be learned
from whatever the engine actually ran. But a distributed stage runs its sub-plan inside a
Ray worker, and the `MetadataHub` lives on the driver — so a worker that calls the plain
`execute_plan` throws its measurements away. Only the disk-aggregate map task was metered;
distributed sorts, windows, joins, distinct and writes contributed nothing.

That skew is self-reinforcing. The distributed path runs the *largest* inputs and is the
one that spills, so the operators most in need of a fitted `peak_bytes` model were exactly
the ones never feeding it. This module is the two-call fix:

* `execute_metered` replaces a worker's `nat.execute_plan(...)`, returning the sub-plan's
  `ExecMetrics` document alongside its batches;
* `record_worker_metrics` runs on the **driver**, folding the workers' documents into the
  hub (and, optionally, into the `QueryProfile`).

Calibration buckets by operator `kind`, so a worker's sub-plan-local `op_id`s need no
global correlation with the driver's tree — which is why `record_exec_metrics` is called
with no `planned` ops and the rows carry no signature. Both calls are best-effort: an
engine that predates `execute_plan_metered`, or a malformed document, degrades to the
unmetered result rather than failing the query.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterable
from typing import Any

import pyarrow as pa

from batcher._internal.native import engine

__all__ = ["execute_metered", "record_worker_metrics"]


def execute_metered(
    plan_ir: str,
    sources: list[list[pa.RecordBatch]],
    engine_config: str,
) -> tuple[list[pa.RecordBatch], str]:
    """Run a sub-plan in a worker, returning its batches and its `ExecMetrics` document.

    A drop-in for `nat.execute_plan` inside a Ray task. The returned JSON string is opaque
    here — it travels back to the driver, where `record_worker_metrics` parses it. Degrades
    to `("", unmetered batches)` on an engine without the metered entry point, so a worker
    never fails a query to collect a statistic.

    Args:
        plan_ir: The lowered sub-plan IR, already JSON-encoded.
        sources: Input relations, one list of batches per source id.
        engine_config: The driver's engine config JSON, shipped to the worker.

    Returns:
        A pair of the result batches and the raw `ExecMetrics` JSON (`""` if unavailable).
    """
    nat = engine()
    metered = getattr(nat, "execute_plan_metered", None)
    if metered is None:
        return nat.execute_plan(plan_ir, sources, engine_config), ""
    return metered(plan_ir, sources, engine_config)


def record_worker_metrics(
    hub: Any,
    metrics_jsons: Iterable[str],
    metrics_out: list[list[dict[str, Any]]] | None = None,
) -> None:
    """Fold distributed workers' sub-plan metrics into the hub (driver side).

    Each worker measured its own shard, so the documents are recorded independently rather
    than merged: calibration wants one sample per (kind, rows) observation, and a
    W-worker stage legitimately contributes W samples. `metrics_out` is the separate
    channel the conductor's `QueryProfile` renders as the map sub-plan section — that one
    *is* merged, by `merge_metric_ops`, because it is a display of one logical stage.

    Args:
        hub: The `FeedbackSink` to record into, or `None` to only fill `metrics_out`.
        metrics_jsons: Raw `ExecMetrics` documents, one per worker task; `""` is skipped.
        metrics_out: When given, each worker's parsed op-list is appended to it.
    """
    from batcher.config import active_config

    morsel_rows = active_config().execution.morsel_rows
    for metrics_json in metrics_jsons:
        if not metrics_json:
            continue
        if hub is not None:
            from batcher import core

            core.record_exec_metrics(hub, metrics_json, morsel_rows)
        if metrics_out is not None:
            with contextlib.suppress(ValueError, TypeError):
                metrics_out.append(json.loads(metrics_json).get("ops", []))
