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

__all__ = ["drain_worker_metrics", "execute_metered", "record_worker_metrics"]


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
    batches, metrics_json = metered(plan_ir, sources, engine_config)
    return batches, _stamped_with_this_worker(metrics_json)


def _stamped_with_this_worker(metrics_json: str) -> str:
    """Tag every op in a metrics document with the hardware state of the node that ran it.

    A worker's measurements — times, bytes, faults — describe the *worker's* hardware, but
    they are recorded into the hub on the driver, which is frequently a different machine and
    on a heterogeneous cluster is a different machine from most of the workers. Stamping here,
    where the measurement was taken, is what keeps the attribution honest across the trip: the
    driver's transcription then reads this rather than assuming its own.

    Without it, a fleet of large workers driven from a small head node teaches Kyber the head
    node's cost coefficients for work that ran on the workers, and every plan sized from them
    is wrong in the same direction. Best-effort: an unparseable document is passed through
    unchanged rather than dropping a worker's whole contribution over a tagging step.

    Args:
        metrics_json: The raw `ExecMetrics` document from this worker's engine.

    The worker's CPU *clamp* travels the same way and for the same reason. Whether a cgroup's
    quota is binding is a property of the container the work ran in, and on a cluster that is
    the worker's container, never the driver's. A driver reading its own (unthrottled) counters
    on a worker's behalf would report a quiet box for a fleet being clamped to a third of its
    quota — which is the reading that makes `plan.feedback.oversubscribed` shrink the very
    reservations that are already starved.

    Returns:
        The document with `hw_fingerprint` and `cpu_throttled_ratio` set on every op, or the
        original on any failure.
    """
    if not metrics_json:
        return metrics_json
    from batcher._internal.hardware import fingerprint
    from batcher._internal.hardware.cgroup import cgroup_throttled_ratio
    from batcher._internal.hardware.cpu import cpu_thermal_events

    try:
        doc = json.loads(metrics_json)
        here = fingerprint()
        throttled = cgroup_throttled_ratio() or 0.0
        thermal = cpu_thermal_events()
        for op in doc.get("ops", []):
            op["hw_fingerprint"] = here
            op["cpu_throttled_ratio"] = throttled
            op["cpu_thermal_events"] = thermal
        return json.dumps(doc)
    except (ValueError, TypeError, AttributeError):
        return metrics_json


def record_worker_metrics(
    hub: Any,
    metrics_jsons: Iterable[str],
    metrics_out: list[dict[str, Any]] | None = None,
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
        metrics_out: When given, each worker's parsed `ExecMetrics` **document** is appended
            to it — not just its op-list. The document also carries the ``query`` block, the
            whole-execution CPU / memory / disk reading, and a worker's share of that is only
            recoverable here: it is measured per task, so dropping it on the driver is the
            difference between a distributed run reporting what it cost the cluster and
            reporting nothing at all.
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
                metrics_out.append(json.loads(metrics_json))


def drain_worker_metrics(actors: Iterable[Any], hub: Any, metrics_out: Any = None) -> None:
    """Pull each Flight worker's `ExecMetrics` documents and fold them in (driver side).

    The disk-shuffle routes hand their metrics back as a task result, because a task
    returns once and its value travels with it. A Flight worker is a long-lived actor
    whose methods return addresses, tickets and paths — so its measurements have to be
    *drained* instead, which is what this does after the barrier.

    Pulled rather than pushed because nothing subscribes to the event bus inside a Ray
    worker: a measurement published there is delivered to no one. One extra round trip per
    stage, carrying a few kilobytes of JSON, against a shuffle that has just moved the data.

    Best-effort in every direction. An actor that has died, an engine without the drain
    method, a malformed document — each costs its own contribution and nothing else. The
    stage's rows are already computed by the time this runs, and no statistic is worth
    failing a finished query for.

    Args:
        actors: The Flight worker handles this stage ran on.
        hub: The `FeedbackSink` to record into, or `None` to only fill `metrics_out`.
        metrics_out: When given, each worker's parsed document is appended to it for the
            conductor's `QueryProfile`.
    """
    if hub is None and metrics_out is None:
        return
    import ray

    pending = []
    for actor in actors:
        drain = getattr(actor, "drain_metrics", None)
        if drain is None:  # an actor from an engine that predates the drain
            continue
        with contextlib.suppress(Exception):
            pending.append(drain.remote())
    if not pending:
        return
    documents: list[str] = []
    for ref in pending:
        with contextlib.suppress(Exception):  # a worker that died after its work landed
            documents.extend(ray.get(ref) or [])
    record_worker_metrics(hub, documents, metrics_out)
