"""Per-query event log — one JSON document per query (Spark's event-log analog).

When ``observability.event_log`` is on, each executed query writes a structured record to
``$BATCHER_HOME/logs`` (or ``~/.batcher/logs``): the logical and optimized plan, the
Kyber/Carbonite decisions, and the measured per-operator profile — the same `QueryProfile`
`explain(analyze=True)` renders. It is the developer/operator artifact for understanding,
after the fact, what a query planned and did.

It is **opt-in** (see `ObservabilityConfig.event_log`). An enabled log attaches the
collector for the whole query and then assembles, JSON-encodes, and writes one document
per query — ~0.3 ms, which on a small `collect()` is a quarter of the entire control
plane. Left on by default, every query paid for an artifact almost none of them had a
reader for.

Note the cost is *not* the disk: the write is a page-cached `open`/`write`/`close` and
releases the GIL, and moving it to a background writer thread measured **no improvement at
all** (1.13 ms async vs 1.09 ms sync) because the expensive part — the profile assembly and
the `json.dumps` — is GIL-bound Python that a thread cannot run in parallel with the query
anyway. The only way to not pay it is to not do it.

The collector is attached to the execution context only when the feature is on, so a
disabled event log adds nothing.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from itertools import count
from pathlib import Path
from typing import Any

from batcher._internal.logging import note_suppressed
from batcher.plan.profile import ProfileCollector

__all__ = [
    "event_log_collector",
    "pipeline_signature",
    "query_label",
    "report_failure",
    "report_stream",
    "start_query_report",
    "write_event_log",
]

# Per-process query counter, so two queries in the same millisecond get distinct ids.
_counter = count()
# Prune the directory once every this many writes, so the O(files) scan is amortized
# across queries instead of paid on every small query.
_PRUNE_EVERY = 64


def event_log_collector() -> ProfileCollector | None:
    """A `ProfileCollector` when a profile consumer is enabled, else `None` (zero overhead).

    The same measured profile feeds the JSON event log, the OpenTelemetry span emit, and
    the event bus (the dashboard's timeline and plan DAG), so the collector is attached
    when *any* of them is on and the (small) profile-assembly cost is paid once for all
    three. Checking the bus matters: without it, opening the dashboard with the event log
    turned off would show queries appearing and finishing but no per-operator detail —
    the profile would never have been measured.
    """
    from batcher._internal import events
    from batcher.config import active_config

    obs = active_config().observability
    if not (obs.event_log or obs.otel_traces or events.listening()):
        return None
    return ProfileCollector()


def pipeline_signature(plan: object) -> str:
    """A stable hash of `plan`'s *shape*, used to group runs into a pipeline.

    Two runs of the same pipeline over different data share a signature; a structurally
    different query does not. Reuses Kyber's `plan_signature` — the same identity the
    learned-stats loop keys on — rather than inventing a second scheme, so "the dashboard's
    pipeline" and "the thing the optimizer learned about" are the same thing.

    Args:
        plan: The `LogicalPlan` about to execute.

    Returns:
        A short hex signature, or `""` if the plan cannot be signed.
    """
    try:
        from batcher.kyber.signature import plan_signature

        return plan_signature(plan)
    except Exception as exc:  # pragma: no cover - an unsignable plan must not fail the query
        note_suppressed("api", "sign the plan for the event log", exc)
        return ""


def start_query_report(label: str, signature: str = "") -> str:
    """Allocate this query's id, announce it on the event bus, and return the id.

    Called by the conductor before execution so the terminal progress bar and the dashboard
    can show a query *while* it runs, not only once it has finished. The id it mints is the
    same one `write_event_log` later stamps on the profile and the on-disk document, so the
    live view and the archived artifact refer to the query by one name.

    Returns `""` when no sink is attached, which is the default. Minting an id costs a
    `strftime` and a `getpid`, and this runs on every terminal op — so the common case,
    where nobody is watching, must not pay for a name nothing will ever read.

    Args:
        label: A short human name for the query (the terminal op and root operator).
        signature: The pipeline signature from `pipeline_signature`, grouping repeated runs.

    Returns:
        The query id, to hand back to `write_event_log`, or `""` if nothing is listening.
    """
    from batcher._internal import events

    if not events.listening():
        return ""
    query_id = _query_id(next(_counter))
    events.publish(
        events.QUERY_START, query_id=query_id, name=label, label=label, signature=signature
    )
    return query_id


def write_event_log(
    collector: ProfileCollector | None,
    *,
    total_ms: float,
    rows: int,
    query_id: str | None = None,
) -> None:
    """Report `collector`'s profile: the JSON event log and/or OpenTelemetry spans.

    Writes no profile when no consumer is enabled (`collector is None`) or the query never
    reached the optimizer (a metadata-answered fast path leaves `optimized_ir` unset) — but
    it still closes the query out on the event bus, because a `QUERY_START` that never gets
    its `QUERY_END` leaves a progress bar spinning and a dashboard row stuck on "running"
    for a query that finished. The profile is assembled once and fed to both sinks.
    Best-effort: a filesystem or exporter error is swallowed so observability never fails a
    query.
    """
    if collector is None or collector.optimized_ir is None:
        _publish_end(query_id, total_ms=total_ms, rows=rows, profile=None)
        return
    from batcher._internal.logging import get_logger
    from batcher.api.terminal.otel import emit_query_spans
    from batcher.config import active_config

    cfg = active_config().observability
    seq = next(_counter)
    # Reuse the id `start_query_report` already announced, so the live view and the archived
    # document name the same query; mint one only for a caller that never announced a start.
    query_id = query_id or _query_id(seq)
    profile = collector.to_profile(total_ms=total_ms, rows=rows, query_id=query_id)
    _publish_stages(profile, query_id)
    _publish_end(query_id, total_ms=total_ms, rows=rows, profile=profile.to_dict())
    if cfg.event_log:
        try:
            log_dir = _resolve_dir(cfg.event_log_dir)
            (log_dir / f"{query_id}.json").write_text(json.dumps(profile.to_dict(), default=str))
            # Pruning scans the directory (O(files)); amortize it across writes so a small
            # query doesn't pay it every time.
            if seq % _PRUNE_EVERY == 0:
                _prune(log_dir, cfg.event_log_max_files)
        except Exception:  # pragma: no cover - event logging must never break a query
            get_logger("api").debug("event-log write failed", exc_info=True)
    # The emitter is itself a no-op unless OTel is enabled and a provider is configured.
    emit_query_spans(profile)


def query_label(plan: object) -> str:
    """A short human name for `plan` — its root operator, e.g. ``"aggregate"``.

    Deliberately cheap and structural. The label is a glance-value on a progress bar and a
    list row, so it reads the plan's class name rather than rendering the plan text, which
    on a wide projection would be kilobytes of string work per query for a field that gets
    truncated to ~22 columns anyway.

    Args:
        plan: The `LogicalPlan` about to execute.

    Returns:
        A lowercase operator name.
    """
    return type(plan).__name__.lower()


def report_failure(query_id: str | None, *, total_ms: float, exc: BaseException) -> None:
    """Close a failed query out on the event bus, recording the exception's message.

    Args:
        query_id: The id `start_query_report` returned, or None if none was announced.
        total_ms: Wall time from execution start to the failure.
        exc: The exception that ended the query.
    """
    from batcher._internal import events

    if not query_id:
        return
    events.publish(
        events.QUERY_END,
        query_id=query_id,
        ok=False,
        total_ms=total_ms,
        rows=0,
        error=f"{type(exc).__name__}: {exc}",
        profile=None,
    )


def report_stream(batches: Iterator[Any], *, label: str, signature: str = "") -> Iterator[Any]:
    """Yield `batches` unchanged, publishing per-batch progress to the event bus.

    This is the one place the engine reports *live* progress, and it can only exist on the
    streaming path: `collect` measures inside Rust and hands the profile back at the end,
    whereas `iter_batches` already surfaces each `RecordBatch` in Python, so reading
    `num_rows` off one costs nothing and touches no tuple. Per batch, not per row — an
    `O(rows)` control-plane loop would violate the data-plane boundary.

    Passes through untouched when nothing is listening, so the default path adds one
    generator frame and a tuple check per batch.

    Args:
        batches: The batch iterator to wrap.
        label: A short human name for the stream (the root operator).
        signature: The pipeline signature grouping repeated runs of this shape.

    Returns:
        An iterator yielding exactly the batches it was given.
    """
    from batcher._internal import events
    from batcher.observe import ensure_sinks

    # Streaming is a terminal op too, so it must attach the configured sinks the same way
    # `_collect` does — otherwise the one path that can report *live* progress is the one
    # path with nothing listening to it.
    ensure_sinks()
    if not events.listening():
        yield from batches
        return
    query_id = _query_id(next(_counter))
    events.publish(
        events.QUERY_START,
        query_id=query_id,
        name=label,
        label=label,
        stage="streaming",
        signature=signature,
    )
    rows = 0
    t0 = time.perf_counter()
    try:
        for batch in batches:
            rows += batch.num_rows
            events.publish(
                events.PROGRESS,
                query_id=query_id,
                name=label,
                rows=batch.num_rows,
                bytes=batch.nbytes,
            )
            yield batch
    except GeneratorExit:
        # The consumer stopped early (`break`, or a `head()`-style partial read). That is
        # ordinary use of a stream, not a failure — close the query out as OK with the rows
        # actually delivered, or the progress bar would spin forever and the dashboard would
        # show a red row for a `for batch in ...: break` that did exactly what was asked.
        _publish_end(query_id, total_ms=_ms_since(t0), rows=rows, profile=None)
        raise
    except BaseException as exc:
        report_failure(query_id, total_ms=_ms_since(t0), exc=exc)
        raise
    else:
        _publish_end(query_id, total_ms=_ms_since(t0), rows=rows, profile=None)


def _ms_since(t0: float) -> float:
    """Milliseconds elapsed since the `perf_counter` reading `t0`."""
    return (time.perf_counter() - t0) * 1000.0


def _publish_stages(profile: object, query_id: str) -> None:
    """Replay the measured profile onto the bus as per-operator start/end pairs.

    The engine measures operators in Rust and hands the whole profile back at once, so the
    per-stage events are emitted here, after the fact, rather than live. That is an honest
    limit of where the measurement happens: the dashboard's timeline is exact, and it fills
    in when the query completes rather than growing during it.
    """
    from batcher._internal import events

    if not events.listening():
        return
    for op in getattr(profile, "ops", ()):
        events.publish(
            events.STAGE_START,
            query_id=query_id,
            name=op.kind,
            op_id=op.op_id,
            est_rows=None if op.est_rows != op.est_rows else op.est_rows,  # NaN → None
        )
        events.publish(
            events.STAGE_END,
            query_id=query_id,
            name=op.kind,
            op_id=op.op_id,
            rows_out=op.rows_out,
            elapsed_ms=op.elapsed_ms,
            spilled=op.spilled,
        )
    for decision in getattr(profile, "decisions", ()):
        events.publish(events.DECISION, query_id=query_id, **decision.to_dict())


def _publish_end(
    query_id: str | None,
    *,
    total_ms: float,
    rows: int,
    profile: dict | None,
) -> None:
    """Close the query out on the bus. A no-op when the caller never announced a start."""
    from batcher._internal import events

    if not query_id:
        return
    events.publish(
        events.QUERY_END,
        query_id=query_id,
        ok=True,
        total_ms=total_ms,
        rows=rows,
        profile=profile,
    )


def _query_id(seq: int) -> str:
    """A sortable, process-unique per-query id: ``YYYYmmdd-HHMMSS-<pid>-<seq>``.

    The pid disambiguates concurrent processes (two batch jobs in the same second would
    otherwise both write ``...-000000.json`` and clobber each other); the per-process
    counter disambiguates queries within a process.
    """
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{seq:06d}"


# Directories already `mkdir`-ed this process. The path is still resolved every call (cheap
# string work, and it correctly tracks a changed `event_log_dir` / `BATCHER_HOME`), but the
# `mkdir` syscall — a fixed per-query cost on the default-on event-log path — is issued only
# the first time a given resolved directory is seen.
_CREATED_DIRS: set[str] = set()


def _resolve_dir(configured: str) -> Path:
    """The event-log directory, created if absent (``$BATCHER_HOME/logs`` by default)."""
    if configured:
        path = Path(configured)
    else:
        base = os.environ.get("BATCHER_HOME") or os.path.join(os.path.expanduser("~"), ".batcher")
        path = Path(base) / "logs"
    key = str(path)
    if key not in _CREATED_DIRS:
        path.mkdir(parents=True, exist_ok=True)
        _CREATED_DIRS.add(key)
    return path


def _prune(log_dir: Path, max_files: int) -> None:
    """Keep at most `max_files` event-log documents, deleting the oldest first."""
    if max_files <= 0:
        return
    files = sorted(log_dir.glob("*.json"), key=lambda p: p.name)
    for stale in files[:-max_files]:
        stale.unlink(missing_ok=True)
