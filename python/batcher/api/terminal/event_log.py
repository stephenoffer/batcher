"""Per-query event log — one JSON document per query (Spark's event-log analog).

When ``observability.event_log`` is on (the default), each executed query writes a
structured record to ``$BATCHER_HOME/logs`` (or ``~/.batcher/logs``): the logical and
optimized plan, the Kyber/Carbonite decisions, and the measured per-operator profile —
the same `QueryProfile` `explain(analyze=True)` renders. It is the developer/operator
artifact for understanding, after the fact, what a query planned and did.

The collector is attached to the execution context only when the feature is on, so a
disabled event log adds nothing; an enabled one adds the profile assembly plus one small
file write per query (the native execution already runs metered for the feedback loop).
"""

from __future__ import annotations

import json
import os
import time
from itertools import count
from pathlib import Path

from batcher.plan.profile import ProfileCollector

__all__ = ["event_log_collector", "write_event_log"]

# Per-process query counter, so two queries in the same millisecond get distinct ids.
_counter = count()
# Prune the directory once every this many writes, so the O(files) scan is amortized
# across queries instead of paid on every small query.
_PRUNE_EVERY = 64


def event_log_collector() -> ProfileCollector | None:
    """A `ProfileCollector` when a profile consumer is enabled, else `None` (zero overhead).

    The same measured profile feeds the JSON event log and the OpenTelemetry span emit, so
    the collector is attached when *either* is on and the (small) profile-assembly cost is
    paid once for both.
    """
    from batcher.config import active_config

    obs = active_config().observability
    if not (obs.event_log or obs.otel_traces):
        return None
    return ProfileCollector()


def write_event_log(collector: ProfileCollector | None, *, total_ms: float, rows: int) -> None:
    """Report `collector`'s profile: the JSON event log and/or OpenTelemetry spans.

    A no-op when no consumer is enabled (`collector is None`) or the query never reached
    the optimizer (a metadata-answered fast path leaves `optimized_ir` unset). The profile
    is assembled once and fed to both sinks. Best-effort: a filesystem or exporter error is
    swallowed so observability never fails a query.
    """
    if collector is None or collector.optimized_ir is None:
        return
    from batcher._internal.logging import get_logger
    from batcher.api.terminal.otel import emit_query_spans
    from batcher.config import active_config

    cfg = active_config().observability
    seq = next(_counter)
    query_id = _query_id(seq)
    profile = collector.to_profile(total_ms=total_ms, rows=rows, query_id=query_id)
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
