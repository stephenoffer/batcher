"""Centralized logging for the whole engine — one configured `batcher.*` hierarchy.

Every subsystem logs through `get_logger("<subsystem>")` (→ `batcher.<subsystem>`), so a
single `configure` call sets the level, handlers (console + optional rotating file), and
format for all of them at once. Lives in the neutral `_internal` layer so `kyber`,
`carbonite`, `core`, `io`, and `api` can all use it without crossing a layer boundary.

Logging is off-until-configured by design: `get_logger` is free (no handlers attached
until `ensure_configured` runs), so importing batcher costs nothing, and a library user
who never opts in sees only Python's default last-resort WARNING behavior. `configure`
leaves the Rust data-plane tracing bridge to `core` (the layer allowed to touch the
native engine), which calls `init_native_tracing` from this module's settings.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from batcher.config.config import ObservabilityConfig

__all__ = [
    "configure",
    "ensure_configured",
    "get_logger",
    "log_kv",
    "native_tracing_settings",
    "note_suppressed",
    "suppress_console_handler",
]

_ROOT = "batcher"
# The `LogRecord` attribute `log_kv` stashes its structured fields under. Namespaced so it
# cannot collide with a stdlib record attribute (which `logging` would refuse to overwrite).
_FIELDS_ATTR = "batcher_fields"
# The settings the current handlers reflect, so a repeat `configure` with the same
# config is a no-op and a changed one rebuilds. `None` means "never configured".
_applied: tuple | None = None
# The (level, json) the Rust tracing bridge should use — set by `configure`, read by
# `native_tracing_settings`. Kept as its own value (not destructured from `_applied`) so
# the two are not positionally coupled.
_native_settings: tuple[str, bool] | None = None


def get_logger(name: str = "") -> logging.Logger:
    """Return the `batcher`/`batcher.<name>` logger (e.g. ``get_logger("kyber")``)."""
    return logging.getLogger(_ROOT if not name else f"{_ROOT}.{name}")


def log_kv(
    logger: logging.Logger,
    level: int,
    msg: str,
    /,
    *,
    exc_info: bool | BaseException = False,
    **fields: object,
) -> None:
    """Log `msg` with structured `fields` attached, rendered per the configured format.

    The one way the engine logs anything with detail. The human formatter appends the
    fields as ``key=value`` pairs; the JSON formatter nests them under ``"fields"``; the
    web UI reads them as real typed data instead of re-parsing a sentence. Writing
    ``log_kv(log, INFO, "spilled", op="hash_join", bytes=n)`` once therefore serves all
    three, where an f-string would have served only the first.

    Args:
        logger: The `batcher.*` logger to record on, from `get_logger`.
        level: A `logging` level constant.
        msg: The short, stable, human-readable event name — not a formatted sentence.
        exc_info: An exception (or `True`) to attach. The formatters turn it into
            ``exc_type``/``exc_message`` fields and a traceback, so a structured consumer
            can alert on the failure *class* rather than regex-matching a sentence.
        **fields: Structured detail; must be JSON-encodable.
    """
    if logger.isEnabledFor(level):
        logger.log(level, msg, exc_info=exc_info, extra={_FIELDS_ATTR: fields})


def note_suppressed(subsystem: str, step: str, exc: BaseException) -> None:
    """Record a failure on a best-effort path that is deliberately not propagated.

    Several paths here must never break a query when they fail: persisting learned stats,
    reading a footer to prune a file, probing for a GPU, cleaning up a temp directory. That
    decision is right. Writing it as ``except Exception: pass`` is not — it makes the
    difference between "this optimization did not apply" and "this optimization has been
    broken since March" unobservable, and the learned-stats loop is the thing that is
    supposed to make plans improve across runs.

    DEBUG level, because a best-effort failure is not the user's problem; but it is on the
    record for whoever asks why the plans stopped improving.

    Args:
        subsystem: The `batcher.<name>` logger to record on, e.g. ``"kyber"``.
        step: A short, stable name for what was attempted, e.g. ``"persist row-count"``.
        exc: The exception being suppressed.
    """
    log_kv(
        get_logger(subsystem),
        logging.DEBUG,
        "best-effort step failed",
        # The traceback, not only the type and the message. "This optimization did not
        # apply" and "this optimization has been broken since March" look identical without
        # a stack, and this is the one record that distinguishes them.
        exc_info=exc,
        step=step,
        error=type(exc).__name__,
        detail=str(exc),
    )


def ensure_configured() -> None:
    """Configure logging once from the active config, if not already done.

    Cheap and idempotent — the conductor calls it at the start of a terminal op so the
    `batcher.*` loggers and the event log honor the user's `ObservabilityConfig` without
    the user having to call `configure` explicitly.
    """
    if _applied is not None:
        return
    from batcher.config import active_config

    configure(active_config().observability)


def configure(cfg: ObservabilityConfig) -> None:
    """Install console/file handlers and level for the `batcher` logger hierarchy.

    Idempotent: re-applying the same settings does nothing; changed settings rebuild the
    handlers. Also bridges the Rust data-plane tracing to the same level when the native
    engine is loaded.
    """
    global _applied, _native_settings
    key = (
        cfg.resolved_log_level,
        cfg.console,
        cfg.log_file,
        cfg.log_file_max_bytes,
        cfg.log_file_backups,
        cfg.log_format,
    )
    if key == _applied:
        return
    logger = logging.getLogger(_ROOT)
    level = _level_value(cfg.resolved_log_level)
    logger.setLevel(level)
    # Batcher manages its own handlers; don't also propagate to the root logger (which
    # would double-emit if the app configured logging too).
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    formatter = _JsonFormatter() if cfg.log_format == "json" else _HumanFormatter()
    if cfg.console:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    if cfg.log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            cfg.log_file,
            maxBytes=max(0, cfg.log_file_max_bytes),
            backupCount=max(0, cfg.log_file_backups),
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # Always mirror to the bus, regardless of the console/file choice: the terminal
    # progress renderer and the web UI are bus sinks, and a user who set `console=False`
    # to keep stderr clean still wants the UI to show what happened. It also means the
    # hierarchy always has at least one handler, so records can never reach Python's
    # last-resort stderr writer even with every configured sink turned off.
    logger.addHandler(_BusHandler())
    _applied = key
    _native_settings = (cfg.resolved_native_log_level, cfg.log_format == "json")


def suppress_console_handler(suppressed: bool) -> None:
    """Detach (or restore) the plain stderr handler, for a renderer that owns the terminal.

    When the live progress reporter is attached it draws log lines itself, interleaved
    correctly with the progress bar. Leaving the plain `StreamHandler` in place as well
    would print every record twice and let a log line land in the middle of the bar. Only
    the reporter calls this, and it restores the handler on detach, so the terminal has
    exactly one owner at any moment.

    Args:
        suppressed: True to remove the console handler, False to reconfigure from config.
    """
    logger = logging.getLogger(_ROOT)
    if not suppressed:
        global _applied
        _applied = None  # force `configure` to rebuild the handler set
        ensure_configured()
        return
    for handler in list(logger.handlers):
        if type(handler) is logging.StreamHandler:
            logger.removeHandler(handler)


def native_tracing_settings() -> tuple[str, bool] | None:
    """The ``(level, as_json)`` the Rust tracing bridge should use, or `None` if unset.

    `core` (the layer allowed to touch the native engine) reads this to call
    `bc_py::init_tracing` — keeping the native import out of this neutral module so the
    layer-independence contract holds. `None` before `configure` has run.
    """
    return _native_settings


def _level_value(name: str) -> int:
    """Map a level name to its numeric value, defaulting to WARNING on an unknown name."""
    if not isinstance(name, str):
        return logging.WARNING
    # `logging.getLevelName` returns the *string* ``"Level FOO"`` (not an int) for an
    # unrecognized name, which then makes `logger.setLevel` raise ``ValueError``. Guard on
    # the return type so an unknown name genuinely falls back to WARNING as documented.
    value = logging.getLevelName(name.upper())
    return value if isinstance(value, int) else logging.WARNING


def _record_fields(record: logging.LogRecord) -> dict[str, object]:
    """The structured fields `log_kv` attached, or `{}` for a plain `logger.info` call."""
    fields = getattr(record, _FIELDS_ATTR, None)
    return fields if isinstance(fields, dict) else {}


def _query_id() -> str:
    """The query in flight on this context, or ``""``.

    Read at *format* time rather than at call time so a plain `logger.warning` inside a
    subsystem gets correlated too, without every call site having to pass an id it does not
    have. This is the field that makes a log line joinable to a query: the bus has carried
    `query_id` on every event since it existed, and the records written to stderr and to a
    log file carried nothing, so a shipper could show a line and never say which of the
    forty queries in the job it belonged to.
    """
    from batcher._internal import events

    return events.current_query_id()


def _rfc3339(created: float) -> str:
    """A UTC timestamp a log shipper can parse without being told the format.

    ``logging.Formatter.formatTime`` produces ``2026-08-25 12:34:56,789`` — local time, a
    space instead of ``T``, a comma instead of a decimal point, and no zone. Elasticsearch,
    Loki, and the OpenTelemetry collector all reject that as a timestamp and fall back to
    ingest time, which silently reorders a log stream whose whole value is its order.
    """
    whole = time.gmtime(created)
    millis = int((created - int(created)) * 1000)
    return f"{time.strftime('%Y-%m-%dT%H:%M:%S', whole)}.{millis:03d}Z"


def _logfmt(fields: dict[str, object]) -> str:
    """`fields` as logfmt ``key=value`` pairs, quoting exactly the values that need it.

    Quoting is the difference between a parseable line and a line that looks parseable.
    ``detail=No such file or directory`` reads to any logfmt parser as ``detail=No``
    followed by three keyless tokens, and the fields most likely to contain a space are
    exception messages and paths — the ones worth reading. The console reporter already
    quoted correctly; the stderr and file handlers did not, so the same record was valid
    logfmt through one sink and not the other.
    """
    out = []
    for key, value in fields.items():
        if value is None:
            text = "null"
        elif value is True:
            text = "true"
        elif value is False:
            text = "false"
        else:
            text = str(value)
        if not text or any(ch in text for ch in ' "=\n'):
            text = '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
        out.append(f"{key}={text}")
    return " ".join(out)


class _HumanFormatter(logging.Formatter):
    """A compact one-line layout: ``HH:MM:SS LEVEL  batcher.x: message  key=value``.

    The `batcher.` prefix is stripped from the logger name — every record in this
    hierarchy has it, so printing it on all of them carries no information and costs
    eight columns of a terminal line.
    """

    def __init__(self) -> None:
        super().__init__(fmt="%(message)s", datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        subsystem = record.name.removeprefix(f"{_ROOT}.").removeprefix(_ROOT)
        head = f"{self.formatTime(record, self.datefmt)} {record.levelname:<7} "
        head += f"{subsystem or 'engine':<10} {record.getMessage()}"
        fields = dict(_record_fields(record))
        query_id = _query_id()
        if query_id:
            fields.setdefault("query_id", query_id)
        if fields:
            head += "  " + _logfmt(fields)
        if record.exc_info:
            head += "\n" + self.formatException(record.exc_info)
        return head


class _JsonFormatter(logging.Formatter):
    """One JSON object per record, for structured log shippers.

    The keys are stable and the timestamp is RFC 3339 in UTC, which is what every shipper
    parses without configuration. ``query_id`` is attached whenever a query is in flight,
    so a line can be joined to the plan, the profile, and the event-log document that
    describe the same run — all of which are already keyed by it.

    An exception is carried three ways rather than one: ``exc_type`` and ``exc_message`` as
    queryable fields, and ``exc`` as the formatted traceback for a human. Alerting on a
    class of failure needs the first two; a string that has to be regex-matched is not a
    field.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "time": _rfc3339(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        query_id = _query_id()
        if query_id:
            payload["query_id"] = query_id
        # Thread and process are what make a distributed log stream readable: the same
        # subsystem logs from the driver and from every worker, and without them the two
        # are indistinguishable in a single index.
        payload["pid"] = os.getpid()
        payload["thread"] = record.threadName
        fields = _record_fields(record)
        if fields:
            payload["fields"] = fields
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc_type"] = record.exc_info[0].__name__
            payload["exc_message"] = str(record.exc_info[1])
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _BusHandler(logging.Handler):
    """Mirrors every `batcher.*` record onto the event bus as a `LOG` event.

    This is what puts logs and metrics in the same stream: the web UI's log pane and the
    terminal renderer's scrollback are both just sinks on the bus, so neither needs its
    own handler or its own copy of the formatting rules. Costs nothing when no sink is
    attached — `publish` returns on a tuple check before the payload is built.
    """

    def emit(self, record: logging.LogRecord) -> None:
        from batcher._internal import events

        if not events.listening():
            return
        fields = dict(_record_fields(record))
        if record.exc_info and record.exc_info[0] is not None:
            # Without this the dashboard's log pane showed "best-effort step failed" with no
            # way to find out what failed: the traceback was on the record and stopped at
            # the two handlers that format it, neither of which the UI reads.
            fields.setdefault("exc_type", record.exc_info[0].__name__)
            fields.setdefault("exc_message", str(record.exc_info[1]))
        events.publish(
            events.LOG,
            name=record.name.removeprefix(f"{_ROOT}."),
            level=record.levelname,
            message=record.getMessage(),
            fields=fields,
        )
