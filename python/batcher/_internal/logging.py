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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from batcher.config.config import ObservabilityConfig

__all__ = ["configure", "ensure_configured", "get_logger", "native_tracing_settings"]

_ROOT = "batcher"
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
        cfg.log_level,
        cfg.console,
        cfg.log_file,
        cfg.log_file_max_bytes,
        cfg.log_file_backups,
        cfg.log_format,
    )
    if key == _applied:
        return
    logger = logging.getLogger(_ROOT)
    level = _level_value(cfg.log_level)
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
    if not logger.handlers:
        # No sink chosen — keep records from hitting Python's last-resort stderr handler.
        logger.addHandler(logging.NullHandler())
    _applied = key
    _native_settings = (cfg.log_level, cfg.log_format == "json")


def native_tracing_settings() -> tuple[str, bool] | None:
    """The ``(level, as_json)`` the Rust tracing bridge should use, or `None` if unset.

    `core` (the layer allowed to touch the native engine) reads this to call
    `bc_py::init_tracing` — keeping the native import out of this neutral module so the
    layer-independence contract holds. `None` before `configure` has run.
    """
    return _native_settings


def _level_value(name: str) -> int:
    """Map a level name to its numeric value, defaulting to WARNING on an unknown name."""
    return logging.getLevelName(name.upper()) if isinstance(name, str) else logging.WARNING


class _HumanFormatter(logging.Formatter):
    """A compact one-line layout: ``HH:MM:SS LEVEL  batcher.x: message``."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S"
        )


class _JsonFormatter(logging.Formatter):
    """One JSON object per record, for structured log shippers."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)
