"""One-line switches for logging, verbosity, and the progress bar.

Batcher logs through the stdlib `logging` module on the ``batcher.*`` hierarchy, and its
progress bar and dashboard are driven by `ObservabilityConfig`. Both are reachable by
building a config object and calling `set_config`, which is three lines and a `replace`
for something users do constantly. These are the one-liners.

They are ordinary config setters, so they compose with everything else: `set_log_level`
inside a `config_context` is scoped to that block, and turning progress on here is picked
up by the conductor at the next terminal op. The logging switches additionally reconfigure
the live handlers immediately, so a call takes effect on the very next log record rather
than at the next query.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from collections.abc import Iterator

from batcher._internal.errors import ConfigError
from batcher.config.config import VERBOSITY_LEVELS, active_config, set_config

__all__ = [
    "disable_logging",
    "enable_logging",
    "get_logger",
    "set_log_level",
    "set_progress",
    "set_verbosity",
]

_LEVEL_NAMES = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


def get_logger(name: str = "") -> logging.Logger:
    """Return Batcher's stdlib logger, or one of its children.

    Batcher owns the ``batcher`` logger hierarchy and nothing else, so your application's
    own `logging` configuration keeps working unchanged. Use this to attach a handler, set
    a level on one subsystem, or silence the engine entirely with standard `logging` calls
    rather than a Batcher-specific API.

    Examples:
        .. doctest::

            >>> from batcher.config import get_logger
            >>> get_logger().name
            'batcher'
            >>> get_logger("kyber").name
            'batcher.kyber'

    Args:
        name: A subsystem name such as ``"kyber"``, ``"core"``, or ``"io"``. Empty
            returns the root ``batcher`` logger.

    Returns:
        The `logging.Logger` for that name.
    """
    from batcher._internal.logging import get_logger as _get

    return _get(name)


def set_log_level(level: str | int) -> None:
    """Set the log level for every Batcher logger, and apply it immediately.

    The one call that makes Batcher talk. Accepts a level name in any case
    (``"debug"``), a `logging` constant (``logging.DEBUG``), or a Batcher verbosity name
    (``"trace"``), which also raises the Rust data plane's tracing level. Turning the
    level up also attaches the console handler if logging was disabled, so a single call
    is always enough to see output.

    Examples:
        .. doctest::

            >>> import logging
            >>> from batcher.config import get_option, set_log_level
            >>> set_log_level("debug")
            >>> get_option("observability.log_level")
            'DEBUG'
            >>> set_log_level(logging.WARNING)
            >>> get_option("observability.log_level")
            'WARNING'

    Args:
        level: A level name, a `logging` level constant, or a verbosity preset name.

    Raises:
        ConfigError: If `level` is not a recognized level name, constant, or preset.
    """
    _update(log_level=_level_name(level), console=True)


def _level_name(level: str | int) -> str:
    """Normalize a level name, `logging` constant, or verbosity preset to a level name."""
    if isinstance(level, int):
        name = logging.getLevelName(level)
        if not isinstance(name, str) or name not in _LEVEL_NAMES:
            msg = f"unknown log level {level!r}; expected one of {', '.join(_LEVEL_NAMES)}"
            raise ConfigError(msg)
        return name
    upper = level.upper()
    if upper in _LEVEL_NAMES:
        return upper
    preset = {v.name: v for v in VERBOSITY_LEVELS}.get(level.lower())
    if preset is not None:
        return preset.log_level
    msg = (
        f"unknown log level {level!r}; expected a level name ({', '.join(_LEVEL_NAMES)}), "
        f"a logging constant, or a verbosity preset "
        f"({', '.join(v.name for v in VERBOSITY_LEVELS)})"
    )
    raise ConfigError(msg)


def enable_logging(level: str | int = "INFO", *, log_file: str | None = None) -> None:
    """Turn on Batcher's console logging at `level`, optionally also to a file.

    Batcher is quiet until asked, so this is the usual first call when something needs
    explaining. It attaches a stderr handler to the ``batcher`` hierarchy at `level`; pass
    `log_file` to additionally write a rotating log, which is what you want when the
    interesting run is the one that already happened.

    Examples:
        .. doctest::

            >>> from batcher.config import disable_logging, enable_logging, get_option
            >>> enable_logging("debug")
            >>> get_option("observability.console")
            True
            >>> disable_logging()

    Args:
        level: The level to log at. Same accepted spellings as `set_log_level`.
        log_file: Path to a rotating log file, or None for console only.

    Raises:
        ConfigError: If `level` is not a recognized level.
    """
    updates: dict[str, object] = {"log_level": _level_name(level), "console": True}
    if log_file is not None:
        updates["log_file"] = log_file
    _update(**updates)


def disable_logging() -> None:
    """Silence Batcher's console logging without touching your own logging setup.

    Detaches the stderr handler and raises the threshold to CRITICAL. The event log and
    the dashboard, which are separate sinks, keep recording — so a run stays diagnosable
    afterwards even though it printed nothing.

    Examples:
        .. doctest::

            >>> from batcher.config import disable_logging, get_option
            >>> disable_logging()
            >>> get_option("observability.console")
            False

    Returns:
        None.
    """
    _update(console=False, log_level="CRITICAL")


def set_verbosity(level: str | int) -> None:
    """Set the overall verbosity preset, which drives log level and progress together.

    The ``-v``/``-vv`` dial: one of ``silent``, ``quiet``, ``normal``, ``verbose``,
    ``debug``, ``trace``, or the equivalent integer 0-5. Prefer this over `set_log_level`
    when you want "more output" generally rather than a specific threshold, because it
    moves the progress bar with it.

    Examples:
        .. doctest::

            >>> from batcher.config import set_verbosity, get_option
            >>> set_verbosity("verbose")
            >>> get_option("observability.verbosity")
            'verbose'
            >>> set_verbosity("normal")

    Args:
        level: A preset name or its integer rank (0 = silent, 5 = trace).

    Raises:
        ConfigError: If `level` is not a recognized preset.
    """
    names = [v.name for v in VERBOSITY_LEVELS]
    if isinstance(level, int) and not 0 <= level < len(names):
        msg = (
            f"verbosity {level!r} out of range; expected 0-{len(names) - 1} "
            f"or one of {', '.join(names)}"
        )
        raise ConfigError(msg)
    if isinstance(level, str) and level.lower() not in names:
        msg = f"unknown verbosity {level!r}; expected one of {', '.join(names)}"
        raise ConfigError(msg)
    # Clear the explicit overrides so the preset actually drives both dials — leaving a
    # previously-set `log_level` in place would silently win over the preset the user
    # just asked for, which is the opposite of what a single dial is for.
    resolved = level.lower() if isinstance(level, str) else level
    _update(verbosity=resolved, log_level=None, progress=None)


def set_progress(enabled: bool | str = True) -> None:
    """Turn the live terminal progress bar on, off, or back to automatic.

    The default is ``"auto"``, which renders only into a real interactive terminal and
    stays silent when output is redirected, so a log file never fills with escape codes.
    Pass True to force it on (useful in a notebook or a CI job that renders ANSI), or
    False to turn it off entirely.

    Examples:
        .. doctest::

            >>> from batcher.config import get_option, set_progress
            >>> set_progress(False)
            >>> get_option("observability.progress")
            'off'
            >>> set_progress("auto")

    Args:
        enabled: True for ``"on"``, False for ``"off"``, or one of the mode strings
            ``"on"``, ``"off"``, ``"auto"`` directly.

    Raises:
        ConfigError: If `enabled` is a string other than on/off/auto.
    """
    if isinstance(enabled, bool):
        mode = "on" if enabled else "off"
    elif enabled in ("on", "off", "auto"):
        mode = enabled
    else:
        msg = f"unknown progress mode {enabled!r}; expected True, False, 'on', 'off', or 'auto'"
        raise ConfigError(msg)
    _update(progress=mode)


@contextlib.contextmanager
def _reconfigured() -> Iterator[None]:
    """Force the logging handlers to rebuild from the new config on the way out."""
    yield
    from batcher._internal import logging as _log

    _log._applied = None
    _log.ensure_configured()


def _update(**fields: object) -> None:
    """Replace fields of the active `ObservabilityConfig` and re-apply the handlers."""
    with _reconfigured():
        current = active_config()
        observability = dataclasses.replace(current.observability, **fields)  # type: ignore[arg-type]
        set_config(current.replace(observability=observability))
