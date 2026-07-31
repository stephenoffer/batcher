"""Turning the sinks on and off — the one place that owns observability's global state.

The bus is stateless and the sinks are ordinary objects, but *which* sinks are attached to
a process is inherently global: there is one terminal and one dashboard port. That state
lives here, in one module with one lock, rather than as a module-level flag in each sink —
so "is the UI running?" and "is progress being rendered?" each have exactly one answer.

`ensure_sinks` is the conductor's entry point: it is called at the start of every terminal
op, is idempotent, and attaches the console reporter only if the user's config asks for it.
`start_ui` is the user's entry point, and is deliberately explicit — a dashboard that bound
a port on import would be a surprising thing for a library to do.
"""

from __future__ import annotations

import atexit
import threading
import webbrowser
from collections.abc import Callable
from typing import TYPE_CHECKING

from batcher._internal.logging import ensure_configured, get_logger, suppress_console_handler
from batcher.observe.console import ConsoleReporter, should_render
from batcher.observe.store import ActivityStore

if TYPE_CHECKING:
    from batcher.observe.server import UIServer

__all__ = ["ensure_sinks", "start_ui", "stop_ui", "ui_url"]

_lock = threading.Lock()
_console: ConsoleReporter | None = None
_console_detach: Callable[[], None] | None = None
_server: UIServer | None = None
# The observability settings the attached sinks reflect, so a repeat `ensure_sinks` with an
# unchanged config is a tuple compare and a changed one re-syncs. `None` means "never run".
_applied: tuple[str, bool, str, int, bool] | None = None


def ensure_sinks() -> None:
    """Attach the sinks the active config asks for, once per process.

    Called by the conductor before *every* terminal op, so it has to be nearly free on the
    repeat path: it short-circuits on an unchanged `(progress, ui, host, port)` tuple and
    does no TTY probing, no lock, and no handler work until one of them actually changes.
    A per-query `isatty` syscall to re-answer a question whose answer cannot change is
    exactly the kind of fixed overhead the small-query budget cannot absorb.
    """
    global _applied
    from batcher.config import active_config

    config = active_config()
    cfg = config.observability
    sampling = config.accelerator.telemetry_sampling
    key = (cfg.resolved_progress, cfg.ui, cfg.ui_host, cfg.ui_port, sampling)
    if key == _applied:
        return
    ensure_configured()
    with _lock:
        _sync_console(cfg.resolved_progress)
    if cfg.ui and _server is None:
        start_ui(port=cfg.ui_port, host=cfg.ui_host)
    _sync_device_series(sampling)
    _applied = key


def _sync_device_series(wanted: bool) -> None:
    """Start or stop the device sampler to match `accelerator.telemetry_sampling`.

    Sampling is a thread and a driver round trip per device per interval, so it is off unless
    asked for. It belongs on this path rather than at import for the reason the dashboard does:
    a library that started a background thread when it was imported would start one in every
    short-lived Ray worker on the cluster, where the sampling would cost more than the stage it
    was measuring.

    Both directions are handled, so a `config_context` that turns sampling off inside a block
    actually stops the thread rather than leaving it running until the process exits.
    """
    from batcher.observe.accelerators.series import (
        sampling_active,
        start_device_series,
        stop_device_series,
    )

    if wanted:
        start_device_series()
    elif sampling_active():
        stop_device_series()


def _sync_console(mode: str) -> None:
    """Attach or detach the console reporter to match `mode`. Assumes `_lock` is held.

    The reporter is attached only when it would actually render — a real terminal, or
    ``progress="on"``. Attaching it to a redirected stream would put a stripped-down copy
    of every query summary into the user's log file, which is the opposite of the clean
    output this whole path exists to produce.
    """
    global _console, _console_detach
    wanted = should_render(mode)
    if wanted == (_console is not None):
        return
    if not wanted:
        if _console_detach is not None:
            _console_detach()
        _console, _console_detach = None, None
        suppress_console_handler(False)
        return
    # The reporter takes over the terminal: it renders log records itself, correctly
    # interleaved with the progress bar, so the plain stderr handler steps aside.
    suppress_console_handler(True)
    _console = ConsoleReporter(live=True)
    _console_detach = _console.attach()


def start_ui(*, port: int = 4040, host: str = "127.0.0.1", open_browser: bool = False) -> str:
    """Start the Batcher web dashboard and return its URL.

    Serves the query list, plan DAG, per-operator timings, and live log stream on its own
    port, so it never interferes with the process it observes. Idempotent — calling it while
    a dashboard is already running returns the existing URL rather than binding a second
    port. The server thread is a daemon, and is stopped at interpreter exit.

    Binds to loopback by default. The dashboard shows query text, plans, and logs; serving
    those on a routable address should be a deliberate choice, so `host` must be set
    explicitly to make that happen.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> url = bt.start_ui(port=0)  # port=0 lets the OS pick a free one
            >>> url.startswith("http://127.0.0.1:")
            True
            >>> bt.stop_ui()

    Args:
        port: TCP port to bind; 0 asks the OS for any free port.
        host: Interface to bind. Loopback by default.
        open_browser: Open the dashboard in the default browser once it is listening.

    Returns:
        The URL the dashboard is reachable at, e.g. ``http://127.0.0.1:4040``.

    Raises:
        OSError: If neither `port` nor any of the following ports could be bound.
    """
    global _server
    with _lock:
        if _server is not None:
            return _server.url
        server, url, fallback = _bind(host, port)
        _server = server
    # Logging is what tells a user in a terminal where the dashboard actually went — which
    # matters most in the `port=0` case, where they did not choose the number, and in the
    # fallback case, where the number they chose is not the one they got.
    ensure_configured()
    log = get_logger("observe")
    if fallback:
        log.warning("Batcher UI: port %d was busy, listening on %s instead", port, url)
    else:
        log.warning("Batcher UI listening on %s", url)
    if open_browser:
        webbrowser.open(url)
    return url


#: How many consecutive ports to try after the requested one before giving up. A dashboard
#: is a convenience, so a busy port should move it rather than fail the call — but silently
#: scanning a wide range would be its own surprise, so the walk is short and logged.
_PORT_FALLBACK_TRIES = 20


def _bind(host: str, port: int) -> tuple[UIServer, str, bool]:
    """Bind the dashboard, walking forward from `port` if it is already taken.

    Returns the started server, its URL, and whether a fallback port was used. `port=0`
    already means "any free port" to the OS, so it is never walked.
    """
    # The HTTP server, its routes, and the JSON handlers are imported the first time a
    # dashboard is actually started, not when `batcher` is. Nothing else in the package
    # touches them, and they were a third of what importing `observe` cost — paid by every
    # process, to be ready for a UI almost none of them open.
    from batcher.observe.server import UIServer

    last: OSError | None = None
    tries = 1 if port == 0 else _PORT_FALLBACK_TRIES
    for offset in range(tries):
        server = UIServer(ActivityStore(), host=host, port=port + offset)
        try:
            return server, server.start(), offset > 0
        except OSError as exc:
            last = exc
    msg = (
        f"could not start the Batcher UI: ports {port}-{port + tries - 1} on {host} are all "
        f"in use. Pass a different port, or port=0 to let the OS choose one."
    )
    raise OSError(msg) from last


def stop_ui() -> None:
    """Stop the web dashboard and detach it from the event bus.

    Safe to call when no dashboard is running, and safe to call twice — so shutting down in
    a `finally` block or an exit hook needs no guard of its own.

    An explicit stop sticks: even with ``observability.ui=True``, the next query will not
    silently restart the dashboard. Call `start_ui` again to bring it back.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.stop_ui()  # no-op when nothing is running

    Returns:
        None.
    """
    global _server
    with _lock:
        server, _server = _server, None
    if server is not None:
        server.stop()
    # The dashboard is the usual reason a device sampler is running, and a stopped dashboard
    # with a thread still hitting NVML every second is a leak nobody would look for. The stop
    # sticks for the same reason the dashboard's does — `ensure_sinks` short-circuits on an
    # unchanged config — so a process that wants sampling back changes the setting or calls
    # `start_device_series` itself. The accumulated window survives either way.
    from batcher.observe.accelerators.series import sampling_active, stop_device_series

    if sampling_active():
        stop_device_series()


def ui_url() -> str | None:
    """The running dashboard's URL, or `None` when no dashboard is running.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.ui_url() is None
            True

    Returns:
        The dashboard URL, or None if `start_ui` has not been called.
    """
    return _server.url if _server is not None else None


# A dashboard left running must not outlive the process that started it, and must release
# its port even on an abrupt teardown.
atexit.register(stop_ui)
