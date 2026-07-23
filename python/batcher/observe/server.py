"""The Batcher UI — a local web dashboard for queries, plans, metrics, and logs.

Serves the single-page app in `assets/` plus a small JSON API over an `ActivityStore`, on
its own port so it never interferes with the process it is observing. The Spark UI analog:
open it while a job runs and watch the plan, the per-operator throughput, the spill and
memory picture, and the log stream side by side.

Built on `http.server` in a daemon thread. That is a deliberate ceiling, not an oversight:
the payload is a handful of small JSON documents polled by one or two browser tabs on
localhost, so a real ASGI stack would buy nothing and cost every user of the engine a web
framework dependency. If this ever needs to serve a cluster rather than a laptop, it wants
a different design, not a bigger thread pool.

**Binds to loopback by default.** The dashboard exposes query text, plans, and logs — the
contents of the data, in effect. Serving that on `0.0.0.0` is a decision the operator makes
explicitly by setting `ui_host`, never a default they back into.
"""

from __future__ import annotations

import json
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from batcher._internal.logging import get_logger
from batcher.observe.store import ActivityStore
from batcher.observe.system import system_snapshot

__all__ = ["UIServer"]

_ASSETS = Path(__file__).parent / "assets"
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}


class UIServer:
    """A running dashboard: an HTTP server thread plus the store feeding it.

    Construct and `start`; the port actually bound is on `port` (which differs from the
    requested one when 0 was passed to let the OS choose). `stop` is idempotent and
    detaches the store from the bus, so a restarted UI does not accumulate sinks.
    """

    def __init__(
        self,
        store: ActivityStore,
        *,
        host: str = "127.0.0.1",
        port: int = 4040,
    ) -> None:
        self._store = store
        self._host = host
        self._requested_port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._detach: Any = None

    @property
    def port(self) -> int:
        """The bound port, or the requested one before `start`."""
        return self._httpd.server_address[1] if self._httpd else self._requested_port

    @property
    def url(self) -> str:
        """The dashboard URL, e.g. ``http://127.0.0.1:4040``."""
        return f"http://{self._host}:{self.port}"

    def start(self) -> str:
        """Bind the port, attach the store to the bus, and serve in a daemon thread.

        Returns:
            The URL the dashboard is reachable at.
        """
        if self._httpd is not None:
            return self.url
        handler = partial(_Handler, self._store)
        self._httpd = ThreadingHTTPServer((self._host, self._requested_port), handler)
        # Daemon so a forgotten `start_ui()` can never keep the interpreter alive at exit —
        # a dashboard must not turn a finished script into a hung process.
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="batcher-ui",
            daemon=True,
        )
        self._thread.start()
        self._detach = self._store.attach()
        return self.url

    def stop(self) -> None:
        """Shut the server down and detach the store. Safe to call more than once."""
        if self._detach is not None:
            self._detach()
            self._detach = None
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


class _Handler(BaseHTTPRequestHandler):
    """Routes: the static single-page app, and a read-only JSON API over the store."""

    # HTTP/1.1 so the browser keeps the connection alive across the UI's poll loop
    # instead of reconnecting several times a second.
    protocol_version = "HTTP/1.1"

    def __init__(self, store: ActivityStore, *args: Any, **kwargs: Any) -> None:
        self._store = store
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        """Dispatch a GET to the API or the static assets."""
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        try:
            if route.startswith("/api"):
                self._serve_json(route, parse_qs(parsed.query))
            else:
                self._serve_asset(route)
        except BrokenPipeError:  # pragma: no cover - the browser navigated away mid-write
            pass
        except Exception:  # pragma: no cover - a UI bug must not kill the server thread
            get_logger("observe").debug("UI request failed: %s", self.path, exc_info=True)
            self._respond(500, b'{"error":"internal"}', "application/json")

    def _serve_json(self, route: str, query: dict[str, list[str]]) -> None:
        """Serve one of the read-only API routes."""
        if route == "/api/summary":
            payload: Any = self._store.summary()
        elif route == "/api/queries":
            payload = {"queries": self._store.queries()}
        elif route == "/api/pipelines":
            payload = {"pipelines": self._store.pipelines()}
        elif route == "/api/system":
            payload = system_snapshot()
        elif route == "/api/timeseries":
            payload = self._store.timeseries()
        elif route == "/api/operators":
            payload = {"operators": self._store.operators()}
        elif route == "/api/failures":
            payload = {"groups": self._store.failures()}
        elif route == "/api/pipeline":
            payload = self._store.pipeline(_str_param(query, "signature"))
        elif route == "/api/health":
            payload = self._store.health(system_snapshot())
        elif route == "/api/compare":
            payload = self._store.compare(_str_param(query, "a"), _str_param(query, "b"))
        elif route.startswith("/api/query/"):
            payload = self._store.query(route.rsplit("/", 1)[-1])
            if payload is None:
                self._respond(404, b'{"error":"unknown query"}', "application/json")
                return
        elif route == "/api/logs":
            payload = self._store.logs(since=_int_param(query, "since"))
        else:
            self._respond(404, b'{"error":"unknown route"}', "application/json")
            return
        body = json.dumps(payload, default=str).encode()
        self._respond(200, body, "application/json")

    def _serve_asset(self, route: str) -> None:
        """Serve a file from `assets/`, defaulting `/` to the app shell."""
        name = "index.html" if route == "/" else route.lstrip("/")
        path = (_ASSETS / name).resolve()
        # Containment check: the UI serves a fixed asset directory, so a path that escapes
        # it is a traversal attempt, not a missing file.
        if not path.is_file() or _ASSETS.resolve() not in path.parents:
            self._respond(404, b"not found", "text/plain")
            return
        content_type = _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        self._respond(200, path.read_bytes(), content_type)

    def _respond(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # The dashboard polls; a cached /api/queries would freeze the view.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Silence the stdlib per-request stderr line; route it to our DEBUG logger.

        Without this override, `http.server` prints a line to stderr for every poll — a
        dashboard that spams the terminal it exists to keep clean.
        """
        get_logger("observe").debug("ui %s", format % args)


def _str_param(query: dict[str, list[str]], name: str) -> str:
    """Read one string query parameter, or `""` when absent."""
    values = query.get(name)
    return values[0] if values else ""


def _int_param(query: dict[str, list[str]], name: str, default: int = 0) -> int:
    """Read one integer query parameter, falling back to `default` on absent/garbage."""
    values = query.get(name)
    if not values:
        return default
    try:
        return int(values[0])
    except ValueError:
        return default
