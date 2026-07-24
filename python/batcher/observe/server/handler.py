"""The HTTP layer: dispatch, compression, caching, and the headers a browser needs.

Deliberately small. The payload is a handful of JSON documents polled by one or two tabs on
localhost, so this is `http.server` and not a framework — but "small" is not the same as
"sloppy", and the things below are the ones whose absence a real browser notices:

* **Conditional requests.** The app shell and its assets are static for the life of the
  process, so they carry a strong `ETag` and answer a re-request with `304`. Without it a
  reload re-downloads ~250 KB of unchanged JavaScript.
* **Compression.** The JSON documents are highly repetitive and gzip to roughly a fifth of
  their size. That matters most on the poll loop, which fetches eight of them a second.
* **`HEAD`.** A health check that cannot ask "is this up" without downloading the page is
  a health check that costs more than the thing it checks.
* **Security headers.** The dashboard renders query text and log lines, which are user
  data. `nosniff` plus a `self`-only CSP means a log line containing markup can never be
  interpreted as anything but text, whatever a future renderer forgets to escape.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from batcher._internal.logging import get_logger
from batcher.observe.metrics import prometheus_text
from batcher.observe.server import routes
from batcher.observe.store import ActivityStore

__all__ = ["Handler"]

_ASSETS = Path(__file__).resolve().parent.parent / "assets"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".map": "application/json",
    ".txt": "text/plain; charset=utf-8",
    ".woff2": "font/woff2",
    ".ico": "image/x-icon",
}

#: Below this, compressing costs more CPU and bytes (the gzip header) than it saves.
_GZIP_MIN_BYTES = 1024

#: A `self`-only policy. The dashboard loads nothing from anywhere else — no CDN, no font
#: host, no analytics — so the strictest policy that lets it work is the correct one, and it
#: is what stops a log line or a query label from ever becoming executable.
_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self'; font-src 'self'; "
    "object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


class Handler(BaseHTTPRequestHandler):
    """Routes: the static single-page app, and a read-only JSON API over the store."""

    # HTTP/1.1 so the browser keeps the connection alive across the UI's poll loop
    # instead of reconnecting several times a second.
    protocol_version = "HTTP/1.1"
    #: Identify as ourselves rather than as the Python version, which is a free hint to
    #: anyone who reaches a dashboard that was bound wider than loopback.
    server_version = "batcher-ui"
    sys_version = ""

    def __init__(self, store: ActivityStore, *args: Any, **kwargs: Any) -> None:
        self._store = store
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        """Dispatch a GET to the API or the static assets."""
        self._dispatch(body=True)

    def do_HEAD(self) -> None:
        """Answer a HEAD exactly as the GET would, minus the body."""
        self._dispatch(body=False)

    def _dispatch(self, *, body: bool) -> None:
        parsed = urlparse(self.path)
        route = unquote(parsed.path).rstrip("/") or "/"
        try:
            if route == "/metrics":
                text = prometheus_text().encode()
                self._respond(200, text, "text/plain; charset=utf-8", body=body)
            elif route.startswith("/api"):
                self._serve_json(route, parse_qs(parsed.query), body=body)
            else:
                self._serve_asset(route, body=body)
        except BrokenPipeError:  # pragma: no cover - the browser navigated away mid-write
            pass
        except ConnectionResetError:  # pragma: no cover - the tab was closed mid-write
            pass
        except Exception:  # pragma: no cover - a UI bug must not kill the server thread
            get_logger("observe").debug("UI request failed: %s", self.path, exc_info=True)
            self._respond(500, b'{"error":"internal"}', "application/json", body=body)

    #: The one write the dashboard makes. Naming a pipeline is durable state a person sets,
    #: so it cannot be a GET — but it touches only the pipeline registry (a file of names and
    #: notes), never the engine or a run, which is what keeps "the API cannot change a result"
    #: true even with a write on it.
    _WRITE_ROUTE = "/api/pipeline/meta"
    #: A body larger than this is not a pipeline name; reject it before reading it, so the
    #: endpoint cannot be used to make the dashboard buffer an arbitrary amount of memory.
    _MAX_BODY = 8192

    def do_POST(self) -> None:
        """Handle the single write route; reject every other mutation."""
        route = unquote(urlparse(self.path).path).rstrip("/")
        if route != self._WRITE_ROUTE:
            self._reject_method()
            return
        try:
            self._write_pipeline_meta()
        except BrokenPipeError:  # pragma: no cover
            pass
        except Exception:  # pragma: no cover - a write bug must not kill the server thread
            get_logger("observe").debug("UI write failed: %s", self.path, exc_info=True)
            self._respond(500, b'{"error":"internal"}', "application/json")

    def do_PUT(self) -> None:
        """PUT/DELETE/PATCH are never served: the one write is a POST."""
        self._reject_method()

    do_DELETE = do_PUT
    do_PATCH = do_PUT

    def _reject_method(self) -> None:
        """Every mutating verb we don't serve, answered once."""
        self.send_response(405)
        self.send_header("Allow", "GET, HEAD, POST")
        self.send_header("Content-Length", "0")
        self._common_headers()
        self.end_headers()

    def _write_pipeline_meta(self) -> None:
        """Rename or annotate a pipeline from a JSON body, then return the updated record.

        Validates before touching the registry: a body must be small, JSON, an object, and
        carry a `pipeline_id`. A bad request gets a 400 with a reason rather than a 500 or a
        silent no-op, because a control that fails invisibly is worse than one that errors.
        """
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > self._MAX_BODY:
            self._respond(400, b'{"error":"missing or oversized body"}', "application/json")
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except ValueError:
            self._respond(400, b'{"error":"body is not valid JSON"}', "application/json")
            return
        if not isinstance(payload, dict) or not payload.get("pipeline_id"):
            self._respond(400, b'{"error":"pipeline_id is required"}', "application/json")
            return
        # `name`/`note` are passed through only when present, so setting one leaves the other
        # untouched; a non-string is coerced rather than rejected, since the registry clamps.
        name = str(payload["name"]) if "name" in payload else None
        note = str(payload["note"]) if "note" in payload else None
        result = self._store.set_pipeline_meta(str(payload["pipeline_id"]), name=name, note=note)
        self._respond(200, json.dumps(result).encode(), "application/json")

    def _serve_json(self, route: str, query: dict[str, list[str]], *, body: bool) -> None:
        """Serve one of the read-only API routes."""
        if route == "/api":
            payload: Any = routes.index()
        elif route.startswith("/api/query/"):
            payload = self._store.query(route.rsplit("/", 1)[-1])
            if payload is None:
                self._respond(404, b'{"error":"unknown query"}', "application/json", body=body)
                return
        else:
            handler = routes.resolve(route)
            if handler is None:
                self._respond(404, b'{"error":"unknown route"}', "application/json", body=body)
                return
            payload = handler(self._store, query)
        encoded = json.dumps(payload, default=str, separators=(",", ":")).encode()
        self._respond(200, encoded, "application/json", body=body)

    def _serve_asset(self, route: str, *, body: bool) -> None:
        """Serve a file from `assets/`, defaulting `/` to the app shell."""
        name = "index.html" if route == "/" else route.lstrip("/")
        path = (_ASSETS / name).resolve()
        # Containment check: the UI serves a fixed asset directory, so a path that escapes
        # it is a traversal attempt, not a missing file.
        if not path.is_file() or _ASSETS not in path.parents:
            self._respond(404, b"not found", "text/plain; charset=utf-8", body=body)
            return
        content_type = _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        self._respond(200, path.read_bytes(), content_type, cacheable=True, body=body)

    def _respond(
        self,
        status: int,
        payload: bytes,
        content_type: str,
        *,
        cacheable: bool = False,
        body: bool = True,
    ) -> None:
        """Write one response, negotiating compression and revalidation along the way."""
        headers: list[tuple[str, str]] = []
        if cacheable:
            # A strong validator over the exact bytes: the assets are read from disk on every
            # request (so an edit during development is picked up), and hashing them is far
            # cheaper than shipping them.
            etag = f'"{hashlib.blake2b(payload, digest_size=16).hexdigest()}"'
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self.send_header("ETag", etag)
                self._common_headers()
                self.end_headers()
                return
            headers.append(("ETag", etag))
            # `no-cache` means "revalidate", not "do not store": the browser keeps the copy
            # and asks whether it is still current, which is what makes the 304 above happen.
            headers.append(("Cache-Control", "no-cache"))
        else:
            # The dashboard polls; a cached /api/queries would freeze the view.
            headers.append(("Cache-Control", "no-store"))

        if len(payload) >= _GZIP_MIN_BYTES and "gzip" in self.headers.get("Accept-Encoding", ""):
            payload = gzip.compress(payload, compresslevel=6)
            headers.append(("Content-Encoding", "gzip"))
            headers.append(("Vary", "Accept-Encoding"))

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        for name, value in headers:
            self.send_header(name, value)
        self._common_headers()
        self.end_headers()
        if body:
            self.wfile.write(payload)

    def _common_headers(self) -> None:
        """The headers every response carries, whatever it is."""
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", _CSP)

    def log_message(self, format: str, *args: Any) -> None:
        """Silence the stdlib per-request stderr line; route it to our DEBUG logger.

        Without this override, `http.server` prints a line to stderr for every poll — a
        dashboard that spams the terminal it exists to keep clean.
        """
        get_logger("observe").debug("ui %s", format % args)
