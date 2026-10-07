"""A real local HTTP server for the API-source tests, driven by a Python handler.

The HTTP sources are tested over a socket rather than with a patched `urlopen`, so the
request a source sends (method, path, query string, headers, body) is the bytes on the
wire, and redirects, status codes and headers go through `urllib` exactly as they would
against a real API. The handler is a plain function of the recorded request; every request
is kept on ``server.requests`` so a test can pin what was sent.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit


@dataclass
class FakeRequest:
    """One request as the server received it."""

    method: str
    path: str
    query: dict[str, str]
    headers: dict[str, str]
    body: bytes = b""

    def json(self) -> Any:
        return json.loads(self.body or b"null")


@dataclass
class FakeResponse:
    """What the handler answers: a status, headers, and a body (bytes or a JSON value)."""

    status: int = 200
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)


Handler = Callable[[FakeRequest], FakeResponse]


class FakeApi:
    """A threaded HTTP server on an ephemeral localhost port.

    Args:
        handler: Called once per request; returns the response to send.
    """

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[FakeRequest] = []
        api = self

        class _Handler(BaseHTTPRequestHandler):
            def _serve(self) -> None:
                parts = urlsplit(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                request = FakeRequest(
                    method=self.command,
                    path=parts.path,
                    query={k: v[-1] for k, v in parse_qs(parts.query).items()},
                    headers={k.lower(): v for k, v in self.headers.items()},
                    body=self.rfile.read(length) if length else b"",
                )
                api.requests.append(request)
                response = api.handler(request)
                body = response.body
                if not isinstance(body, bytes):
                    body = json.dumps(body).encode()
                self.send_response(response.status)
                for name, value in response.headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _serve

            def log_message(self, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> FakeApi:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
