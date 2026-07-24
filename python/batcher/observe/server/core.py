"""The dashboard's lifecycle: bind a port, serve in a daemon thread, detach cleanly.

Built on `http.server`. That is a deliberate ceiling, not an oversight: the payload is a
handful of small JSON documents polled by one or two browser tabs on localhost, so a real
ASGI stack would buy nothing and cost every user of the engine a web framework dependency.
If this ever needs to serve a cluster rather than a laptop, it wants a different design,
not a bigger thread pool.

**Binds to loopback by default.** The dashboard exposes query text, plans, and logs — the
contents of the data, in effect. Serving that on `0.0.0.0` is a decision the operator makes
explicitly by setting `ui_host`, never a default they back into.
"""

from __future__ import annotations

import threading
from functools import partial
from http.server import ThreadingHTTPServer
from typing import Any

from batcher.observe.server.handler import Handler
from batcher.observe.store import ActivityStore

__all__ = ["UIServer"]


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
        handler = partial(Handler, self._store)
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
