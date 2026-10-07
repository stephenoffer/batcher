"""One `ProviderLimit` obeyed by several Ray workers at once (AP-392).

The acceptance is that several workers obey one configured provider quota. An HTTP endpoint
on the driver host counts the requests in flight; `ds.ml.generate` runs over it with
``collect(distributed=True, num_workers=2)``. The control runs the same job with per-worker
concurrency only and must exceed the quota, which proves the job really sent from more than
one worker at once, so the shared run staying at the quota is the quota's doing.

Runs against whatever cluster ``RAY_ADDRESS`` names (a fresh local one under
``RAY_ADDRESS=local``). On a multi-node cluster the endpoint binds every interface and is
reached at the driver's address, so workers on other nodes can call it.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture(autouse=True)
def _need_ray():
    pytest.importorskip("ray")


class _CountingServer:
    def __init__(self) -> None:
        self.peak = 0
        self.total = 0
        live = {"now": 0}
        lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers["Content-Length"]))
                with lock:
                    live["now"] += 1
                    owner.total += 1
                    owner.peak = max(owner.peak, live["now"])
                time.sleep(0.05)
                with lock:
                    live["now"] -= 1
                data = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: object) -> None:
                return

        self.server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        host = socket.gethostbyname(socket.gethostname())
        self.url = f"http://{host}:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()


def _run(url: str, shared: bt.ml.ProviderLimit | None) -> list:
    rows = 400
    ds = bt.from_arrow(pa.table({"q": [f"q{i}" for i in range(rows)]}))
    engine = bt.ml.http_engine(url, "m", concurrency=8, shared_limit=shared)
    out = ds.ml.generate(engine, prompt_column="q", batch_size=50)
    return out.collect(distributed=True, num_workers=2).column("response").to_pylist()


def test_two_workers_obey_one_concurrency_quota():
    control = _CountingServer()
    try:
        assert _run(control.url, None) == ["ok"] * 400
    finally:
        control.close()
    assert control.peak > 8, "the control never overlapped two workers; nothing is under test"

    server = _CountingServer()
    quota = bt.ml.ProviderLimit("test-two-workers", max_concurrency=3)
    try:
        assert _run(server.url, quota) == ["ok"] * 400
    finally:
        server.close()
        quota.close()
    assert server.total == 400
    assert server.peak <= 3
