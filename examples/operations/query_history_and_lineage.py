"""Reading back what ran: query history, column lineage, and OpenLineage events.

Three views of the same queries, from three places:

* `bt.query_history()` reads the event log the engine already writes, one document per
  completed query, and returns it as a `Dataset` you can filter and sort.
* `Dataset.lineage()` answers "which source columns does each output column come from"
  off the plan, without running anything.
* With `observability.openlineage` on, every query posts an OpenLineage START and then a
  COMPLETE (or FAIL) event carrying that lineage to a receiver such as Marquez. Here the
  receiver is a small HTTP server in this process, so nothing leaves the machine.

Everything is written under a temporary directory, never under `~/.batcher`.

    python examples/operations/query_history_and_lineage.py
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import batcher as bt
from batcher import col
from batcher.config import active_config, config_context


class _Receiver(HTTPServer):
    """A stand-in for a lineage backend: it keeps every event posted to it."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.events: list[dict] = []

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"

    def wait_for(self, count: int, timeout: float = 10.0) -> list[dict]:
        """Events are posted from a background thread, so wait for them to arrive."""
        deadline = time.monotonic() + timeout
        while len(self.events) < count and time.monotonic() < deadline:
            time.sleep(0.02)
        return list(self.events)


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        assert self.path == "/api/v1/lineage"
        self.server.events.append(json.loads(body))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        """Keep the example's output to what it prints on purpose."""


def main() -> None:
    receiver = _Receiver()
    threading.Thread(target=receiver.serve_forever, daemon=True).start()

    with tempfile.TemporaryDirectory() as tmp:
        log_dir = str(Path(tmp) / "logs")
        base = active_config()
        cfg = base.replace(
            observability=dataclasses.replace(
                base.observability,
                event_log=True,
                event_log_dir=log_dir,
                openlineage=True,
                openlineage_url=receiver.url,
                openlineage_namespace="example",
            )
        )
        orders = bt.from_pydict({"region": ["eu", "us", "eu", "us"], "amount": [5, 7, 11, 13]})
        regions = bt.from_pydict({"region": ["eu", "us"], "manager": ["ana", "bo"]})

        by_region = orders.group_by("region").agg(total=col("amount").sum()).sort("region")
        report = by_region.join(regions, on="region").sort("region")

        # Lineage is read off the plan: `total` comes from `amount`, `manager` from the
        # second input. In-memory inputs have no path, so they are labelled by position.
        lineage = report.lineage()
        print("lineage:", lineage)
        assert lineage["total"] == ["<source 0>.amount"]
        assert lineage["manager"] == ["<source 1>.manager"]

        with config_context(cfg):
            assert by_region.to_pydict()["total"] == [16, 20]
            assert report.to_pydict()["manager"] == ["ana", "bo"]
            # Two queries, each a START then a COMPLETE. They are posted from a background
            # thread, so wait for them rather than reading the list straight away.
            events = receiver.wait_for(4)
        receiver.shutdown()

        # The history is itself a query, so it is read after lineage emission is switched
        # off, from the directory the event log wrote. One row per completed query; the
        # ids sort chronologically.
        history = bt.query_history(log_dir).sort("query_id")
        rows = history.select("query_id", "rows_produced", "total_elapsed_ms").to_pydict()
        print("history:", rows)
        assert rows["rows_produced"] == [2, 2]
        assert all(ms > 0 for ms in rows["total_elapsed_ms"])

    kinds = sorted(e["eventType"] for e in events)
    print("lineage events:", kinds)
    assert kinds == ["COMPLETE", "COMPLETE", "START", "START"]

    # Each run's START and COMPLETE name one job, so the backend sees one run per query,
    # opened and closed on the same job.
    runs: dict[str, dict[str, dict]] = {}
    for event in events:
        runs.setdefault(event["run"]["runId"], {})[event["eventType"]] = event
    assert len(runs) == 2
    for run in runs.values():
        assert run["START"]["job"]["name"] == run["COMPLETE"]["job"]["name"]

    # An in-memory input is named per object: `orders` is the same dataset in both runs,
    # and `regions` is a different one, rather than every input being `<source 0>`.
    first, second = sorted(
        ([i["name"] for i in run["START"]["inputs"]] for run in runs.values()), key=len
    )
    print("input datasets:", first, second)
    assert first == second[:1]
    assert second[0] != second[1]


if __name__ == "__main__":
    main()
