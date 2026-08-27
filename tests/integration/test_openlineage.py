"""OpenLineage run events carry the column lineage the engine already computes.

The emit is exercised against a **real HTTP receiver** on localhost rather than a patched
transport, because the parts most likely to break are the ones a patch removes: the
background drain thread, the URL the POST is built against, and the bearer header. A test
that stubbed `_post` would pass with every one of those wrong.

The distributed shape is asserted separately, off a constructed profile, because the facet
that distinguishes a cluster run from a local one must be checkable without a cluster —
otherwise the one property CI cannot see is also the one nothing checks.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import batcher as bt
from batcher.api.terminal.lineage import _batcher_run_facet, _build
from batcher.config import Config, ObservabilityConfig, active_config, config_context

pytestmark = pytest.mark.integration


class _Receiver(BaseHTTPRequestHandler):
    """Collects posted lineage events into the server's `events` list."""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self.server.events.append(  # type: ignore[attr-defined]
            {
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "event": json.loads(body),
            }
        )
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        """Silence the stdlib access log."""


@pytest.fixture()
def receiver():
    """A localhost lineage receiver, yielding the list it collects events into."""
    server = HTTPServer(("127.0.0.1", 0), _Receiver)
    server.events = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _drain(server, *, until: str | None = None, expected: int = 1, timeout: float = 10.0):
    """Wait until `until` has arrived (or `expected` events have), then return them all.

    Waiting on a *count* is what made this flaky: the START of a failing query satisfies
    ``expected=1`` immediately, so the FAIL that follows it was still in flight when the
    assertion ran. Waiting on the event type the test is actually about does not race.
    """
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        kinds = [p["event"]["eventType"] for p in server.events]
        if (until in kinds) if until else (len(kinds) >= expected):
            break
        time.sleep(0.02)
    return list(server.events)


def _with_lineage(**fields) -> Config:
    """The active config with `fields` overridden on its observability block."""
    base = active_config()
    return base.replace(
        observability=ObservabilityConfig(
            **{**{f.name: getattr(base.observability, f.name) for f in _obs_fields()}, **fields}
        )
    )


def _obs_fields():
    import dataclasses

    return dataclasses.fields(ObservabilityConfig)


def _url(server) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}"


def test_a_query_emits_start_and_complete_with_column_lineage(receiver, tmp_path):
    """A read-and-project query posts START then COMPLETE, carrying real column origins."""
    path = str(tmp_path / "people.parquet")
    bt.from_pydict({"first": ["a"], "last": ["b"], "age": [30]}).write(path, format="parquet")

    with config_context(
        _with_lineage(
            openlineage=True,
            openlineage_url=_url(receiver),
            openlineage_namespace="test-ns",
            openlineage_api_key="s3cret",
        )
    ):
        result = (
            bt.read.parquet(path).select(name=bt.concat(bt.col("first"), bt.col("last"))).collect()
        )
    assert result.num_rows == 1

    posts = _drain(receiver, until="COMPLETE")
    kinds = [p["event"]["eventType"] for p in posts]
    assert "START" in kinds and "COMPLETE" in kinds, kinds

    # The transport details a patched `_post` would have hidden.
    assert all(p["path"] == "/api/v1/lineage" for p in posts)
    assert all(p["auth"] == "Bearer s3cret" for p in posts)

    complete = next(p["event"] for p in posts if p["event"]["eventType"] == "COMPLETE")
    assert complete["job"]["namespace"] == "test-ns"
    assert [i["name"] for i in complete["inputs"]] == [path]

    # The positive control: `name` is built from exactly `first` and `last`, so those two
    # origins must be present. Without this, an assertion that some other column is absent
    # would only be a claim about the renderer.
    facet = complete["outputs"][0]["facets"]["columnLineage"]
    origins = {(f["name"], f["field"]) for f in facet["fields"]["name"]["inputFields"]}
    assert origins == {(path, "first"), (path, "last")}
    # `age` was never selected, so it derives nothing and must not appear as an origin.
    assert not any(field == "age" for _, field in origins)

    # START and COMPLETE must describe one run, not two.
    start = next(p["event"] for p in posts if p["event"]["eventType"] == "START")
    assert start["run"]["runId"] == complete["run"]["runId"]
    assert start["job"]["name"] == complete["job"]["name"]


def test_a_failing_query_closes_its_run(receiver):
    """A query that raises emits FAIL, so the run does not stay open forever.

    The failure has to happen during *execution*, not during plan construction: a plan
    that never builds never opens a run, so there is nothing to close. A raising
    `map_batches` is the cheapest way to fail inside the engine.
    """
    with config_context(_with_lineage(openlineage=True, openlineage_url=_url(receiver))):

        def _boom(batch):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            bt.from_pydict({"x": [1, 2]}).map_batches(_boom).collect()

    posts = _drain(receiver, until="FAIL")
    kinds = [p["event"]["eventType"] for p in posts]
    assert "FAIL" in kinds, kinds
    fail = next(p["event"] for p in posts if p["event"]["eventType"] == "FAIL")
    assert fail["run"]["facets"]["errorMessage"]["message"]


def test_nothing_is_emitted_when_the_feature_is_off(receiver):
    """The default configuration posts nothing at all, endpoint reachable or not."""
    prior = os.environ.pop("OPENLINEAGE_URL", None)
    try:
        with config_context(_with_lineage(openlineage_url=_url(receiver))):
            bt.from_pydict({"x": [1, 2]}).agg(s=bt.col("x").sum()).collect()
        assert _drain(receiver, expected=1, timeout=1.0) == []
    finally:
        if prior is not None:
            os.environ["OPENLINEAGE_URL"] = prior


def test_the_run_facet_distinguishes_a_cluster_run(monkeypatch):
    """A distributed profile is reported as distributed, with its worker operators counted.

    This is the property CI has no cluster to observe, so it is asserted against a
    constructed profile: the facet must not report a many-node run as a local one.
    """
    from batcher.plan.profile.types import OpProfile, QueryProfile

    worker = OpProfile(op_id=1, kind="aggregate", depth=0, measured=True, rows_in=100, rows_out=4)
    local = QueryProfile(ops=(), query_id="q1", rows=4, total_ms=12.0)
    cluster = QueryProfile(
        ops=(),
        query_id="q1",
        rows=4,
        total_ms=12.0,
        distributed=True,
        worker_ops=(worker, worker),
    )

    assert _batcher_run_facet(local)["distributed"] is False
    assert _batcher_run_facet(local)["workerOperators"] == 0

    facet = _batcher_run_facet(cluster)
    assert facet["distributed"] is True
    assert facet["workerOperators"] == 2


def test_the_same_plan_gets_the_same_job_name_on_both_paths():
    """Job identity is the plan signature, so a run's history groups across executions."""
    ds = bt.from_pydict({"x": [1, 2, 3]}).filter(bt.col("x") > 1)
    other = bt.from_pydict({"x": [9, 9]}).filter(bt.col("x") > 1)
    unrelated = bt.from_pydict({"x": [1, 2, 3]}).agg(s=bt.col("x").sum())

    name = _build("q", "START", plan=ds._plan, sources=ds._sources, profile=None)["job"]["name"]
    same = _build("q", "START", plan=other._plan, sources=other._sources, profile=None)["job"][
        "name"
    ]
    different = _build(
        "q", "START", plan=unrelated._plan, sources=unrelated._sources, profile=None
    )["job"]["name"]

    assert name == same
    assert name != different
