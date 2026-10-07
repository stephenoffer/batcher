"""Stable per-row request ids for remote model calls (AP-387).

The acceptance is that a retried remote request stays traceable to the same source row. It
is checked against a real local HTTP server that fails the first attempt of every request
with a 503, so the id on the retry is observed on the wire rather than assumed.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.llm.generate import _dedup_requests, _generate_batch
from batcher.ml.llm.requests import GenerateSpec, request_ids

pytestmark = pytest.mark.unit


class _FlakyServer:
    """An OpenAI-shaped endpoint that 503s each prompt's first attempt and records headers."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str | None]] = []
        attempts: dict[str, int] = {}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                prompt = body["messages"][-1]["content"]
                owner.seen.append((prompt, self.headers.get("X-Client-Request-Id")))
                attempts[prompt] = attempts.get(prompt, 0) + 1
                if attempts[prompt] == 1:
                    self.send_response(503)
                    self.end_headers()
                    return
                reply = {"choices": [{"message": {"content": prompt.upper()}}]}
                data = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: object) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()


@pytest.fixture
def server():
    s = _FlakyServer()
    yield s
    s.close()


def test_a_retried_request_resends_the_same_derived_id(server):
    ds = bt.from_pydict({"q": ["alpha", "beta"]})
    engine = bt.ml.http_engine(server.url, "m", concurrency=1, retries=2, backoff=0.0)
    out = ds.ml.generate(engine, prompt_column="q", request_id_column="rid").to_pydict()
    assert out["response"] == ["ALPHA", "BETA"]
    by_prompt: dict[str, set] = {}
    for prompt, rid in server.seen:
        by_prompt.setdefault(prompt, set()).add(rid)
    assert len(server.seen) == 4  # each prompt was attempted twice
    for prompt, rid in zip(out["q"], out["rid"], strict=True):
        assert by_prompt[prompt] == {rid}  # both attempts carried the recorded id
        assert rid.startswith("bt-") and rid.isascii() and len(rid) == 35


def test_a_supplied_id_column_is_sent_verbatim_and_kept(server):
    ds = bt.from_pydict({"q": ["alpha"], "order_id": [981]})
    engine = bt.ml.http_engine(server.url, "m", concurrency=1, retries=2, backoff=0.0)
    out = ds.ml.generate(engine, prompt_column="q", request_id_column="order_id").to_pydict()
    assert out["order_id"] == [981]  # the source column is untouched, not re-typed
    assert {rid for _p, rid in server.seen} == {"981"}


def test_the_header_name_is_configurable(monkeypatch):
    import batcher.ml.serving.http as http_mod

    seen: list[dict] = []

    def fake(url, body, *, headers, **kw):
        seen.append(headers)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(http_mod, "post_json", fake)
    engine = bt.ml.http_engine(
        "http://x/v1", "m", concurrency=1, request_id_header="Idempotency-Key"
    )()
    engine([{"prompt": "p", "request_id": "r-1"}, "plain"])
    assert seen[0]["Idempotency-Key"] == "r-1"
    assert "Idempotency-Key" not in seen[1]  # no id, no header


def test_derived_ids_do_not_depend_on_how_rows_are_batched():
    batch = pa.RecordBatch.from_pydict({"q": ["a", "b", "c", "b"], "k": [1, 2, 3, 4]})
    spec = GenerateSpec(prompt_column="q", request_id_column="rid")
    whole = request_ids(spec, batch, [{"prompt": p} for p in batch.column("q").to_pylist()])
    halves = [
        request_ids(spec, part, [{"prompt": p} for p in part.column("q").to_pylist()])
        for part in (batch.slice(0, 2), batch.slice(2))
    ]
    assert whole == halves[0] + halves[1]
    assert whole[1] == whole[3]  # identical requests share an id without a key
    keyed = GenerateSpec(prompt_column="q", request_id_column="rid", request_id_key=("k",))
    ids = request_ids(keyed, batch, [{"prompt": p} for p in batch.column("q").to_pylist()])
    assert len(set(ids)) == 4  # the key separates them


def test_the_id_changes_with_a_per_row_override():
    batch = pa.RecordBatch.from_pydict({"q": ["a", "a"], "mt": [16, 2000]})
    spec = GenerateSpec(prompt_column="q", max_tokens_column="mt", request_id_column="rid")
    out = _generate_batch(lambda reqs: [r["prompt"] for r in reqs], batch, spec)
    rid = out.column("rid").to_pylist()
    assert rid[0] != rid[1]


def test_dedup_collapses_only_rows_sending_the_same_request_and_id():
    reqs = [
        {"prompt": "a", "request_id": "x"},
        {"prompt": "a", "request_id": "x"},
        {"prompt": "a", "request_id": "y"},
    ]
    uniques, inverse = _dedup_requests(reqs)
    assert len(uniques) == 2 and inverse == [0, 0, 1]


def test_request_id_options_are_validated():
    ds = bt.from_pydict({"q": ["a"], "k": [1]})

    def echo():
        return lambda reqs: [r["prompt"] for r in reqs]

    with pytest.raises(PlanError, match="needs request_id_column"):
        ds.ml.generate(echo, prompt_column="q", request_id_key="k")
    with pytest.raises(PlanError, match="request_id_key would be ignored"):
        ds.ml.generate(echo, prompt_column="q", request_id_column="k", request_id_key="k")
    with pytest.raises(PlanError, match="request_id_key"):
        ds.ml.generate(echo, prompt_column="q", request_id_column="rid", request_id_key="nope")
    with pytest.raises(PlanError, match="name them apart"):
        ds.ml.generate(echo, prompt_column="q", request_id_column="response")
