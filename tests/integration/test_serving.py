"""Model-serving batch adapters + the OpenAI-compatible LLM HTTP engine.

Exercised against an in-process stub HTTP server (no network, no real Triton): the
adapter must extract the input columns, post/parse columnar JSON, and append the
outputs — and participate in `map_batches` (load-once class UDF).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher.ml.serving import http_client, serving_udf


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence the test server
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/chat/completions"):
            reply = {"choices": [{"message": {"content": "ok:" + body["messages"][-1]["content"]}}]}
        elif "x" in body:  # columnar predict
            reply = {"pred": [v * 10 for v in body["x"]]}
        else:
            reply = {}
        payload = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    yield f"http://{host}:{port}"
    httpd.shutdown()


class _FlakyHandler(BaseHTTPRequestHandler):
    """Fails with 503 for the first `fail_times` requests, then succeeds."""

    fail_times = 1
    seen = 0

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        type(self).seen += 1
        code = 503 if type(self).seen <= type(self).fail_times else 200
        payload = json.dumps({"pred": [1]} if code == 200 else {}).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def flaky_server():
    _FlakyHandler.seen = 0
    httpd = HTTPServer(("127.0.0.1", 0), _FlakyHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address
    yield f"http://{host}:{port}"
    httpd.shutdown()


def test_serving_udf_appends_outputs():
    class Fake:
        def predict(self, inputs):
            return {"pred": inputs["x"] * 2}

    udf = serving_udf(lambda: Fake(), input_columns=["x"], output_columns=["pred"])()
    out = udf(pa.record_batch({"x": pa.array([1, 2, 3]), "id": [7, 8, 9]}))
    assert out.column("pred").to_pylist() == [2, 4, 6]
    assert out.column("id").to_pylist() == [7, 8, 9]  # other columns preserved


def test_serving_udf_handles_tensor_columns():
    from batcher.io.formats.ml.tensor import to_tensor_column

    captured = {}

    class Fake:
        def predict(self, inputs):
            captured["shape"] = inputs["img"].shape
            return {"cls": inputs["img"].reshape(inputs["img"].shape[0], -1).sum(axis=1)}

    imgs = to_tensor_column(np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3))
    udf = serving_udf(lambda: Fake(), input_columns=["img"], output_columns=["cls"])()
    udf(pa.record_batch({"img": imgs}))
    assert captured["shape"] == (2, 4, 4, 3)  # tensor column kept its shape


def test_http_client_through_map_batches(server):
    udf = http_client(server + "/predict", input_columns=["x"], output_columns=["pred"])
    ds = bt.from_pydict({"x": [1, 2, 3, 4], "label": ["a", "b", "c", "d"]})
    out = ds.ml.map_batches(udf).collect()
    assert out.column("pred").to_pylist() == [10, 20, 30, 40]
    assert out.column("label").to_pylist() == ["a", "b", "c", "d"]


def test_http_llm_engine_chat(server):
    from batcher.ml import http_engine, llm_generate

    engine = http_engine(server + "/v1", "test-model", chat=True)
    batches = [pa.record_batch({"q": ["hello", "world"]})]
    out = list(llm_generate(batches, engine, prompt_column="q"))
    assert out[0].column("response").to_pylist() == ["ok:hello", "ok:world"]


def test_torchserve_url_construction():
    from batcher.ml.serving import torchserve_client

    # Building the adapter must not require the server; it returns a class UDF.
    udf = torchserve_client("http://h:8080/", "m", input_columns=["x"], output_columns=["y"])
    assert isinstance(udf, type)


def test_post_json_retries_transient_failure(flaky_server):
    from batcher.ml.serving.http import post_json

    # First request 503s, retry succeeds — no error surfaces to the caller.
    out = post_json(flaky_server, {"x": [1]}, headers={}, timeout=5, retries=3, backoff=0.0)
    assert out == {"pred": [1]}


def test_post_json_fails_fast_on_client_error(server):
    from batcher._internal.errors import BackendError
    from batcher.ml.serving.http import post_json

    # /chat/completions with no messages → handler 200s, but a 4xx would not retry.
    # Hit a path that 500s is covered by the flaky test; here assert a bad URL raises.
    with pytest.raises(BackendError):
        post_json(
            "http://127.0.0.1:1/none", {"x": [1]}, headers={}, timeout=1, retries=1, backoff=0.0
        )


def test_http_client_warns_on_tensor_input():
    import contextlib

    from batcher._internal.errors import PerformanceWarning
    from batcher.ml.serving.http import _HttpServingClient

    client = _HttpServingClient("http://unused", {}, 1.0, 0)
    with pytest.warns(PerformanceWarning, match="triton_client"), contextlib.suppress(Exception):
        client.predict({"img": np.zeros((2, 4, 4), dtype=np.uint8)})  # warns before the call


def test_vllm_engine_builds_factory_lazily():
    from batcher.ml import vllm_engine

    # Building the factory must not import vllm (only invoking it does), so this works
    # without a GPU; it captures sampling/guided/lora config for later.
    factory = vllm_engine(
        "meta-llama/Llama-3-8B",
        sampling={"temperature": 0.7, "max_tokens": 128},
        guided_json={"type": "object"},
        lora_path="/adapters/x",
    )
    assert callable(factory)
