"""Regressions for production hardening of model serving, the ML pipeline, and UDF drops.

Each test here pins a defect that passed every existing gate while being wrong at scale:

* `serve_deployment` ran the model forward pass **on the asyncio event loop**, so one
  GPU call stalled the replica and every request queued behind it.
* `serving_udf` issued one blocking request per batch — no split to the server's
  ``max_batch_size``, no in-flight pipelining, no warmup.
* Only the HTTP client retried; Triton (and any custom `ServingClient`) got none, and
  the HTTP backoff had no jitter, so a fleet retried in lockstep.
* `http_client` JSON-encoded tensors row by row after warning that doing so is slow.
* `run_pipeline` joined its worker threads with a 1 second timeout and returned, so
  threads outlived the call; abandoning the generator leaked them outright.
* `Stage.num_gpus` was a hint nothing consumed.
* `_resilient_call` dropped a failing row and surfaced **nothing** — silent data loss.

Nothing here needs a network, a GPU, or a running Triton/Serve: every backend is a fake.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types
import urllib.error
import warnings
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from batcher._internal.errors import BackendError, PerformanceWarning
from batcher.ml.serving.base import serving_udf

pytestmark = pytest.mark.unit


# --- helpers -----------------------------------------------------------------


def _batch(rows: int) -> pa.RecordBatch:
    return pa.record_batch({"x": list(range(rows)), "id": list(range(100, 100 + rows))})


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _alive_pipeline_threads() -> list[threading.Thread]:
    return [
        t
        for t in threading.enumerate()
        if t.name.startswith("batcher-ml-pipeline") and t.is_alive()
    ]


# --- 1. serve_deployment must not block the event loop ------------------------


def _install_fake_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a pass-through fake of `ray.serve` so no cluster is needed."""

    def deployment(**_kwargs: Any):
        def wrap(cls: type) -> type:
            return cls

        return wrap

    def batch(**_kwargs: Any):
        def wrap(fn: Any) -> Any:
            return fn

        return wrap

    serve = types.SimpleNamespace(deployment=deployment, batch=batch)
    ray = types.ModuleType("ray")
    ray.serve = serve  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.serve", serve)


def test_serve_deployment_runs_predict_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """The forward pass must run in an executor thread, never on the loop thread."""
    from batcher.ml.serving.online import serve_deployment

    _install_fake_serve(monkeypatch)
    seen: dict[str, int] = {}

    def build():
        def predict(inputs: list[Any]) -> list[Any]:
            seen["predict_thread"] = threading.get_ident()
            return [v * 2 for v in inputs]

        return predict

    deployment = serve_deployment(build)
    instance = deployment()

    async def drive() -> list[Any]:
        seen["loop_thread"] = threading.get_ident()
        return await instance._batched([1, 2, 3])

    out = asyncio.run(drive())

    assert out == [2, 4, 6]
    # Before the fix `_predict` was called synchronously inside the coroutine, so both
    # idents were the loop thread and a slow model stalled every queued request.
    assert seen["predict_thread"] != seen["loop_thread"]


# --- 2. serving_udf: warmup, max_batch_size split, in-flight pipelining -------


def test_serving_udf_warms_up_the_client_once() -> None:
    """A client exposing `warmup` is warmed at connect, before the first real batch."""

    class _Client:
        def __init__(self) -> None:
            self.warmups = 0
            self.predicts = 0

        def warmup(self) -> None:
            assert self.predicts == 0  # warmup happens before any real traffic
            self.warmups += 1

        def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            self.predicts += 1
            return {"y": inputs["x"] * 2}

    client = _Client()
    udf = serving_udf(lambda: client, input_columns=["x"])()

    assert client.warmups == 1
    udf(_batch(4))
    assert client.warmups == 1  # warmed once per worker, not per batch


def test_serving_udf_splits_to_the_server_max_batch_size() -> None:
    """A batch larger than the server's window is split, and the outputs re-join in order."""
    seen: list[int] = []

    class _Client:
        def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            seen.append(len(inputs["x"]))
            return {"y": inputs["x"] * 2}

    udf = serving_udf(lambda: _Client(), input_columns=["x"], max_batch_size=4)()
    out = udf(_batch(10))

    assert seen == [4, 4, 2]  # never exceeds the server's declared window
    assert out.column("y").to_pylist() == [v * 2 for v in range(10)]
    assert out.column("id").to_pylist() == list(range(100, 110))  # sibling stays aligned


def test_serving_udf_pipelines_requests_in_flight() -> None:
    """Two sub-batches must be in flight at once, so the server is never left idle."""
    barrier = threading.Barrier(2, timeout=5.0)

    class _Client:
        def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            barrier.wait()  # only completes if a second request overlaps this one
            return {"y": inputs["x"] * 2}

    udf = serving_udf(lambda: _Client(), input_columns=["x"], max_batch_size=1, pipeline_depth=2)()
    # Serially this deadlocks until the barrier times out and raises BrokenBarrierError.
    out = udf(_batch(2))

    assert out.column("y").to_pylist() == [0, 2]  # and order is still input order


def test_serving_udf_keeps_a_single_request_when_the_batch_fits() -> None:
    """No split and no thread pool when the batch already fits the server's window."""
    seen: list[int] = []

    class _Client:
        def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            seen.append(len(inputs["x"]))
            return {"y": inputs["x"] * 2}

    udf = serving_udf(lambda: _Client(), input_columns=["x"], max_batch_size=64)()
    udf(_batch(8))

    assert seen == [8]


# --- 3. every serving client retries, with jittered backoff -------------------


def test_serving_udf_retries_a_transient_backend_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A flaky backend (Triton included) is retried; only exhaustion raises."""
    monkeypatch.setattr("batcher.ml.serving.base.time.sleep", lambda _s: None)
    calls: list[int] = []

    class _Flaky:
        def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("server restarting")
            return {"y": inputs["x"] * 2}

    udf = serving_udf(lambda: _Flaky(), input_columns=["x"], retries=2)()
    out = udf(_batch(2))

    assert len(calls) == 3
    assert out.column("y").to_pylist() == [0, 2]


def test_serving_udf_retry_exhaustion_raises_backend_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exhausted retries surface as a typed `BackendError`, not the raw transport error."""
    monkeypatch.setattr("batcher.ml.serving.base.time.sleep", lambda _s: None)

    class _Dead:
        def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            raise ConnectionError("refused")

    udf = serving_udf(lambda: _Dead(), input_columns=["x"], retries=1)()
    with pytest.raises(BackendError):
        udf(_batch(2))


def test_serving_udf_does_not_retry_a_programming_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A TypeError/ValueError won't get better by retrying, so it fails fast."""
    monkeypatch.setattr("batcher.ml.serving.base.time.sleep", lambda _s: None)
    calls: list[int] = []

    class _Bug:
        def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            calls.append(1)
            raise TypeError("bad shape")

    udf = serving_udf(lambda: _Bug(), input_columns=["x"], retries=3)()
    with pytest.raises(TypeError):
        udf(_batch(2))
    assert calls == [1]


def test_triton_client_retries_and_warms_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Triton adapter gets both a readiness warmup and bounded retry."""
    monkeypatch.setattr("batcher.ml.serving.base.time.sleep", lambda _s: None)
    events: list[str] = []

    class _InferInput:
        def __init__(self, name: str, shape: list[int], dtype: str) -> None:
            self.name = name

        def set_data_from_numpy(self, arr: np.ndarray) -> None:
            self._arr = arr

    class _Requested:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Result:
        def as_numpy(self, name: str) -> np.ndarray:
            return np.array([1, 2])

    class _Server:
        def __init__(self, url: str) -> None:
            self.attempts = 0

        def is_model_ready(self, model: str, model_version: str = "") -> bool:
            events.append("ready")
            return True

        def infer(self, *a: Any, **k: Any) -> _Result:
            self.attempts += 1
            events.append("infer")
            if self.attempts < 2:
                raise ConnectionError("triton restarting")
            return _Result()

    fake = types.ModuleType("tritonclient.http")
    fake.InferInput = _InferInput  # type: ignore[attr-defined]
    fake.InferRequestedOutput = _Requested  # type: ignore[attr-defined]
    fake.InferenceServerClient = _Server  # type: ignore[attr-defined]
    pkg = types.ModuleType("tritonclient")
    pkg.http = fake  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tritonclient", pkg)
    monkeypatch.setitem(sys.modules, "tritonclient.http", fake)

    from batcher.ml.serving.triton import triton_client

    udf = triton_client(
        "localhost:8000", "m", input_columns=["x"], output_columns=["out"], retries=2
    )()
    out = udf(_batch(2))

    assert events[0] == "ready"  # warmed before the first inference
    assert events.count("infer") == 2  # the transient failure was retried
    assert out.column("out").to_pylist() == [1, 2]


def test_post_json_backoff_is_jittered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backoff must be jittered so a fleet of workers doesn't retry in lockstep."""
    from batcher.ml.serving import http as http_mod

    slept: list[float] = []
    monkeypatch.setattr(http_mod.time, "sleep", slept.append)

    def boom(*_a: Any, **_k: Any):
        raise urllib.error.URLError("down")

    monkeypatch.setattr("urllib.request.urlopen", boom)

    with pytest.raises(BackendError):
        http_mod.post_json(
            "http://x/predict", {"a": [1]}, headers={}, timeout=1.0, retries=3, backoff=0.5
        )

    assert len(slept) == 3
    deterministic = [0.5, 1.0, 2.0]
    # Every delay stays inside its exponential window, but is not the bare value.
    for delay, cap in zip(slept, deterministic, strict=True):
        assert 0.0 <= delay <= cap
    assert slept != deterministic


# --- 4. http_client must not JSON-encode tensors row by row -------------------


def _capture_post(monkeypatch: pytest.MonkeyPatch, response: dict[str, Any]) -> list[bytes]:
    import json

    bodies: list[bytes] = []

    def fake_urlopen(req: Any, timeout: float | None = None) -> _FakeResponse:
        bodies.append(req.data)
        return _FakeResponse(json.dumps(response).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return bodies


def test_http_client_sends_tensors_binary_not_nested_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tensor input goes out base64-encoded, so no O(rows x dims) Python conversion."""
    import json

    from batcher.io.formats.ml.tensor import to_tensor_column

    bodies = _capture_post(monkeypatch, {"y": [1.0, 2.0, 3.0]})
    tensor = to_tensor_column(np.arange(12, dtype=np.float32).reshape(3, 4))
    batch = pa.record_batch({"t": tensor})

    from batcher.ml.serving.http import http_client

    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        udf = http_client("http://host/predict", input_columns=["t"], output_columns=["y"])()
        out = udf(batch)

    payload = json.loads(bodies[0])
    encoded = payload["t"]
    assert isinstance(encoded, dict)  # an envelope, not a list of lists
    assert encoded["shape"] == [3, 4]
    assert encoded["dtype"] == np.dtype(np.float32).str
    assert isinstance(encoded["data"], str)  # base64 bytes, produced in C
    assert out.column("y").to_pylist() == [1.0, 2.0, 3.0]


def test_http_client_round_trips_a_binary_tensor_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same envelope is understood on the way back, so a server can reply in binary."""
    import base64

    values = np.arange(6, dtype=np.float32).reshape(3, 2)
    envelope = {
        "y": {
            "dtype": values.dtype.str,
            "shape": [3, 2],
            "data": base64.b64encode(values.tobytes()).decode(),
        }
    }
    _capture_post(monkeypatch, envelope)

    from batcher.ml.serving.http import http_client

    udf = http_client("http://host/predict", input_columns=["x"], output_columns=["y"])()
    out = udf(_batch(3))

    got = out.column("y").to_numpy(zero_copy_only=False)
    np.testing.assert_array_equal(np.stack(list(got)).reshape(3, 2), values)


def test_http_client_json_encoding_still_available_and_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`tensor_encoding='json'` keeps the legacy wire shape — and still warns it's slow."""
    import json

    from batcher.io.formats.ml.tensor import to_tensor_column

    bodies = _capture_post(monkeypatch, {"y": [1.0, 2.0]})
    batch = pa.record_batch({"t": to_tensor_column(np.arange(4, dtype=np.float32).reshape(2, 2))})

    from batcher.ml.serving.http import http_client

    udf = http_client(
        "http://host/predict",
        input_columns=["t"],
        output_columns=["y"],
        tensor_encoding="json",
    )()
    with pytest.warns(PerformanceWarning):
        udf(batch)

    assert json.loads(bodies[0])["t"] == [[0.0, 1.0], [2.0, 3.0]]


def test_http_client_rejects_an_unknown_tensor_encoding() -> None:
    """An unknown encoding is a typed error at build time, not a wire surprise."""
    from batcher.ml.serving.http import http_client

    with pytest.raises(BackendError):
        http_client(
            "http://host/predict",
            input_columns=["x"],
            output_columns=["y"],
            tensor_encoding="protobuf",
        )


# --- 5 & 6. run_pipeline shutdown determinism, and the dead num_gpus hint -----


def _double() -> Any:
    return lambda b: pa.record_batch({"x": [v * 2 for v in b.column("x").to_pylist()]})


def test_run_pipeline_joins_every_thread_on_normal_completion() -> None:
    """No worker thread may outlive the call — a 1 second join timeout is not shutdown."""
    from batcher.ml.pipeline import Stage, run_pipeline

    before = len(_alive_pipeline_threads())
    batches = [pa.record_batch({"x": [i]}) for i in range(20)]
    out = list(run_pipeline(iter(batches), [Stage(_double), Stage(_double)]))

    assert len(out) == 20
    assert len(_alive_pipeline_threads()) == before


def test_run_pipeline_shuts_down_when_the_consumer_abandons_it() -> None:
    """Closing the generator mid-stream must stop and join the feeder and every stage."""
    from batcher.ml.pipeline import Stage, run_pipeline

    before = len(_alive_pipeline_threads())

    def source():
        for i in range(10_000):
            yield pa.record_batch({"x": [i]})

    gen = run_pipeline(source(), [Stage(_double, credits=1), Stage(_double, credits=1)])
    next(gen)  # start the pipeline, then walk away from it
    gen.close()

    assert len(_alive_pipeline_threads()) == before


def test_run_pipeline_still_propagates_a_stage_error() -> None:
    """Deterministic shutdown must not swallow the first stage exception."""
    from batcher.ml.pipeline import Stage, run_pipeline

    def boom_factory():
        def worker(_b: pa.RecordBatch) -> pa.RecordBatch:
            raise RuntimeError("stage failed")

        return worker

    with pytest.raises(RuntimeError, match="stage failed"):
        list(run_pipeline((pa.record_batch({"x": [i]}) for i in range(10)), [Stage(boom_factory)]))
    assert not _alive_pipeline_threads()


def test_stage_num_gpus_is_accepted_but_still_consumed_by_nothing() -> None:
    """Pins the known gap: `num_gpus` is settable and documented, but affects nothing.

    Deleting it as speculative API is not a one-file change — `docs/ml/model-serving-
    patterns.md` teaches ``Stage(..., num_gpus=1)`` in an executed example, so removing
    the field breaks the docs build. This test exists so the field cannot quietly grow a
    meaning, or quietly disappear, without someone reading this note first.
    """
    from batcher.ml.pipeline import Stage, run_pipeline

    gpu = Stage(_double, num_gpus=1.0, name="forward")
    cpu = Stage(_double, num_gpus=0.0, name="decode")
    assert gpu.num_gpus == 1.0

    # Identical results and identical scheduling: the hint changes nothing single-node.
    batches = [pa.record_batch({"x": [i]}) for i in range(5)]
    with_gpu = [b.column("x").to_pylist() for b in run_pipeline(iter(batches), [gpu])]
    with_cpu = [b.column("x").to_pylist() for b in run_pipeline(iter(batches), [cpu])]
    assert with_gpu == with_cpu


# --- 7. dropped UDF rows must be surfaced -------------------------------------


def test_resilient_call_surfaces_dropped_rows() -> None:
    """A dropped row publishes an event and increments a caller-visible count."""
    from batcher._internal import events
    from batcher.core.udf.call import _resilient_call

    seen: list[events.Event] = []
    unsubscribe = events.subscribe(seen.append)
    try:
        batch = pa.record_batch({"x": list(range(8))})

        def call(sub: pa.RecordBatch) -> pa.RecordBatch:
            values = sub.column("x").to_pylist()
            if 3 in values or 5 in values:
                raise ValueError("corrupt row")
            return sub

        budget = [4]
        out = _resilient_call(call, batch, budget, False)
    finally:
        unsubscribe()

    kept = [v for b in out for v in b.column("x").to_pylist()]
    assert kept == [0, 1, 2, 4, 6, 7]  # the two bad rows were isolated and dropped
    assert budget[0] == 2  # two charged against the allowance
    # Before the fix nothing was recorded anywhere: the rows just vanished.
    assert budget[1] == 2
    drops = [e for e in seen if e.fields.get("dropped_rows")]
    assert drops, "a dropped row must reach the event bus"
    assert drops[-1].fields["dropped_rows"] == 2
    assert "corrupt row" in drops[-1].fields["error"]


def test_resilient_call_drop_count_survives_concurrent_batches() -> None:
    """`execute` runs `_resilient_call` on a thread pool, so the count must not race."""
    from concurrent.futures import ThreadPoolExecutor

    from batcher.core.udf.call import _resilient_call

    def call(sub: pa.RecordBatch) -> pa.RecordBatch:
        if any(v % 2 for v in sub.column("x").to_pylist()):
            raise ValueError("odd rows are corrupt")
        return sub

    budget = [1000]
    batches = [pa.record_batch({"x": list(range(i * 8, i * 8 + 8))}) for i in range(16)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda b: _resilient_call(call, b, budget, False), batches))

    dropped = 16 * 4  # every odd row across every batch
    assert budget[1] == dropped
    assert budget[0] == 1000 - dropped


def test_resilient_call_reports_nothing_when_no_row_is_dropped() -> None:
    """The drop counter stays absent on the clean path — zero cost, no noise."""
    from batcher._internal import events
    from batcher.core.udf.call import _resilient_call

    seen: list[events.Event] = []
    unsubscribe = events.subscribe(seen.append)
    try:
        batch = pa.record_batch({"x": list(range(8))})
        budget = [4]
        out = _resilient_call(lambda sub: sub, batch, budget, False)
    finally:
        unsubscribe()

    assert [v for b in out for v in b.column("x").to_pylist()] == list(range(8))
    assert budget == [4]
    assert not [e for e in seen if e.fields.get("dropped_rows")]


def test_split_feed_passes_broadcast_inputs_through_whole():
    """A shared (1, ...) input must not be sliced to empty on sub-batches after the first."""
    from batcher.ml.serving.base import _split_feed

    per_row = np.arange(4).reshape(4, 1)
    shared = np.array([[7.0, 8.0]])  # one row, broadcast to the whole batch
    parts = _split_feed({"x": per_row, "cfg": shared}, num_rows=4, max_batch_size=2)
    assert len(parts) == 2
    for part in parts:
        assert part["cfg"].shape == (1, 2)  # never emptied
    assert parts[0]["x"].tolist() == [[0], [1]]
    assert parts[1]["x"].tolist() == [[2], [3]]


def test_merge_results_handles_scalar_per_subbatch_outputs():
    """A 0-d per-sub-batch output concatenates instead of raising."""
    from batcher.ml.serving.base import _merge_results

    parts = [{"summary": np.array(1.5)}, {"summary": np.array(2.5)}]
    merged = _merge_results(parts)
    assert merged["summary"].tolist() == [1.5, 2.5]


def test_tensor_envelope_detection_ignores_a_lookalike_dict():
    """A model output dict with dtype/shape/data keys but non-envelope value types is not
    decoded as a tensor."""
    from batcher.ml.serving.http import _decode_value, _is_tensor_envelope

    lookalike = {"dtype": {"kind": "f"}, "shape": "big", "data": [1, 2, 3]}
    assert not _is_tensor_envelope(lookalike)
    # A real marker-carrying envelope is still detected.
    import base64

    real = {
        "__tensor__": True,
        "dtype": "<f4",
        "shape": [2],
        "data": base64.b64encode(np.array([1.0, 2.0], dtype="<f4").tobytes()).decode(),
    }
    assert _is_tensor_envelope(real)
    assert _decode_value(real).tolist() == [1.0, 2.0]


def test_post_json_wraps_a_non_json_body():
    """A 200 with an HTML/proxy body raises BackendError, not an opaque JSONDecodeError."""
    import batcher.ml.serving.http as http_mod
    from batcher._internal.errors import BackendError

    class _Resp:
        def read(self):
            return b"<html>502 Bad Gateway</html>"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    real = urllib.request.urlopen
    urllib.request.urlopen = lambda req, timeout=None: _Resp()
    try:
        with __import__("pytest").raises(BackendError, match="non-JSON"):
            http_mod.post_json("http://x", {"a": 1}, headers={}, timeout=1, retries=0)
    finally:
        urllib.request.urlopen = real


def test_triton_client_passes_a_request_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """`timeout=` reaches the Triton `infer` call as `client_timeout`, so a wedged replica
    aborts and is retried rather than blocking the worker forever."""
    seen: dict[str, Any] = {}

    class _InferInput:
        def __init__(self, name: str, shape: list[int], dtype: str) -> None:
            self.name = name

        def set_data_from_numpy(self, arr: np.ndarray) -> None:
            pass

    class _Requested:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Result:
        def as_numpy(self, name: str) -> np.ndarray:
            return np.array([1, 2])

    class _Server:
        def __init__(self, url: str) -> None:
            pass

        def is_model_ready(self, model: str, model_version: str = "") -> bool:
            return True

        def infer(self, *a: Any, **k: Any) -> _Result:
            seen["client_timeout"] = k.get("client_timeout")
            return _Result()

    fake = types.ModuleType("tritonclient.http")
    fake.InferInput = _InferInput  # type: ignore[attr-defined]
    fake.InferRequestedOutput = _Requested  # type: ignore[attr-defined]
    fake.InferenceServerClient = _Server  # type: ignore[attr-defined]
    pkg = types.ModuleType("tritonclient")
    pkg.http = fake  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tritonclient", pkg)
    monkeypatch.setitem(sys.modules, "tritonclient.http", fake)

    from batcher.ml.serving.triton import triton_client

    udf = triton_client(
        "localhost:8000", "m", input_columns=["x"], output_columns=["out"], timeout=5.0
    )()
    udf(_batch(2))
    assert seen["client_timeout"] == 5.0


def test_serving_rejects_a_non_numeric_input_column_by_name():
    """A string input column fails with an actionable, column-named error instead of an
    opaque triton dtype error or base64 garbage."""
    from batcher._internal.errors import BackendError
    from batcher.ml.serving.base import serving_udf

    class _Client:
        def predict(self, inputs):
            return {"out": np.array([1])}

    udf = serving_udf(lambda: _Client(), input_columns=["text"], output_columns=["out"])()
    batch = pa.record_batch({"text": ["a", "b"]})
    with pytest.raises(BackendError, match="text"):
        udf(batch)


def test_serving_accepts_a_numeric_input_column():
    from batcher.ml.serving.base import serving_udf

    class _Client:
        def predict(self, inputs):
            return {"out": inputs["x"] * 2}

    udf = serving_udf(lambda: _Client(), input_columns=["x"], output_columns=["out"])()
    out = udf(pa.record_batch({"x": [1, 2, 3]}))
    assert out.column("out").to_pylist() == [2, 4, 6]
