"""A serving model's batch window comes from the server, which is where it is declared.

An engine batch is orders of magnitude larger than the window a serving configuration
typically names -- `max_batch_size: 8` against sixteen thousand rows -- and Triton rejects an
oversized request whole rather than splitting it. So a connector that only splits when the
caller restates a number the server already knows fails on the first real batch of every real
model, after the read stage has already run.

The window is therefore asked for, and the asking is optional in both directions: a client
that cannot ask does not define the method, and a client whose ask fails keeps the old
send-it-whole behaviour rather than failing a worker that has not run a row.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from batcher.ml.serving.base import serving_udf

pytestmark = pytest.mark.unit


class _Recorder:
    """A serving client that records the row count of every request it is sent."""

    def __init__(self, window: object = None, raises: bool = False) -> None:
        self.window = window
        self.raises = raises
        self.rows: list[int] = []

    def batch_window(self):
        if self.raises:
            raise RuntimeError("config endpoint unreachable")
        return self.window

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        rows = len(next(iter(inputs.values())))
        self.rows.append(rows)
        return {"out": np.zeros(rows, dtype=np.float32)}


def _run(client: _Recorder, rows: int = 10, **kwargs) -> pa.RecordBatch:
    udf = serving_udf(lambda: client, input_columns=["x"], output_columns=["out"], **kwargs)()
    return udf(pa.record_batch({"x": pa.array(np.arange(rows, dtype=np.float32))}))


def test_the_declared_window_splits_the_batch() -> None:
    client = _Recorder(window=4)
    out = _run(client, rows=10)
    assert client.rows == [4, 4, 2]
    assert out.num_rows == 10


def test_an_explicit_window_still_wins() -> None:
    # The caller has seen the deployment; a declared number must not override a stated one.
    client = _Recorder(window=4)
    _run(client, rows=10, max_batch_size=5)
    assert client.rows == [5, 5]


def test_a_model_that_declares_no_window_is_sent_whole() -> None:
    client = _Recorder(window=None)
    _run(client, rows=10)
    assert client.rows == [10]


def test_a_zero_window_is_not_a_window() -> None:
    # Triton's `max_batch_size: 0` means the model does not batch in Triton's sense; reading
    # it as a split size would ask for empty requests.
    client = _Recorder(window=0)
    _run(client, rows=10)
    assert client.rows == [10]


def test_a_non_integer_window_is_ignored() -> None:
    client = _Recorder(window="8")
    _run(client, rows=10)
    assert client.rows == [10]


def test_an_unreachable_config_does_not_fail_the_worker() -> None:
    # The window is an optimization over what the caller could have passed by hand. Failing
    # here would take down a worker before it has run a row, for a number it can do without.
    client = _Recorder(raises=True)
    _run(client, rows=10)
    assert client.rows == [10]


def test_a_client_that_cannot_be_asked_behaves_exactly_as_before() -> None:
    class _Plain:
        def __init__(self) -> None:
            self.rows: list[int] = []

        def predict(self, inputs):
            rows = len(next(iter(inputs.values())))
            self.rows.append(rows)
            return {"out": np.zeros(rows, dtype=np.float32)}

    client = _Plain()
    udf = serving_udf(lambda: client, input_columns=["x"], output_columns=["out"])()
    udf(pa.record_batch({"x": pa.array(np.arange(10, dtype=np.float32))}))
    assert client.rows == [10]


def test_the_window_is_read_once_per_worker_not_once_per_batch() -> None:
    class _Counting(_Recorder):
        asks = 0

        def batch_window(self):
            _Counting.asks += 1
            return super().batch_window()

    client = _Counting(window=4)
    udf = serving_udf(lambda: client, input_columns=["x"], output_columns=["out"])()
    for _ in range(3):
        udf(pa.record_batch({"x": pa.array(np.arange(10, dtype=np.float32))}))
    assert _Counting.asks == 1


def test_the_http_client_can_split_and_pipeline_requests(monkeypatch) -> None:
    """An engine batch is thousands of rows, and an endpoint sized for a serving batch answers
    one of those with a 413, a timeout, or an out-of-memory error on its own GPU. Without a
    window to split on there was no way to send it anything smaller."""
    from batcher.ml.serving.http import http_client

    sent: list[int] = []

    class _Client:
        def predict(self, inputs):
            rows = len(next(iter(inputs.values())))
            sent.append(rows)
            return {"out": np.zeros(rows, dtype=np.float32)}

    monkeypatch.setattr("batcher.ml.serving.http._HttpServingClient", lambda *a, **k: _Client())
    udf = http_client(
        "http://host/predict",
        input_columns=["x"],
        output_columns=["out"],
        max_batch_size=4,
        pipeline_depth=2,
    )()
    out = udf(pa.record_batch({"x": pa.array(np.arange(10, dtype=np.float32))}))
    assert sorted(sent) == [2, 4, 4]
    assert out.num_rows == 10


def test_the_http_client_still_posts_a_batch_whole_by_default(monkeypatch) -> None:
    from batcher.ml.serving.http import http_client

    sent: list[int] = []

    class _Client:
        def predict(self, inputs):
            rows = len(next(iter(inputs.values())))
            sent.append(rows)
            return {"out": np.zeros(rows, dtype=np.float32)}

    monkeypatch.setattr("batcher.ml.serving.http._HttpServingClient", lambda *a, **k: _Client())
    udf = http_client("http://host/predict", input_columns=["x"], output_columns=["out"])()
    udf(pa.record_batch({"x": pa.array(np.arange(10, dtype=np.float32))}))
    assert sent == [10]
