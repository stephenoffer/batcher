"""The model-inference entry points expose the same fault tolerance as `map_batches`.

`ds.ml.map`, `.filter` and `.flat_map` have always taken `max_errored_rows`, and
`map_batches` the full retry/timeout surface. The six methods that actually call a model or
a remote endpoint — where a transient failure is not an edge case but the normal operating
condition of a hosted API — took none of it. A single 503 from a provider, or one row a
content filter refuses, failed a batch inference job over millions of rows.

Every test here drives a stub that fails once (or always, for one row) and asserts the job
still produces the right answer, so nothing needs a GPU, a network, or a model.
"""

from __future__ import annotations

import inspect

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.dataset.ml import DatasetML

pytestmark = pytest.mark.unit

#: The options `map_batches` has always taken, which every model entry point must too.
_RESILIENCE = {"max_errored_rows", "timeout", "max_retries", "retry_backoff", "retry_on"}


@pytest.mark.parametrize(
    "method", ["map_batches", "infer", "predict", "generate", "extract", "classify", "embed"]
)
def test_every_model_entry_point_takes_the_resilience_options(method: str) -> None:
    params = set(inspect.signature(getattr(DatasetML, method)).parameters)
    assert params >= _RESILIENCE, f"{method} is missing {sorted(_RESILIENCE - params)}"


def _fails_once(reply):
    """An engine factory whose first call raises, as a rate-limited endpoint does."""
    state = {"n": 0}

    def factory():
        def engine(prompts):
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("transient 503")
            return [reply] * len(prompts)

        return engine

    return factory


def test_generate_retries_a_transient_engine_failure() -> None:
    ds = bt.from_pydict({"q": ["a", "b"]})
    got = ds.ml.generate(_fails_once("hi"), prompt_column="q", max_retries=2)
    assert got.to_pydict()["response"] == ["hi", "hi"]


def test_extract_retries_a_transient_engine_failure() -> None:
    ds = bt.from_pydict({"q": ["a", "b"]})
    got = ds.ml.extract(
        _fails_once('{"v": 1}'), schema={"v": "int64"}, prompt_column="q", max_retries=2
    )
    assert got.to_pydict()["v"] == [1, 1]


def test_classify_retries_a_transient_engine_failure() -> None:
    ds = bt.from_pydict({"q": ["a", "b"]})
    got = ds.ml.classify(_fails_once("yes"), labels=["yes", "no"], prompt_column="q", max_retries=2)
    assert got.to_pydict()["label"] == ["yes", "yes"]


def test_embed_retries_a_transient_encoder_failure() -> None:
    state = {"n": 0}

    class Encoder:
        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("transient")
            vectors = pa.array([[1.0, 2.0]] * batch.num_rows)
            return batch.append_column("embedding", vectors)

    ds = bt.from_pydict({"q": ["a", "b"]})
    got = ds.ml.embed(Encoder, output_columns=["q", "embedding"], max_retries=2)
    assert got.to_pydict()["embedding"] == [[1.0, 2.0], [1.0, 2.0]]


def test_infer_retries_a_transient_model_failure() -> None:
    state = {"n": 0}

    def model(batch: pa.RecordBatch) -> pa.RecordBatch:
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("transient")
        return batch.append_column("prediction", pa.array([1] * batch.num_rows))

    ds = bt.from_pydict({"q": ["a", "b"]})
    got = ds.ml.infer(model, output_columns=["q", "prediction"], max_retries=2)
    assert got.to_pydict()["prediction"] == [1, 1]


def test_predict_retries_a_transient_model_failure() -> None:
    state = {"n": 0}

    class Model:
        def predict(self, features: np.ndarray) -> list[float]:
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("transient")
            return [float(row[0]) for row in features]

    ds = bt.from_pydict({"x": [1.0, 2.0]})
    got = ds.ml.predict(Model(), features=["x"], max_retries=2)
    assert got.to_pydict()["prediction"] == [1.0, 2.0]


def test_a_row_the_model_always_refuses_is_dropped_against_the_budget() -> None:
    """The other half: a poison row that no retry can fix must not fail the job.

    A content filter, a malformed record, one image that will not decode — at a million
    rows these are certainties, and the only recourse was to fail the whole run.
    """

    def factory():
        def engine(prompts):
            if any(p == "bad" for p in prompts):
                raise RuntimeError("content filter")
            return [p.upper() for p in prompts]

        return engine

    ds = bt.from_pydict({"q": ["ok", "bad", "fine"]})
    got = ds.ml.generate(factory, prompt_column="q", max_errored_rows=2).to_pydict()
    assert got["q"] == ["ok", "fine"]
    assert got["response"] == ["OK", "FINE"]


def test_a_spent_budget_still_fails_the_job() -> None:
    """`max_errored_rows` is a budget, not a mute button."""

    def factory():
        def engine(prompts):
            raise RuntimeError("every row is refused")

        return engine

    ds = bt.from_pydict({"q": ["a", "b", "c"]})
    with pytest.raises(RuntimeError, match="every row is refused"):
        ds.ml.generate(factory, prompt_column="q", max_errored_rows=1).to_pydict()
