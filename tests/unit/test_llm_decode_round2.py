"""Round-2 regression tests for the LLM generation path and the decode package.

Every test here pins a behavior that was wrong (or a contract that was only documented,
never enforced) before the change that accompanies it. Nothing needs a GPU, a network,
or a real model: an `Engine` is just ``list -> list[str]``, so a list-comprehension is a
complete engine, and the decode tests drive the pure helpers with fakes.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

pytestmark = pytest.mark.unit

pa = pytest.importorskip("pyarrow")


# --------------------------------------------------------------------------------------
# The decode package split (item A) — the public import path must survive package-izing.
# --------------------------------------------------------------------------------------


def test_decode_public_import_path_survives_the_package_split() -> None:
    """`batcher.ml.decode` must keep exporting the same five helpers from the same path.

    Package-izing a module silently moves every name it held. The public path is the
    contract; the submodules behind it are not.
    """
    decode = importlib.import_module("batcher.ml.decode")
    assert sorted(decode.__all__) == [
        "audio_dataset",
        "download_dataset",
        "image_tensor_dataset",
        "upload_dataset",
        "video_dataset",
    ]
    for name in decode.__all__:
        assert callable(getattr(decode, name)), name


def test_decode_private_test_seams_stay_reachable_on_the_package() -> None:
    """The private helpers the decode suite drives directly must stay on the package.

    A test that patches or calls ``decode._shared_pool`` becomes a no-op (or an
    AttributeError) the moment that name lives only in a submodule — the failure mode
    the concurrent-agents rule calls out for moved files.
    """
    decode = importlib.import_module("batcher.ml.decode")
    for name in ("_bounded_map", "_decode_video_bytes", "_shared_pool"):
        assert hasattr(decode, name), name


# --------------------------------------------------------------------------------------
# llm_generate / llm_udf unification (item B1).
# --------------------------------------------------------------------------------------


class _RecordingEngine:
    """An engine that records how many requests each call received."""

    def __init__(self, log: list[list[Any]]) -> None:
        self._log = log

    def __call__(self, requests: list) -> list[str]:
        self._log.append(list(requests))
        return [f"out-{i}" for i in range(len(requests))]


def _counting_factory() -> tuple[Any, list[int], list[list[Any]]]:
    """A factory plus the two logs the tests assert on: builds, and per-call requests."""
    builds: list[int] = []
    calls: list[list[Any]] = []

    def factory() -> Any:
        builds.append(1)
        return _RecordingEngine(calls)

    return factory, builds, calls


def _batches(count: int, rows: int) -> list[Any]:
    return [
        pa.RecordBatch.from_pydict({"q": [f"b{b}r{r}" for r in range(rows)]}) for b in range(count)
    ]


def test_llm_generate_yields_one_output_batch_per_input_batch() -> None:
    """`llm_generate` documents "each input batch with output_column appended, in order".

    It did not do that: an `InferencePool` re-chunked the stream to a target row count, so
    four 10-row batches came back as a single 40-row batch. That silently changes the
    caller's batch boundaries — and therefore the memory shape and the backpressure — of
    a function whose whole purpose is to stream.
    """
    from batcher.ml.llm import llm_generate

    factory, _builds, _calls = _counting_factory()
    out = list(llm_generate(iter(_batches(4, 10)), factory, prompt_column="q"))

    assert [b.num_rows for b in out] == [10, 10, 10, 10]


def test_llm_generate_builds_one_engine_by_default() -> None:
    """The default must load the model **once**, not once per pooled worker.

    The factory contract is "called once per worker so the model loads once", but the
    default pool had two workers, so a GPU-resident `vllm_engine` loaded two full copies
    of the weights into one process. For any model that fills its GPU — the normal case
    in offline batch — that is an OOM rather than a speedup.
    """
    from batcher.ml.llm import llm_generate

    factory, builds, _calls = _counting_factory()
    list(llm_generate(iter(_batches(3, 4)), factory, prompt_column="q"))

    assert len(builds) == 1


def test_llm_generate_matches_llm_udf_row_for_row() -> None:
    """The two entry points are one capability and must produce identical columns.

    This is the property the unification exists to guarantee: whatever scheduling wraps
    it, the columnar result of `llm_generate` is exactly what `llm_udf` produces for the
    same batches and the same engine.
    """
    from batcher.ml.llm import llm_generate, llm_udf

    batches = _batches(3, 5)

    factory_a, _b, _c = _counting_factory()
    streamed = list(llm_generate(iter(batches), factory_a, prompt_column="q"))

    factory_b, _b2, _c2 = _counting_factory()
    udf = llm_udf(factory_b, prompt_column="q")()
    mapped = [udf(b) for b in batches]

    assert [b.to_pydict() for b in streamed] == [b.to_pydict() for b in mapped]


def test_llm_generate_can_still_opt_into_pooled_rebatching() -> None:
    """Coalescing batches is a real throughput lever for a self-batching engine, so it
    stays available — as an explicit opt-in rather than the silent default."""
    from batcher.ml.llm import llm_generate

    factory, _builds, calls = _counting_factory()
    out = list(
        llm_generate(iter(_batches(4, 10)), factory, prompt_column="q", target_batch_rows=40)
    )

    assert sum(b.num_rows for b in out) == 40
    assert max(len(c) for c in calls) > 10


# --------------------------------------------------------------------------------------
# Per-row sampling parameters (item B2).
# --------------------------------------------------------------------------------------


def test_per_row_max_tokens_reaches_the_engine_on_each_request() -> None:
    """A per-row token budget must travel with its row, like `adapter_column` already does.

    Without it a batch is capped by one global `max_tokens`, so a mixed workload has to
    either overpay for every short row or truncate every long one.
    """
    from batcher.ml.llm import llm_udf

    calls: list[list[Any]] = []
    udf = llm_udf(
        lambda: _RecordingEngine(calls),
        prompt_column="q",
        max_tokens_column="budget",
    )()
    batch = pa.RecordBatch.from_pydict({"q": ["a", "b"], "budget": [16, 512]})
    udf(batch)

    sent = {r["prompt"]: r["max_tokens"] for r in calls[0]}
    assert sent == {"a": 16, "b": 512}


def test_per_row_temperature_reaches_the_engine_and_nulls_fall_back() -> None:
    """A null in the per-row column means "use the engine's own default" — the same
    null convention `adapter_column` uses for the base model."""
    from batcher.ml.llm import llm_udf

    calls: list[list[Any]] = []
    udf = llm_udf(
        lambda: _RecordingEngine(calls),
        prompt_column="q",
        temperature_column="temp",
    )()
    batch = pa.RecordBatch.from_pydict({"q": ["a", "b"], "temp": [0.9, None]})
    udf(batch)

    by_prompt = {r["prompt"] if isinstance(r, dict) else r: r for r in calls[0]}
    assert by_prompt["a"]["temperature"] == pytest.approx(0.9)
    assert "temperature" not in by_prompt["b"]


def test_vllm_builds_per_request_sampling_params_only_when_a_row_overrides() -> None:
    """The vLLM adapter must turn those per-row keys into per-request `SamplingParams`.

    A batch where no row overrides anything keeps the single shared params object, so the
    common case pays nothing for the feature.
    """
    from batcher.ml.llm.engines.vllm import _per_request_params

    base = _FakeParams(temperature=0.0, max_tokens=128)

    assert _per_request_params(base, ["a", "b"]) is base

    out = _per_request_params(base, [{"prompt": "a", "max_tokens": 8}, {"prompt": "b"}])
    assert isinstance(out, list)
    assert out[0].max_tokens == 8
    assert out[1].max_tokens == 128
    assert out[0].temperature == pytest.approx(0.0)


class _FakeParams:
    """A stand-in for `vllm.SamplingParams` — a plain attribute bag that clones."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    def clone(self) -> _FakeParams:
        return _FakeParams(**self.__dict__)


# --------------------------------------------------------------------------------------
# Logprobs (item B5).
# --------------------------------------------------------------------------------------


def test_logprob_column_is_appended_when_the_engine_reports_one() -> None:
    """`logprobs=True` surfaces the generation's cumulative logprob as a float column.

    It is the model's own confidence signal. Without it there is no way to route
    low-confidence rows to review or to a larger model, which is the standard
    quality-control loop over a bulk generation.
    """
    from batcher.ml.llm import llm_udf
    from batcher.ml.llm.channels import logprob_sink

    def engine(requests: list) -> list[str]:
        logprob_sink().report([-0.5, -12.0])
        return ["x", "y"]

    udf = llm_udf(lambda: engine, prompt_column="q", logprobs=True)()
    out = udf(pa.RecordBatch.from_pydict({"q": ["a", "b"]}))

    assert out.column("logprob").to_pylist() == [-0.5, -12.0]


def test_logprob_column_is_null_for_an_engine_that_reports_none() -> None:
    """An engine that reports no logprob yields nulls, never a raise — the same
    convention `usage` and `finish_reason` already follow."""
    from batcher.ml.llm import llm_udf

    udf = llm_udf(lambda: lambda reqs: ["x"] * len(reqs), prompt_column="q", logprobs=True)()
    out = udf(pa.RecordBatch.from_pydict({"q": ["a", "b"]}))

    assert out.column("logprob").to_pylist() == [None, None]


def test_logprobs_survive_the_length_sorted_dispatch() -> None:
    """Logprobs come back in *dispatch* order and must be un-permuted onto their rows.

    Requests are dispatched longest-prompt-first, so a reported value lands on the wrong
    row unless it is inverted with the same permutation the outputs are. That failure is
    silent: every row still has a plausible number.
    """
    from batcher.ml.llm import llm_udf
    from batcher.ml.llm.channels import logprob_sink

    def engine(requests: list) -> list[str]:
        # Report each request's prompt length as its "logprob", so a misalignment is
        # visible rather than plausible.
        logprob_sink().report([float(len(str(r))) for r in requests])
        return [str(r) for r in requests]

    udf = llm_udf(lambda: engine, prompt_column="q", logprobs=True)()
    out = udf(pa.RecordBatch.from_pydict({"q": ["a", "bbbb", "cc"]}))

    assert out.column("logprob").to_pylist() == [1.0, 4.0, 2.0]
