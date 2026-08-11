"""Hardening regressions for the LLM batch-inference surface.

Every test here failed before the fix it guards. They cover the defects that make an
offline-inference run *silently* wrong or slow rather than raise: a lossy numeric
coercion, a label that nests inside another label, an instruction dropped for vision /
per-row-LoRA requests, best-of-N thrown away, unbounded images, lockstep retries, and
padding waste from unsorted prompts.

Everything runs against stub engines and stub vLLM objects, so no GPU, network, or
model is needed.
"""

from __future__ import annotations

from typing import ClassVar

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit


def _const_engine(response: str):
    def factory():
        return lambda prompts: [response] * len(prompts)

    return factory


def _echo_engine():
    """Echoes each request back, so a test can assert on what the engine actually saw."""
    seen: list[list] = []

    def factory():
        def engine(prompts):
            seen.append(list(prompts))
            return ["" for _ in prompts]

        return engine

    factory.seen = seen
    return factory


# --- 1. lossy integer narrowing -------------------------------------------------------


def test_a_fractional_value_for_an_int_field_is_null_not_truncated():
    """`int(float("3.9")) == 3` is indistinguishable from a model that genuinely said 3.

    Silent truncation is the worst outcome: the row looks fine and the number is wrong.
    A value that cannot be represented exactly must degrade to null like any other
    uncoercible value.
    """
    ds = bt.from_pydict({"q": ["x"]})
    out = ds.ml.extract(_const_engine('{"n": 3.9}'), schema={"n": "int64"}, prompt_column="q")
    assert out.to_pydict()["n"] == [None]


def test_an_integral_value_still_coerces_for_an_int_field():
    """The guard must not reject the values that were always fine."""
    ds = bt.from_pydict({"q": ["a", "b", "c"]})
    engine = _keyed_engine({"a": '{"n": 7}', "b": '{"n": 7.0}', "c": '{"n": "7"}'})
    out = ds.ml.extract(engine, schema={"n": "int64"}, prompt_column="q")
    assert out.to_pydict()["n"] == [7, 7, 7]


def test_a_large_integer_keeps_full_precision():
    """Routing an int through `float` silently rounds past 2**53."""
    big = 9007199254740993  # 2**53 + 1, not representable as a float64
    ds = bt.from_pydict({"q": ["x"]})
    out = ds.ml.extract(
        _const_engine('{"n": %s}' % big),  # noqa: UP031 - a literal JSON body
        schema={"n": "int64"},
        prompt_column="q",
    )
    assert out.to_pydict()["n"] == [big]


def _keyed_engine(table: dict[str, str], *, default: str = "{}"):
    def factory():
        return lambda prompts: [table.get(p.split("\n")[0], default) for p in prompts]

    return factory


# --- 2. nested labels -----------------------------------------------------------------


def test_a_label_nested_in_another_label_resolves_to_the_longest_match():
    """With labels "positive" and "very positive", a correct answer wrapped in a sentence
    matched BOTH keys, the hit set had two members, and the row was nulled — the model
    was right and the column said nothing."""
    ds = bt.from_pydict({"q": ["x"]})
    engine = _const_engine("The sentiment is very positive.")
    out = ds.ml.classify(engine, labels=["positive", "very positive"], prompt_column="q")
    assert out.to_pydict()["label"] == ["very positive"]


def test_the_shorter_nested_label_still_resolves_on_its_own():
    ds = bt.from_pydict({"q": ["x"]})
    engine = _const_engine("The sentiment is positive.")
    out = ds.ml.classify(engine, labels=["positive", "very positive"], prompt_column="q")
    assert out.to_pydict()["label"] == ["positive"]


def test_a_genuinely_ambiguous_answer_is_still_null():
    """Longest-match must not become a licence to guess between unrelated labels."""
    ds = bt.from_pydict({"q": ["x"]})
    engine = _const_engine("could be positive, could be negative")
    out = ds.ml.classify(engine, labels=["positive", "negative"], prompt_column="q")
    assert out.to_pydict()["label"] == [None]


# --- 3. the instruct suffix on dict requests ------------------------------------------


def test_the_extract_instruction_reaches_a_per_row_lora_request():
    """A dict request (vision or per-row adapter) silently lost the JSON instruction, so
    the model was never told to emit JSON and every row parsed to null."""
    from batcher.ml.llm.structured import _extract_batch, _resolve_schema

    engine = _echo_engine()
    batch = pa.RecordBatch.from_pydict({"q": ["hello"], "ad": ["lora-a"]})
    _extract_batch(
        engine(),
        batch,
        fields=_resolve_schema({"label": "string"}),
        prompt_column="q",
        template=None,
        instruct=True,
        adapter_column="ad",
    )
    request = engine.seen[0][0]
    assert isinstance(request, dict), "an adapter column must still produce a dict request"
    assert request["adapter"] == "lora-a"
    assert "exactly these keys" in request["prompt"]


def test_the_classify_instruction_reaches_a_per_row_lora_request():
    from batcher.ml.llm.structured import _classify_batch

    engine = _echo_engine()
    batch = pa.RecordBatch.from_pydict({"q": ["hello"], "ad": ["lora-a"]})
    _classify_batch(
        engine(),
        batch,
        labels=["yes", "no"],
        lookup={"yes": "yes", "no": "no"},
        prompt_column="q",
        output_column="label",
        template=None,
        instruct=True,
        adapter_column="ad",
    )
    request = engine.seen[0][0]
    assert isinstance(request, dict)
    assert "exactly one of these labels" in request["prompt"]


# --- 4. usage is a per-call channel, not a shared attribute ---------------------------


def test_usage_is_read_from_the_per_call_channel_not_a_stale_attribute():
    """A shared `last_usage` attribute is only correct because the pool happens to hand
    one engine to one thread at a time. The per-call channel removes the coupling: an
    engine that reports through it wins over a stale attribute."""
    from batcher.ml.llm.channels import usage_sink
    from batcher.ml.llm.generate import _generate_batch

    class _Engine:
        last_usage: ClassVar = [(999, 999)]  # stale, from some earlier call

        def __call__(self, prompts):
            usage_sink().report([(1, 2)] * len(prompts))
            return [""] * len(prompts)

    out = _generate_batch(
        _Engine(),
        pa.RecordBatch.from_pydict({"q": ["x"]}),
        prompt_column="q",
        output_column="response",
        template=None,
        image_column=None,
        adapter_column=None,
        parse_json=False,
        usage=True,
    )
    got = out.to_pydict()
    assert got["prompt_tokens"] == [1], "the stale attribute won over the per-call channel"
    assert got["completion_tokens"] == [2]


def test_a_legacy_engine_setting_last_usage_still_works():
    """The documented `last_usage` contract must keep working for user engines."""
    from batcher.ml.llm.generate import _usage_columns

    class _Engine:
        last_usage: ClassVar = [(5, 6)]

    prompt, completion = _usage_columns(_Engine(), 1)
    assert prompt.to_pylist() == [5]
    assert completion.to_pylist() == [6]


# --- 5. best-of-N is honored or refused, never silently discarded --------------------


def test_sampling_n_greater_than_one_is_rejected_rather_than_silently_dropped():
    """`sampling={"n": 4}` produced one string per row with nothing to signal that three
    candidates were generated on the GPU and thrown away."""
    from batcher.ml.llm.engines import vllm_engine

    with pytest.raises(PlanError, match="n=4"):
        vllm_engine("m", sampling={"n": 4})


def test_sampling_n_of_one_is_accepted():
    from batcher.ml.llm.engines import vllm_engine

    assert callable(vllm_engine("m", sampling={"n": 1}))


# --- 6. length bucketing --------------------------------------------------------------


def test_prompts_reach_the_engine_sorted_by_length():
    """Padding waste and prefix-cache misses both fall when similar lengths sit together.

    The engine must see the batch length-ordered even though the caller's row order is
    arbitrary.
    """
    from batcher.ml.llm.generate import _generate_batch

    engine = _echo_engine()
    batch = pa.RecordBatch.from_pydict({"q": ["mmmm", "z", "wwwwwwww", "yy"]})
    _generate_batch(
        engine(),
        batch,
        prompt_column="q",
        output_column="response",
        template=None,
        image_column=None,
        adapter_column=None,
        parse_json=False,
        usage=False,
    )
    lengths = [len(p) for p in engine.seen[0]]
    assert lengths == sorted(lengths, reverse=True), f"engine saw unsorted prompts: {lengths}"


def test_length_sorting_does_not_change_the_output_order():
    """The whole point: bucketing is invisible above the engine."""

    def factory():
        return lambda prompts: [p.upper() for p in prompts]

    ds = bt.from_pydict({"q": ["mmmm", "z", "wwwwwwww", "yy"]})
    out = ds.ml.generate(factory, prompt_column="q")
    assert out.to_pydict()["response"] == ["MMMM", "Z", "WWWWWWWW", "YY"]


def test_length_sorting_keeps_usage_aligned_to_its_row():
    """A permuted dispatch must un-permute the token counts too, or every count lands on
    the wrong row — a silent, plausible-looking corruption."""
    from batcher.ml.llm.channels import usage_sink
    from batcher.ml.llm.generate import _generate_batch

    def engine(prompts):
        usage_sink().report([(len(p), 1) for p in prompts])
        return list(prompts)

    batch = pa.RecordBatch.from_pydict({"q": ["mmmm", "z", "wwwwwwww", "yy"]})
    out = _generate_batch(
        engine,
        batch,
        prompt_column="q",
        output_column="response",
        template=None,
        image_column=None,
        adapter_column=None,
        parse_json=False,
        usage=True,
    )
    got = out.to_pydict()
    assert got["response"] == ["mmmm", "z", "wwwwwwww", "yy"]
    assert got["prompt_tokens"] == [4, 1, 8, 2]


# --- 8. multi-LoRA co-batching --------------------------------------------------------


class _StubOutput:
    def __init__(self, text: str) -> None:
        self.outputs = [type("O", (), {"text": text, "token_ids": [0], "finish_reason": "stop"})()]
        self.prompt_token_ids = [0, 1]


class _CoBatchingLLM:
    """A vLLM stub accepting a per-prompt list of LoRA requests in one scheduler step."""

    def __init__(self) -> None:
        self.calls: list[list] = []

    def generate(self, requests, params, lora_request=None):
        self.calls.append(requests)
        return [_StubOutput(str(r)) for r in requests]


def test_adapters_are_submitted_in_one_scheduler_step_when_vllm_supports_it():
    """Looping adapters serially defeats vLLM's `max_loras` co-batching: N adapters meant
    N sequential `generate` calls, each under-filling the GPU."""
    from batcher.ml.llm.engines import _generate_routed

    llm = _CoBatchingLLM()
    prompts = [
        {"prompt": "a", "adapter": "x"},
        {"prompt": "b", "adapter": "y"},
        {"prompt": "c", "adapter": "x"},
    ]
    table = {None: None, "x": object(), "y": object()}
    texts, _usage, _reasons = _generate_routed(llm, None, prompts, table)
    assert len(llm.calls) == 1, f"expected one co-batched call, got {len(llm.calls)}"
    assert len(texts) == 3


class _LegacyLLM:
    """An older vLLM that rejects a list of LoRA requests."""

    def __init__(self) -> None:
        self.calls: list[list] = []

    def generate(self, requests, params, lora_request=None):
        if isinstance(lora_request, list):
            raise TypeError("lora_request must be a LoRARequest, not a list")
        self.calls.append(requests)
        return [_StubOutput(str(r)) for r in requests]


def test_an_older_vllm_falls_back_to_the_serial_per_adapter_loop():
    from batcher.ml.llm.engines import _generate_routed

    llm = _LegacyLLM()
    prompts = [{"prompt": "a", "adapter": "x"}, {"prompt": "b", "adapter": "y"}]
    table = {None: None, "x": object(), "y": object()}
    texts, _usage, _reasons = _generate_routed(llm, None, prompts, table)
    assert len(llm.calls) == 2, "the fallback must still produce one call per adapter"
    assert len(texts) == 2


# --- 9. finish_reason -----------------------------------------------------------------


def test_finish_reason_is_surfaced_so_truncation_is_detectable():
    """Without it, a generation cut off at `max_tokens` is indistinguishable from a
    complete one, and it silently corrupts a downstream `parse_json`."""
    from batcher.ml.llm.channels import finish_reason_sink
    from batcher.ml.llm.generate import _generate_batch

    def engine(prompts):
        finish_reason_sink().report(["length"] * len(prompts))
        return ["{"] * len(prompts)

    batch = pa.RecordBatch.from_pydict({"q": ["x"]})
    out = _generate_batch(
        engine,
        batch,
        prompt_column="q",
        output_column="response",
        template=None,
        image_column=None,
        adapter_column=None,
        parse_json=False,
        usage=False,
        finish_reason=True,
    )
    assert out.to_pydict()["finish_reason"] == ["length"]


def test_the_vllm_path_reports_a_finish_reason_per_request():
    from batcher.ml.llm.engines import _generate_routed

    llm = _CoBatchingLLM()
    _texts, _usage, reasons = _generate_routed(llm, None, ["a", "b"], {None: None})
    assert reasons == ["stop", "stop"]


# --- 10. prompt truncation ------------------------------------------------------------


def test_an_over_length_prompt_is_truncated_with_a_warning_not_a_failed_batch():
    """One over-length row failed the whole request. Windowing keeps the batch alive and
    says so."""
    from batcher.ml.llm.engines import _truncate_to_window

    class _Tok:
        def encode(self, text):
            return list(range(len(text)))

        def decode(self, ids):
            return "x" * len(ids)

    with pytest.warns(UserWarning, match="truncat"):
        out = _truncate_to_window(["x" * 100, "short"], _Tok(), max_tokens=10)
    assert len(out[0]) == 10
    assert out[1] == "short"


def test_truncation_is_a_no_op_when_everything_fits():
    from batcher.ml.llm.engines import _truncate_to_window

    class _Tok:
        def encode(self, text):
            return list(range(len(text)))

        def decode(self, ids):
            return "x" * len(ids)

    prompts = ["ab", "cde"]
    assert _truncate_to_window(prompts, _Tok(), max_tokens=10) == prompts


# --- 11/12. connection reuse and retry jitter -----------------------------------------


def test_the_http_engine_reuses_one_thread_pool_across_batches():
    """A `ThreadPoolExecutor` built per batch spawned and tore down `concurrency` threads
    every call, and each fresh thread started with no connection — a TCP connect and TLS
    handshake per request. At scale the handshakes outweigh the inference."""
    import concurrent.futures

    from batcher.ml.llm import http_engine

    built = []
    real = concurrent.futures.ThreadPoolExecutor

    class _Counting(real):
        def __init__(self, *a, **k):
            built.append(1)
            super().__init__(*a, **k)

    import batcher.ml.serving.http as http_mod

    real_post = http_mod.post_json
    http_mod.post_json = lambda url, body, **kw: {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    concurrent.futures.ThreadPoolExecutor = _Counting
    try:
        # The factory imports `post_json` at build time, so both patches precede it.
        engine = http_engine("http://x/v1", "m", concurrency=3)()
        engine(["a", "b"])
        engine(["c", "d"])
    finally:
        concurrent.futures.ThreadPoolExecutor = real
        http_mod.post_json = real_post
    assert len(built) == 1, f"a pool was built per batch ({len(built)} pools for 2 batches)"


def test_retry_backoff_is_jittered_so_workers_do_not_retry_in_lockstep():
    """`backoff * 2**attempt` is deterministic, so every worker throttled by one 429
    retries at the same instant and re-triggers the same 429.

    The jitter itself lives in `batcher.ml.serving.base`, which the LLM `http_engine`
    reaches through `post_json`; this guards the property the engine depends on.
    """
    from batcher.ml.serving.base import _jittered_backoff

    delays = {_jittered_backoff(0.5, attempt=2) for _ in range(200)}
    assert len(delays) > 1, "the delay is deterministic; a thundering herd is guaranteed"
    assert all(0.0 <= d <= 2.0 for d in delays), f"jitter must stay within the cap: {delays}"


# --- 13. bounded vision inputs --------------------------------------------------------


def test_a_large_image_is_bounded_before_it_reaches_the_model():
    """A full-resolution image shipped to the model wastes vision tokens and can exceed
    the context window outright."""
    import io

    from PIL import Image

    from batcher.ml.llm.generate import _decode_image_inputs

    buf = io.BytesIO()
    Image.new("RGBA", (4000, 3000)).save(buf, format="PNG")
    column = pa.array([buf.getvalue()], type=pa.binary())

    (image,) = _decode_image_inputs(column)
    assert max(image.size) <= 1024, f"image was not bounded: {image.size}"
    assert image.mode == "RGB", "a palette/alpha image must be normalized for the model"


def test_a_small_image_is_not_upscaled():
    import io

    from PIL import Image

    from batcher.ml.llm.generate import _decode_image_inputs

    buf = io.BytesIO()
    Image.new("RGB", (32, 16)).save(buf, format="PNG")
    (image,) = _decode_image_inputs(pa.array([buf.getvalue()], type=pa.binary()))
    assert image.size == (32, 16)


# --- the request pool is released with the worker -------------------------------------


def test_a_served_engine_releases_its_request_pool_on_close():
    """Nothing was ever shutting these pools down.

    Each served engine builds a `ThreadPoolExecutor` sized to `concurrency` and keeps it for
    the worker's lifetime, which is correct — but `close` is the teardown contract
    `core.udf.lifecycle` and `InferencePool` look for, and neither the engine nor the class
    UDF holding it defined one. So `concurrency` threads outlived every worker, per
    `collect()`, for the life of the process.
    """
    import threading

    from batcher.ml.llm import llm_udf
    from batcher.ml.llm.engines import anthropic_engine, http_engine

    for factory in (
        http_engine("http://example.invalid/v1", model="m", concurrency=4),
        anthropic_engine("m", api_key="k", concurrency=4),
    ):
        before = threading.active_count()
        udf = llm_udf(factory, prompt_column="q")()
        assert callable(getattr(udf, "close", None))
        assert callable(getattr(udf._engine, "close", None))
        udf.close()
        assert threading.active_count() <= before + 1


def test_a_served_embedding_encoder_releases_its_request_pool_on_close():
    from batcher.ml import openai_embedding_encoder, tei_encoder

    for encoder in (
        openai_embedding_encoder("m", "text", concurrency=4)(),
        tei_encoder("text", base_url="http://example.invalid", concurrency=4)(),
    ):
        assert callable(getattr(encoder, "close", None))
        encoder.close()
