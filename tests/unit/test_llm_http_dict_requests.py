"""`http_engine` correctness for per-row request dicts, lenient JSON, and packing types.

Regression coverage for three silent failures:
  * `http_engine` treated a ``{"prompt": ...}`` request dict (produced whenever a per-row
    column is set) as the prompt string, sending a malformed body to a served endpoint.
  * `parse_json` / `extract` rejected Markdown-fenced or prose-wrapped JSON.
  * sequence packing dropped every token of a ``large_list``/``fixed_size_list`` column.
All fakes; no network, model, or GPU.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher.ml.serving.http as http_mod
from batcher.ml.llm import http_engine
from batcher.ml.llm.columns import _loads_lenient


def _capture_bodies():
    """Patch `post_json` to record request bodies and reply with a fixed completion."""
    import contextlib

    bodies: list[dict] = []

    def fake(url, body, **kw):
        bodies.append(body)
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    @contextlib.contextmanager
    def cm():
        real = http_mod.post_json
        http_mod.post_json = fake
        try:
            yield bodies
        finally:
            http_mod.post_json = real

    return cm()


def test_per_row_max_tokens_and_temperature_reach_the_body():
    """A request dict's per-row overrides win over the engine-wide defaults."""
    # The factory binds `post_json` at build time, so build inside the patch.
    with _capture_bodies() as bodies:
        engine = http_engine("http://x/v1", "m", max_tokens=512, temperature=0.0, concurrency=1)()
        out = engine([{"prompt": "hi", "max_tokens": 16, "temperature": 0.7}])
    assert out == ["ok"]
    assert bodies[0]["max_tokens"] == 16
    assert bodies[0]["temperature"] == 0.7
    # The prompt is unwrapped from the dict into the chat message, not sent as an object.
    assert bodies[0]["messages"][-1]["content"] == "hi"


def test_a_plain_string_request_still_uses_engine_defaults():
    with _capture_bodies() as bodies:
        engine = http_engine("http://x/v1", "m", max_tokens=128, concurrency=1)()
        engine(["plain prompt"])
    assert bodies[0]["max_tokens"] == 128
    assert bodies[0]["messages"][-1]["content"] == "plain prompt"


def test_on_error_null_yields_empty_text_instead_of_failing_the_batch():
    """One un-retryable request nulls its row rather than aborting the whole batch."""
    from batcher._internal.errors import BackendError

    calls = {"n": 0}

    def flaky(url, body, **kw):
        calls["n"] += 1
        if "boom" in str(body["messages"][-1]["content"]):
            raise BackendError("400 bad request")
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    import contextlib

    @contextlib.contextmanager
    def patched():
        real = http_mod.post_json
        http_mod.post_json = flaky
        try:
            yield
        finally:
            http_mod.post_json = real

    with patched():
        engine = http_engine("http://x/v1", "m", on_error="null", concurrency=1)()
        out = engine(["fine", "boom", "fine"])
    assert out == ["ok", "", "ok"]


def test_on_error_raise_is_the_default():
    from batcher._internal.errors import BackendError, PlanError

    def boom(url, body, **kw):
        raise BackendError("nope")

    import contextlib

    @contextlib.contextmanager
    def patched():
        real = http_mod.post_json
        http_mod.post_json = boom
        try:
            yield
        finally:
            http_mod.post_json = real

    with patched(), pytest.raises(BackendError):
        engine = http_engine("http://x/v1", "m", concurrency=1)()
        engine(["x"])
    with pytest.raises(PlanError):
        http_engine("http://x/v1", "m", on_error="maybe")


def test_logprobs_are_summed_and_reported():
    """`generate(logprobs=True)` works over HTTP: per-token logprobs sum to one value."""
    from batcher.ml.llm.channels import logprob_sink

    def fake(url, body, **kw):
        assert body.get("logprobs") is True
        return {
            "choices": [
                {
                    "message": {"content": "ok"},
                    "finish_reason": "stop",
                    "logprobs": {"content": [{"logprob": -0.5}, {"logprob": -1.5}]},
                }
            ]
        }

    import contextlib

    @contextlib.contextmanager
    def patched():
        real = http_mod.post_json
        http_mod.post_json = fake
        try:
            yield
        finally:
            http_mod.post_json = real

    with patched(), logprob_sink().capture():
        engine = http_engine("http://x/v1", "m", logprobs=True, concurrency=1)()
        engine(["hi"])
        assert logprob_sink().collected() == [-2.0]


def test_seed_and_response_format_reach_the_body():
    with _capture_bodies() as bodies:
        engine = http_engine(
            "http://x/v1", "m", seed=7, response_format={"type": "json_object"}, concurrency=1
        )()
        engine(["hi"])
    assert bodies[0]["seed"] == 7
    assert bodies[0]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('```\n{"a": 1}\n```', {"a": 1}),
        (
            'Here is the JSON: {"vendor": "Acme", "total": 12.5}. Done.',
            {"vendor": "Acme", "total": 12.5},
        ),
        ('{"note": "a } brace inside", "x": 2}', {"note": "a } brace inside", "x": 2}),
        ("[1, 2, 3]", [1, 2, 3]),
        ("not json at all", None),
        ("", None),
    ],
)
def test_loads_lenient_handles_fenced_and_wrapped_json(text, expected):
    assert _loads_lenient(text) == expected


def test_loads_lenient_recovers_an_unclosed_fence():
    assert _loads_lenient('```json\n{"a": 1}') == {"a": 1}


def test_packing_reads_large_list_tokens():
    """A large_list token column (fast-tokenizer output) is no longer silently dropped."""
    from batcher.ml.llm import pack_sequences

    col = pa.array([[1, 2, 3], [4, 5]], type=pa.large_list(pa.int64()))
    b = pa.RecordBatch.from_arrays([col], names=["tokens"])
    out = list(pack_sequences([b], seq_len=5))
    assert out[0].column("tokens").to_pylist() == [[1, 2, 3, 4, 5]]


def test_packing_reads_fixed_size_list_tokens():
    from batcher.ml.llm import pack_sequences

    col = pa.array([[1, 2], [3, 4], [5, 6]], type=pa.list_(pa.int64(), 2))
    b = pa.RecordBatch.from_arrays([col], names=["tokens"])
    out = list(pack_sequences([b], seq_len=3))
    assert out[0].column("tokens").to_pylist() == [[1, 2, 3], [4, 5, 6]]
