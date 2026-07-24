"""The Anthropic Messages API engine — request shaping, response parsing, tolerance.

Every test fakes `batcher.ml.serving.http.post_json` (the factory imports it at build
time, so the engine is built inside the patch). No network, key, or SDK.
"""

from __future__ import annotations

import contextlib

import pytest

import batcher.ml.serving.http as http_mod
from batcher.ml import anthropic_engine


@contextlib.contextmanager
def _patched(fn):
    real = http_mod.post_json
    http_mod.post_json = fn
    try:
        yield
    finally:
        http_mod.post_json = real


def _reply(text="ok", stop_reason="end_turn", inp=3, out=2):
    return {
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "usage": {"input_tokens": inp, "output_tokens": out},
    }


def test_builds_a_messages_body_with_required_max_tokens():
    bodies = []

    def fake(url, body, **kw):
        assert url.endswith("/messages")
        bodies.append(body)
        return _reply()

    with _patched(fake):
        engine = anthropic_engine("claude-haiku-4-5", system="be terse", concurrency=1)()
        out = engine(["hello"])
    assert out == ["ok"]
    body = bodies[0]
    assert body["model"] == "claude-haiku-4-5"
    assert body["max_tokens"] == 1024
    assert body["system"] == "be terse"
    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_temperature_is_omitted_unless_set():
    bodies = []

    def fake(url, body, **kw):
        bodies.append(body)
        return _reply()

    with _patched(fake):
        anthropic_engine("m", concurrency=1)()(["hi"])
        anthropic_engine("m", temperature=0.5, concurrency=1)()(["hi"])
    assert "temperature" not in bodies[0]
    assert bodies[1]["temperature"] == 0.5


def test_per_row_overrides_win():
    bodies = []

    def fake(url, body, **kw):
        bodies.append(body)
        return _reply()

    with _patched(fake):
        engine = anthropic_engine("m", max_tokens=1024, concurrency=1)()
        engine([{"prompt": "hi", "max_tokens": 16, "temperature": 0.9}])
    assert bodies[0]["max_tokens"] == 16
    assert bodies[0]["temperature"] == 0.9
    assert bodies[0]["messages"][0]["content"] == "hi"


def test_usage_and_finish_reason_are_reported_and_normalized():
    from batcher.ml.llm.channels import finish_reason_sink, usage_sink

    def fake(url, body, **kw):
        return _reply(stop_reason="max_tokens", inp=5, out=7)

    with _patched(fake), usage_sink().capture(), finish_reason_sink().capture():
        engine = anthropic_engine("m", concurrency=1)()
        engine(["hi"])
        assert usage_sink().collected() == [(5, 7)]
        # max_tokens normalizes to the shared "length" vocabulary.
        assert finish_reason_sink().collected() == ["length"]


def test_a_refusal_with_empty_content_does_not_crash():
    def fake(url, body, **kw):
        return {"content": [], "stop_reason": "refusal", "usage": {}}

    with _patched(fake):
        engine = anthropic_engine("m", concurrency=1)()
        out = engine(["dubious"])
    assert out == [""]


def test_on_error_null_skips_a_failed_row():
    from batcher._internal.errors import BackendError

    def fake(url, body, **kw):
        if "boom" in body["messages"][0]["content"]:
            raise BackendError("400")
        return _reply()

    with _patched(fake):
        engine = anthropic_engine("m", on_error="null", concurrency=1)()
        out = engine(["fine", "boom", "fine"])
    assert out == ["ok", "", "ok"]


def test_on_error_is_validated():
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError):
        anthropic_engine("m", on_error="maybe")
