"""Bedrock and Gemini engines — request bodies, reply parsing, and the failure shapes.

Nothing here reaches a network. What is tested is the pair of things that actually differ
between one hosted provider and another: what goes on the wire, and where the text sits in
the reply when the reply is not the happy path.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import PlanError
from batcher.ml.llm.engines.hosted import (
    _converse_body,
    _converse_text,
    _converse_usage,
    _converse_with_retry,
    _declared_max_tokens,
    _gemini_body,
    _gemini_finish_reason,
    _gemini_text,
    _gemini_usage,
    _image_blocks,
    _limiter_view,
    bedrock_engine,
    gemini_engine,
)

pytestmark = pytest.mark.unit


def _bedrock(**kw):
    base = {
        "model": "m",
        "prompt": "hello",
        "system": None,
        "max_tokens": 64,
        "temperature": None,
        "top_p": None,
        "stop_sequences": None,
        "additional_fields": None,
        "overrides": {},
        "image": None,
    }
    base.update(kw)
    return _converse_body(**base)


def _gemini(**kw):
    base = {
        "prompt": "hello",
        "system": None,
        "max_tokens": 64,
        "temperature": None,
        "top_p": None,
        "stop_sequences": None,
        "response_schema": None,
        "safety_settings": None,
        "extra_body": None,
        "overrides": {},
        "image": None,
    }
    base.update(kw)
    return _gemini_body(**base)


@pytest.mark.parametrize("build", [bedrock_engine, gemini_engine])
def test_an_unknown_on_error_mode_is_refused_at_construction(build):
    with pytest.raises(PlanError, match="on_error"):
        build("m", on_error="skip")


def test_bedrock_omits_an_unset_temperature_rather_than_sending_null():
    # Several model families on Bedrock reject a null they did not ask for.
    assert "temperature" not in _bedrock()["inferenceConfig"]
    assert _bedrock(temperature=0.3)["inferenceConfig"]["temperature"] == 0.3


def test_a_per_row_override_beats_the_engine_wide_default():
    body = _bedrock(max_tokens=64, overrides={"max_tokens": 8, "temperature": 0.9})
    assert body["inferenceConfig"]["maxTokens"] == 8
    assert body["inferenceConfig"]["temperature"] == 0.9


def test_bedrock_sends_a_system_turn_only_when_one_is_set():
    assert "system" not in _bedrock()
    assert _bedrock(system="be brief")["system"] == [{"text": "be brief"}]


def test_bedrock_sends_image_bytes_because_boto3_encodes_the_blob_itself():
    # Base64 here would send the base64 of the base64.
    content = _bedrock(image=b"\x89PNG")["messages"][0]["content"]
    assert content[0]["image"]["source"]["bytes"] == b"\x89PNG"
    assert content[1]["text"] == "hello"


def test_bedrock_reads_every_text_block_and_survives_a_guardrail_reply():
    # A guardrail intervention returns 200 with no text block at all.
    reply = {"output": {"message": {"content": [{"text": "a"}, {"text": "b"}]}}}
    assert _converse_text(reply) == "ab"
    assert _converse_text({"output": {"message": {"content": []}}}) == ""
    assert _converse_text({}) == ""


def test_bedrock_usage_is_reported_or_absent_never_wrong():
    assert _converse_usage({"usage": {"inputTokens": 3, "outputTokens": 4}}) == (3, 4)
    assert _converse_usage({}) == (None, None)


def test_a_throttle_is_retried_and_a_validation_error_is_not():
    class _Err(Exception):
        def __init__(self, code):
            super().__init__(code)
            self.response = {"Error": {"Code": code}}

    class _Client:
        def __init__(self, codes):
            self.codes = list(codes)
            self.calls = 0

        def converse(self, **_):
            self.calls += 1
            if self.codes:
                raise _Err(self.codes.pop(0))
            return {"output": {"message": {"content": [{"text": "ok"}]}}}

    throttled = _Client(["ThrottlingException"])
    assert _converse_with_retry(throttled, {}, 2)["output"]["message"]["content"][0]["text"] == "ok"
    assert throttled.calls == 2

    invalid = _Client(["ValidationException"])
    with pytest.raises(Exception, match="ValidationException"):
        _converse_with_retry(invalid, {}, 3)
    assert invalid.calls == 1  # a malformed request does not get better by retrying


def test_gemini_sets_both_the_schema_and_the_mime_type():
    # Setting only the schema returns prose that happens to describe it.
    config = _gemini(response_schema={"type": "object"})["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseSchema"] == {"type": "object"}


def test_gemini_omits_the_schema_fields_when_none_is_asked_for():
    assert "responseSchema" not in _gemini()["generationConfig"]


def test_gemini_puts_a_system_instruction_in_its_own_field():
    assert _gemini(system="be brief")["systemInstruction"] == {"parts": [{"text": "be brief"}]}


def test_gemini_inlines_an_image_before_the_text_part():
    parts = _gemini(image=b"\x89PNG")["contents"][0]["parts"]
    assert parts[0]["inline_data"]["mime_type"] == "image/png"
    assert parts[1]["text"] == "hello"


def test_gemini_reads_the_text_and_survives_a_blocked_reply():
    good = {"candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}]}}]}
    assert _gemini_text(good) == "ab"
    # A safety block returns no candidates; a MAX_TOKENS stop mid-thinking returns no parts.
    assert _gemini_text({"candidates": []}) == ""
    assert _gemini_text({"candidates": [{"content": {}}]}) == ""


def test_gemini_usage_and_finish_reason_are_normalized_to_the_shared_vocabulary():
    assert _gemini_usage({"usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 5}}) == (
        2,
        5,
    )
    assert _gemini_finish_reason({"candidates": [{"finishReason": "MAX_TOKENS"}]}) == "length"
    assert _gemini_finish_reason({"candidates": [{"finishReason": "STOP"}]}) == "stop"
    assert _gemini_finish_reason({"candidates": [{"finishReason": "SAFETY"}]}) == "content_filter"
    assert _gemini_finish_reason({"candidates": []}) == "content_filter"


def test_an_unrecognized_finish_reason_passes_through_rather_than_being_dropped():
    assert _gemini_finish_reason({"candidates": [{"finishReason": "OTHER"}]}) == "OTHER"


def test_the_limiter_sees_the_generation_budget_of_either_provider():
    # One estimator, two wire shapes: translating here is what stops one of them quietly
    # ceasing to count its reservations.
    assert _declared_max_tokens(_bedrock(max_tokens=200)) == 200
    assert _declared_max_tokens(_gemini(max_tokens=300)) == 300
    assert _declared_max_tokens({}) == 0


def test_the_limiter_counts_images_in_either_wire_shape():
    assert _image_blocks(_bedrock(image=b"x")) == 1
    assert _image_blocks(_gemini(image=b"x")) == 1
    assert _image_blocks(_bedrock()) == 0
    assert _image_blocks(_gemini()) == 0


def test_the_limiter_view_charges_for_the_reservation_and_the_image():
    from batcher.ml.llm.engines.limits import _estimated_tokens

    view = _limiter_view(_gemini(max_tokens=100, image=b"x"), "hello")
    assert _estimated_tokens("hello", view) > 100  # the reservation plus the image
