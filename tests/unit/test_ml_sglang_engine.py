"""The SGLang engine adapter — sampling normalization, constraints, and signal unpacking.

Every case drives the pure seams with no `sglang` installed and no GPU: what the engine
sends, and how it reads back what an engine returns.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import PlanError
from batcher.ml.llm.engines.sglang import (
    _cumulative_logprob,
    _finish_reason,
    _one_constraint,
    _render,
    _row_params,
    _sampling_defaults,
    _unpack,
)

pytestmark = pytest.mark.unit


def test_defaults_are_greedy_like_every_other_backend():
    assert _sampling_defaults(None, {})["temperature"] == 0.0


def test_max_tokens_is_accepted_as_the_cross_backend_alias():
    # A `sampling` dict written for vllm_engine/http_engine must transfer unchanged; dropping
    # the alias would silently cap every generation at SGLang's own default.
    params = _sampling_defaults({"max_tokens": 64}, {})
    assert params["max_new_tokens"] == 64
    assert "max_tokens" not in params


def test_an_explicit_max_new_tokens_wins_over_the_alias():
    params = _sampling_defaults({"max_tokens": 64, "max_new_tokens": 8}, {})
    assert params["max_new_tokens"] == 8


def test_a_json_schema_is_serialized_because_sglang_wants_a_string():
    assert _one_constraint({"type": "object"}, None, None) == {"json_schema": '{"type": "object"}'}


def test_a_schema_already_given_as_a_string_passes_through():
    assert _one_constraint('{"type": "object"}', None, None)["json_schema"] == '{"type": "object"}'


@pytest.mark.parametrize(
    "kwargs",
    [
        {"json_schema": {"a": 1}, "regex": "x"},
        {"regex": "x", "ebnf": "root ::= 'a'"},
        {"json_schema": {"a": 1}, "regex": "x", "ebnf": "root ::= 'a'"},
    ],
)
def test_more_than_one_constraint_is_refused_at_construction(kwargs):
    # SGLang silently prefers one over another, which produces output constrained by a rule
    # the caller did not choose — a plausible-looking wrong shape a thousand rows later.
    with pytest.raises(PlanError, match="one structured-output constraint"):
        _one_constraint(kwargs.get("json_schema"), kwargs.get("regex"), kwargs.get("ebnf"))


def test_no_constraint_is_an_empty_fragment():
    assert _one_constraint(None, None, None) == {}


def test_a_row_without_overrides_reuses_the_shared_defaults_object():
    defaults = {"temperature": 0.0}
    assert _row_params(defaults, "a plain prompt") is defaults
    assert _row_params(defaults, {"prompt": "x"}) is defaults


def test_per_row_overrides_are_translated_to_sglangs_spelling():
    params = _row_params({"temperature": 0.0}, {"prompt": "x", "max_tokens": 16})
    assert params["max_new_tokens"] == 16
    assert params["temperature"] == 0.0


def test_signals_are_unpacked_in_request_order():
    outputs = [
        {"text": "a", "meta_info": {"prompt_tokens": 3, "completion_tokens": 2}},
        {"text": "b", "meta_info": {"prompt_tokens": 5, "completion_tokens": 1}},
    ]
    texts, usage, _, _ = _unpack(outputs, 2)
    assert texts == ["a", "b"]
    assert usage == [(3, 2), (5, 1)]


def test_a_single_request_returning_a_bare_dict_is_not_indexed_as_a_string():
    # `outputs[0]` on a dict is a KeyError; on a string it is a character. Normalizing the
    # shape is why the engine's return is always one string per request.
    texts, _, _, _ = _unpack({"text": "hello", "meta_info": {}}, 1)
    assert texts == ["hello"]


def test_missing_outputs_leave_empty_strings_rather_than_shifting_the_batch():
    texts, usage, _, _ = _unpack([{"text": "a", "meta_info": {}}], 3)
    assert texts == ["a", "", ""]
    assert usage[1] is None


def test_the_finish_reason_is_flattened_out_of_sglangs_dict():
    # It is reported as {"type": "stop"}; the `finish_reason` column is a string column, so a
    # raw dict would land a struct in it.
    assert _finish_reason({"type": "length", "length": 128}) == "length"
    assert _finish_reason("stop") == "stop"
    assert _finish_reason(None) is None


def test_the_cumulative_logprob_is_summed_to_match_the_other_backends():
    meta = {"output_token_logprobs": [(-0.5, 1, "a"), (-1.5, 2, "b")]}
    assert _cumulative_logprob(meta) == pytest.approx(-2.0)


def test_no_logprobs_requested_reports_none_rather_than_zero():
    assert _cumulative_logprob({}) is None
    assert _cumulative_logprob({"output_token_logprobs": []}) is None


def test_the_completion_path_sends_the_prompt_unchanged():
    assert _render("hello", chat=False, system=None, tokenizer=object()) == "hello"


def test_the_chat_path_applies_the_models_own_template():
    class _Tok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is False and add_generation_prompt is True
            return "|".join(f"{m['role']}:{m['content']}" for m in messages)

    rendered = _render("q", chat=True, system="be brief", tokenizer=_Tok())
    assert rendered == "system:be brief|user:q"


def test_a_model_with_no_template_falls_back_to_the_raw_prompt():
    class _Tok:
        def apply_chat_template(self, *args, **kwargs):
            raise ValueError("no chat template")

    assert _render("q", chat=True, system=None, tokenizer=_Tok()) == "q"


def test_a_request_dict_is_unpacked_for_its_prompt():
    assert _render({"prompt": "q", "max_tokens": 4}, chat=False, system=None, tokenizer=None) == "q"


def test_a_text_only_batch_sends_no_image_data():
    # An SGLang build serving a text model rejects the keyword, so it must not be sent.
    from batcher.ml.llm.engines.sglang import _image_data

    assert _image_data(["a", {"prompt": "b"}]) is None


def test_a_mixed_batch_carries_one_aligned_entry_per_request():
    # Dropping the image and generating from the prompt alone is the worst outcome here: the
    # model answers a question about a picture it never saw, and nothing says so.
    from batcher.ml.llm.engines.sglang import _image_data

    assert _image_data(["a", {"prompt": "b", "image": b"png"}]) == [None, b"png"]
