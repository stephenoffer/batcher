"""`vllm_engine(truncation=...)` — the context-window policy is explicit (AP-390).

A fake tokenizer (one token per character) stands in for the worker's, so the policy is
testable with no GPU and no vLLM install.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import DataQualityError, PlanError
from batcher.ml.llm.sizing import _truncate_to_window, fit_to_window

pytestmark = pytest.mark.unit


class _CharTokenizer:
    def encode(self, text):
        return list(text)

    def decode(self, ids):
        return "".join(ids)


def test_head_keeps_the_first_tokens_and_is_the_default():
    with pytest.warns(UserWarning, match="truncation='head'"):
        out = _truncate_to_window(["abcdefghij", "ok"], _CharTokenizer(), 4)
    assert out == ["abcd", "ok"]


def test_tail_keeps_the_last_tokens():
    with pytest.warns(UserWarning, match="losing their heads"):
        out = _truncate_to_window(["abcdefghij", "ok"], _CharTokenizer(), 4, policy="tail")
    assert out == ["ghij", "ok"]


def test_error_refuses_to_cut_and_names_the_count():
    with pytest.raises(DataQualityError, match="1 of 2 prompts") as info:
        _truncate_to_window(["abcdefghij", "ok"], _CharTokenizer(), 4, policy="error")
    assert info.value.violations == {"context_window": 1}


def test_error_is_silent_when_everything_fits():
    assert _truncate_to_window(["ab", "cd"], _CharTokenizer(), 4, policy="error") == ["ab", "cd"]


def test_the_policy_reaches_dict_requests_through_fit_to_window():
    prompts = [{"prompt": "abcdefghij", "adapter": "a"}, "xyz"]
    with pytest.warns(UserWarning):
        out = fit_to_window(prompts, _CharTokenizer(), 4, "tail")
    assert out == [{"prompt": "ghij", "adapter": "a"}, "xyz"]


def test_an_unknown_policy_is_refused_before_any_model_loads():
    from batcher.ml import vllm_engine

    with pytest.raises(PlanError, match="truncation must be one of"):
        vllm_engine("some/model", truncation="middle")  # type: ignore[arg-type]


def test_a_valid_policy_builds_a_factory_without_importing_vllm():
    from batcher.ml import vllm_engine

    assert callable(vllm_engine("some/model", truncation="error"))
