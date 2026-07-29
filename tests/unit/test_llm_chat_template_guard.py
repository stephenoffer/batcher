"""An instruction-tuned model sent raw completions is a silent quality loss.

An instruct model was trained on conversations rendered through its own chat template.
Send it a bare completion prompt and it still answers — in a format it was never tuned on.
Nothing raises, no budget is exceeded, and the only symptom is output that is quietly
worse. `vllm_engine`'s own docstring calls it "degraded output with nothing to signal it".

It is detectable: a model shipped for instruction following carries a `chat_template` and a
base model does not. The check fires only when the caller never said which mode they
wanted, because asking an instruct model for raw completions is legitimate — that is what
constrained-choice classification does.
"""

from __future__ import annotations

import sys
import types
import warnings
from typing import ClassVar

import pytest

from batcher._internal.errors import PerformanceWarning
from batcher.ml.llm.engines import templates
from batcher.ml.llm.engines import vllm as vllm_mod

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_once_flag():
    templates._TEMPLATE_WARNED = False
    yield
    templates._TEMPLATE_WARNED = False


class _Tokenizer:
    def __init__(self, template):
        self.chat_template = template


def _install(monkeypatch, template):
    """A fake vLLM whose tokenizer reports `template` as its chat template."""

    class _FakeLLM:
        constructed: ClassVar[list] = []

        def __init__(self, **kw):
            type(self).constructed.append(kw)

        def get_tokenizer(self):
            return _Tokenizer(template)

        def generate(self, requests, *a, **k):
            return [_Out() for _ in requests]

    class _Completion:
        text = ""
        token_ids: ClassVar[tuple] = ()
        finish_reason = "stop"
        cumulative_logprob = None

    class _Out:
        prompt_token_ids: ClassVar[tuple] = ()

        def __init__(self):
            self.outputs = [_Completion()]

    class _SamplingParams:
        def __init__(self, **kw):
            pass

    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setattr(
        "batcher._internal.optional.require",
        lambda mod, name=None, **kw: {"LLM": _FakeLLM, "SamplingParams": _SamplingParams}[name],
    )


def test_an_instruct_model_left_unset_warns(monkeypatch):
    _install(monkeypatch, "{% for m in messages %}{{ m }}{% endfor %}")
    with pytest.warns(PerformanceWarning, match="chat template"):
        vllm_mod.vllm_engine("some/Instruct-Model")()


def test_a_base_model_left_unset_stays_silent(monkeypatch):
    """A base model has no template, so the completion path is exactly right."""
    _install(monkeypatch, None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        vllm_mod.vllm_engine("some/Base-Model")()


def test_an_explicit_false_is_a_decision_not_a_mistake(monkeypatch):
    """Constrained-choice classification deliberately wants raw completions from an
    instruct model; advising against it there would be wrong."""
    _install(monkeypatch, "{{ messages }}")
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        vllm_mod.vllm_engine("some/Instruct-Model", chat=False)()


def test_chat_true_never_warns(monkeypatch):
    _install(monkeypatch, "{{ messages }}")
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        vllm_mod.vllm_engine("some/Instruct-Model", chat=True)()


def test_the_message_names_the_model(monkeypatch):
    _install(monkeypatch, "{{ messages }}")
    with pytest.warns(PerformanceWarning, match="Qwen"):
        vllm_mod.vllm_engine("Qwen/Qwen2.5-7B-Instruct")()


def test_template_detection_tolerates_an_odd_tokenizer():
    """Tokenizer implementations vary; a probe failure must not break engine construction."""

    class _Hostile:
        @property
        def chat_template(self):
            raise RuntimeError("no such attribute")

    assert templates.has_chat_template(_Hostile()) is False
    assert templates.has_chat_template(None) is False
