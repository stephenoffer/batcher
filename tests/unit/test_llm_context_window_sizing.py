"""Sizing a vLLM context window from the data rather than from the model's maximum.

The KV cache is reserved from `max_model_len`, so a 128K default over a corpus of 2K-token
prompts spends nearly all of it on lengths the data never reaches — capacity that would
otherwise be concurrent sequences. The field guides put the gap at 2-10x throughput and
prescribe a manual procedure (sample the corpus, compute P95/P99, set the number).

The invariant every test here defends is that the sizing may only ever err **generous**. A
window smaller than a prompt truncates it, which degrades output silently — strictly worse
than an oversized cache. So the arithmetic over-estimates tokens, adds the full generation
budget, applies headroom, and rounds up; and if the model refuses the result, the run falls
back to the model's own window rather than failing.

No vLLM and no GPU: the engine class is a fake, which is what lets the *wiring* be tested
and not just the arithmetic.
"""

from __future__ import annotations

import sys
import types
from typing import ClassVar

import pytest

from batcher.ml.llm.engines import vllm as vllm_engine_mod
from batcher.ml.llm.sizing import auto_max_model_len, estimate_tokens

pytestmark = pytest.mark.unit


# --- the arithmetic ------------------------------------------------------------------


def test_token_estimate_is_an_upper_bound():
    """Two characters per token is denser than any realistic tokenizer, so the estimate
    cannot come in low — which is the only direction that would truncate a prompt."""
    assert estimate_tokens(1000) == 500
    assert estimate_tokens(0) == 0
    assert estimate_tokens(-5) == 0


def test_a_short_corpus_reclaims_most_of_a_long_default_window():
    """The whole point: 4,000-character prompts do not need a 128K window."""
    window = auto_max_model_len(4000, max_gen_tokens=256, model_default=131072)
    assert window is not None
    assert window < 131072 // 10


def test_the_window_always_covers_the_longest_prompt_plus_generation():
    """The window must hold the prompt *and* everything generated from it."""
    chars, gen = 8000, 2000
    window = auto_max_model_len(chars, max_gen_tokens=gen, model_default=131072)
    assert window >= estimate_tokens(chars) + gen


def test_headroom_covers_prompts_the_sample_never_saw():
    """The sample is one batch of a corpus; the true maximum can exceed it."""
    chars = 8000
    window = auto_max_model_len(chars, model_default=131072)
    assert window > estimate_tokens(chars)


def test_no_proposal_when_there_is_nothing_to_reclaim():
    """A proposal at or above the model's own window saves no cache, so the default is left
    alone rather than pinned to a number that only looks deliberate."""
    assert auto_max_model_len(4000, max_gen_tokens=256, model_default=4096) is None
    assert auto_max_model_len(1_000_000, model_default=8192) is None


def test_an_empty_sample_proposes_nothing():
    """Sizing from no observation would be a guess, and a guess here truncates prompts."""
    assert auto_max_model_len(0) is None


def test_windows_are_bucketed_so_small_drift_does_not_move_them():
    """A workload whose lengths wobble between runs should keep asking for the same window
    rather than rebuilding the cache each time."""
    a = auto_max_model_len(6000, model_default=131072)
    b = auto_max_model_len(6001, model_default=131072)
    assert a == b
    assert a % 1024 == 0


def test_the_window_grows_monotonically_with_prompt_length():
    lengths = [2000, 8000, 32000, 120000]
    windows = [auto_max_model_len(n, model_default=1_000_000) for n in lengths]
    assert windows == sorted(windows)


# --- the wiring ----------------------------------------------------------------------


class _FakeLLM:
    """Records how it was constructed; refuses a window above `max_supported`."""

    constructed: ClassVar[list[dict]] = []
    max_supported = 131072

    def __init__(self, **kwargs):
        type(self).constructed.append(dict(kwargs))
        window = kwargs.get("max_model_len")
        if window is not None and window > type(self).max_supported:
            raise ValueError(f"max_model_len {window} exceeds model maximum")

    def generate(self, requests, *args, **kwargs):
        return [_FakeOutput() for _ in requests]

    def get_tokenizer(self):
        return None


class _FakeCompletion:
    text = ""
    token_ids: ClassVar[tuple] = ()
    finish_reason = "stop"
    cumulative_logprob = None


class _FakeOutput:
    """The shape `_generate_signals` reads: one completion, plus prompt token ids."""

    prompt_token_ids: ClassVar[tuple] = ()

    def __init__(self) -> None:
        self.outputs = [_FakeCompletion()]


def _install_fake_vllm(monkeypatch):
    """Make `require("vllm", ...)` resolve to the fake engine and a trivial SamplingParams."""
    _FakeLLM.constructed = []
    _FakeLLM.max_supported = 131072

    class _SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def _require(module, name=None, **kwargs):
        return {"LLM": _FakeLLM, "SamplingParams": _SamplingParams}[name]

    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setattr("batcher._internal.optional.require", _require)
    return _FakeLLM


def test_auto_defers_the_build_until_it_has_seen_a_batch(monkeypatch):
    """Sizing from the data means the engine cannot exist before the data does."""
    fake = _install_fake_vllm(monkeypatch)
    engine = vllm_engine_mod.vllm_engine("m", max_model_len="auto")()
    assert fake.constructed == [], "the engine was built before any prompt was seen"
    engine(["hello " * 200])
    assert len(fake.constructed) == 1
    assert fake.constructed[0]["max_model_len"] > 0


def test_an_explicit_window_still_builds_eagerly(monkeypatch):
    """Only the opt-in path changes construction timing; everything else is as before."""
    fake = _install_fake_vllm(monkeypatch)
    vllm_engine_mod.vllm_engine("m", max_model_len=8192)()
    assert len(fake.constructed) == 1
    assert fake.constructed[0]["max_model_len"] == 8192


def test_default_path_builds_eagerly_and_pins_no_window(monkeypatch):
    fake = _install_fake_vllm(monkeypatch)
    vllm_engine_mod.vllm_engine("m")()
    assert len(fake.constructed) == 1
    assert "max_model_len" not in fake.constructed[0]


def test_a_refused_window_falls_back_to_the_models_own(monkeypatch):
    """Auto sizing may cost throughput; it must never cost a run. The model's true maximum
    is not knowable before the engine exists, so the refusal is the signal."""
    fake = _install_fake_vllm(monkeypatch)
    fake.max_supported = 1024  # smaller than any window the sizer will propose
    engine = vllm_engine_mod.vllm_engine("m", max_model_len="auto")()
    with pytest.warns(UserWarning, match="max_model_len='auto'"):
        engine(["hello " * 500])
    assert len(fake.constructed) == 2, "expected one refused attempt then one fallback"
    assert "max_model_len" not in fake.constructed[1]


def test_the_generation_budget_is_reserved_not_assumed_to_fit(monkeypatch):
    fake = _install_fake_vllm(monkeypatch)
    prompts = ["x" * 4000]
    small = vllm_engine_mod.vllm_engine("m", max_model_len="auto")()
    small(prompts)
    big = vllm_engine_mod.vllm_engine("m", max_model_len="auto", sampling={"max_tokens": 8000})()
    big(prompts)
    assert fake.constructed[1]["max_model_len"] > fake.constructed[0]["max_model_len"]


def test_the_engine_is_built_once_across_many_batches(monkeypatch):
    """Deferring the build must not turn a load-once model into a load-per-batch one — the
    exact antipattern the class-UDF shape exists to prevent."""
    fake = _install_fake_vllm(monkeypatch)
    engine = vllm_engine_mod.vllm_engine("m", max_model_len="auto")()
    for _ in range(5):
        engine(["hello"])
    assert len(fake.constructed) == 1


def test_a_dict_request_is_measured_by_its_prompt_text(monkeypatch):
    """A vision request is a ``{prompt, image}`` dict; sizing must read the text, not the
    dict's repr, or an image would inflate the window it asks for."""
    fake = _install_fake_vllm(monkeypatch)
    engine = vllm_engine_mod.vllm_engine("m", max_model_len="auto")()
    engine([{"prompt": "hi", "image": object()}])
    assert fake.constructed[0]["max_model_len"] == 2048  # the floor, not an inflated window
