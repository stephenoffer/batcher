"""`ds.ml.generate`, the load-once LLM UDF, and the chat-template path.

The engine is a plain ``list[str] -> list[str]`` callable, which is what lets the whole
generation path be tested without vLLM, a GPU, or a network: a deterministic stub engine
stands in for the model, and the assertions are about the *columnar* contract around it
— prompt construction, output typing, token-usage columns, and that the engine is built
once per worker rather than once per batch (a per-batch build reloads the model, the
most expensive mistake in this file).

`vllm_engine(chat=True)` is covered through `_generate_routed` with a stub `llm`, since
whether a conversation reaches `llm.chat` (which applies the model's chat template) or
`llm.generate` (which does not) is the difference between a tuned model answering well
and answering badly — with no error either way.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.ml.llm.engines import _chat_messages, _generate_routed, _openai_body
from batcher.ml.llm.generate import llm_udf

pytestmark = pytest.mark.unit


def _echo_engine():
    """An `EngineFactory` whose engine upper-cases each prompt."""
    return lambda prompts: [str(p).upper() for p in prompts]


def _batch(**cols) -> pa.RecordBatch:
    return pa.RecordBatch.from_pydict(cols)


# --- ds.ml.generate ----------------------------------------------------------


def test_generate_appends_the_response_column():
    ds = bt.from_pydict({"q": ["hi", "there"]})
    assert ds.ml.generate(_echo_engine, prompt_column="q").to_pydict() == {
        "q": ["hi", "there"],
        "response": ["HI", "THERE"],
    }


def test_generate_names_the_output_column():
    ds = bt.from_pydict({"q": ["hi"]})
    out = ds.ml.generate(_echo_engine, prompt_column="q", output_column="answer")
    assert out.columns == ["q", "answer"]


def test_generate_builds_prompts_from_a_template():
    ds = bt.from_pydict({"name": ["ada"], "topic": ["math"]})
    out = ds.ml.generate(
        lambda: lambda ps: list(ps), prompt_column="name", template="{name} likes {topic}"
    )
    assert out.to_pydict()["response"] == ["ada likes math"]


def test_generate_parses_json_into_a_struct_column():
    engine = lambda: lambda ps: ['{"score": 1}' for _ in ps]  # noqa: E731
    out = bt.from_pydict({"q": ["x"]}).ml.generate(engine, prompt_column="q", parse_json=True)
    assert out.to_pydict()["response"] == [{"score": 1}]


def test_generate_nulls_a_row_whose_json_does_not_parse():
    engine = lambda: lambda ps: ["not json" for _ in ps]  # noqa: E731
    out = bt.from_pydict({"q": ["x"]}).ml.generate(engine, prompt_column="q", parse_json=True)
    assert out.to_pydict()["response"] == [None]


def test_generate_appends_token_usage_columns():
    class _Engine:
        def __call__(self, prompts):
            self.last_usage = [(3, 4)] * len(prompts)
            return ["ok"] * len(prompts)

    out = bt.from_pydict({"q": ["a", "b"]}).ml.generate(
        lambda: _Engine(), prompt_column="q", usage=True
    )
    assert out.to_pydict() == {
        "q": ["a", "b"],
        "response": ["ok", "ok"],
        "prompt_tokens": [3, 3],
        "completion_tokens": [4, 4],
    }


def test_generate_builds_the_engine_once_not_once_per_batch():
    """A per-batch build would reload the model on every batch."""
    builds = []

    class _Engine:
        def __init__(self):
            builds.append(1)

        def __call__(self, prompts):
            return ["r"] * len(prompts)

    bt.from_pydict({"q": list("abcdefgh")}).ml.generate(
        lambda: _Engine(), prompt_column="q"
    ).to_pydict()
    assert len(builds) == 1


def test_generate_stays_lazy_until_a_terminal_op():
    out = bt.from_pydict({"q": ["a"]}).ml.generate(_echo_engine, prompt_column="q")
    assert isinstance(out, bt.Dataset)


def test_generate_is_a_map_batches_stage_the_rest_of_the_engine_understands():
    """Being a `MapBatches` node is what buys generation the streaming allow-list, the
    distributed map path, and the once-per-query UDF prebuild — for free."""
    from batcher.core.udf import has_map_batches
    from batcher.dist.executors.plan_analysis import _is_linear_map_pipeline
    from batcher.plan.logical.transforms import is_streamable

    plan = bt.from_pydict({"q": ["a"]}).ml.generate(_echo_engine, prompt_column="q")._plan
    assert has_map_batches(plan)
    assert is_streamable(plan)
    assert _is_linear_map_pipeline(plan)


def test_streaming_builds_the_engine_once_for_the_whole_query():
    """A streaming query prebuilds its class UDFs once and reuses them across every
    micro-batch. Rebuilding per micro-batch would reload the model each time — which
    is precisely why `llm_udf` returns a class rather than a function."""
    from batcher.core.udf.execute import prebuild_factories

    builds = []

    class _Engine:
        def __init__(self):
            builds.append(1)

        def __call__(self, prompts):
            return ["r"] * len(prompts)

    plan = bt.from_pydict({"q": ["a", "b"]}).ml.generate(lambda: _Engine(), prompt_column="q")._plan
    prebuild_factories(plan)
    assert len(builds) == 1


# --- llm_udf -----------------------------------------------------------------


def test_llm_udf_is_a_class_so_map_batches_loads_once():
    udf = llm_udf(_echo_engine, prompt_column="q")
    assert isinstance(udf, type)


def test_llm_udf_maps_a_batch_to_the_batch_plus_the_column():
    udf = llm_udf(_echo_engine, prompt_column="q")()
    out = udf(_batch(q=["hi"]))
    assert out.schema.names == ["q", "response"]
    assert out.column("response").to_pylist() == ["HI"]


def test_llm_udf_carries_the_adapter_tag_for_multi_lora():
    seen = []

    def factory():
        def engine(requests):
            seen.extend(requests)
            return ["r"] * len(requests)

        return engine

    udf = llm_udf(factory, prompt_column="q", adapter_column="ad")()
    udf(_batch(q=["a", "b"], ad=["lora1", None]))
    assert seen == [{"prompt": "a", "adapter": "lora1"}, {"prompt": "b"}]


# --- chat templating ---------------------------------------------------------


class _StubOut:
    def __init__(self, text):
        self.prompt_token_ids = [0]
        self.outputs = [type("O", (), {"text": text, "token_ids": [0, 0]})()]


class _StubLlm:
    """Records which vLLM entry point was used, and with what."""

    def __init__(self):
        self.chat_calls, self.generate_calls = [], []

    def chat(self, convos, params, lora_request=None):
        self.chat_calls.append(convos)
        return [_StubOut("chat") for _ in convos]

    def generate(self, requests, params, lora_request=None):
        self.generate_calls.append(requests)
        return [_StubOut("completion") for _ in requests]


def test_chat_false_uses_the_completion_endpoint():
    llm = _StubLlm()
    texts, _ = _generate_routed(llm, None, ["hi"], {None: None})
    assert texts == ["completion"]
    assert llm.chat_calls == []


def test_chat_true_routes_through_the_chat_template():
    llm = _StubLlm()
    texts, _ = _generate_routed(llm, None, ["hi"], {None: None}, chat=True)
    assert texts == ["chat"]
    assert llm.generate_calls == []
    assert llm.chat_calls == [[[{"role": "user", "content": "hi"}]]]


def test_chat_prepends_the_system_turn():
    llm = _StubLlm()
    _generate_routed(llm, None, ["hi"], {None: None}, chat=True, system="Be terse.")
    assert llm.chat_calls[0][0] == [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "hi"},
    ]


def test_chat_preserves_order_across_lora_groups():
    llm = _StubLlm()
    prompts = [{"prompt": "a", "adapter": "x"}, {"prompt": "b"}, {"prompt": "c", "adapter": "x"}]
    texts, usage = _generate_routed(llm, None, prompts, {None: "BASE", "x": "LX"}, chat=True)
    assert texts == ["chat", "chat", "chat"]
    assert len(usage) == 3
    assert len(llm.chat_calls) == 2  # one call per adapter group


def test_chat_messages_rejects_a_vision_request():
    with pytest.raises(ValueError, match="cannot carry an image_column"):
        _chat_messages({"prompt": "describe", "image": object()}, None)


def test_chat_messages_from_a_plain_string():
    assert _chat_messages("hi", None) == [{"role": "user", "content": "hi"}]


# --- OpenAI request body -----------------------------------------------------


def test_openai_body_omits_unset_sampling_fields():
    """A null `stop` or an unknown key is rejected by several OpenAI-compatible servers."""
    body = _openai_body("m", "hi", True, None, 16, 0.0)
    assert "top_p" not in body
    assert "stop" not in body


def test_openai_body_includes_top_p_and_stop_when_given():
    body = _openai_body("m", "hi", True, None, 16, 0.2, top_p=0.9, stop=["\n"])
    assert body["top_p"] == 0.9
    assert body["stop"] == ["\n"]


def test_openai_body_builds_chat_messages_with_a_system_turn():
    body = _openai_body("m", "hi", True, "sys", 16, 0.0)
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def test_openai_body_completion_mode_sends_a_bare_prompt():
    body = _openai_body("m", "hi", False, "sys", 16, 0.0)
    assert body["prompt"] == "hi"
    assert "messages" not in body
