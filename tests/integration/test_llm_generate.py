"""LLM batch inference (E) — load-once engine, whole-batch request stream, templating,
structured-output parsing. Verified with deterministic stub engines (no GPU/vLLM)."""

from __future__ import annotations

import json

import pyarrow as pa

from batcher.ml import llm_generate


def _batches(prompts: list[str], col: str = "q") -> list[pa.RecordBatch]:
    return [pa.RecordBatch.from_arrays([pa.array(prompts, type=pa.string())], names=[col])]


def test_generates_response_column_in_order():
    # Stub engine: echo upper-cased — deterministic, order-preserving.
    builds = []

    def factory():
        builds.append(1)  # count how many times the engine is built
        return lambda prompts: [p.upper() for p in prompts]

    out = list(llm_generate(_batches(["a", "b", "c"]), factory, prompt_column="q", num_workers=2))
    table = pa.Table.from_batches(out)
    assert table.column("q").to_pylist() == ["a", "b", "c"]
    assert table.column("response").to_pylist() == ["A", "B", "C"]
    # Built once per worker, never once per batch (the load-once contract).
    assert len(builds) <= 2


def test_prompt_template_builds_from_columns():
    tbl = pa.RecordBatch.from_arrays(
        [pa.array(["cap of France?"]), pa.array(["Be terse."])], names=["question", "system"]
    )

    def factory():
        return lambda prompts: list(prompts)  # echo the rendered prompt

    out = list(
        llm_generate(
            [tbl],
            factory,
            prompt_column="question",
            template="{system} Q: {question}",
            output_column="prompt_seen",
        )
    )
    assert pa.Table.from_batches(out).column("prompt_seen").to_pylist() == [
        "Be terse. Q: cap of France?"
    ]


def test_structured_json_output_parsed_to_struct():
    def factory():
        return lambda prompts: [json.dumps({"label": p, "score": len(p)}) for p in prompts]

    out = list(
        llm_generate(
            _batches(["spam", "ok"]),
            factory,
            prompt_column="q",
            output_column="cls",
            parse_json=True,
        )
    )
    col = pa.Table.from_batches(out).column("cls").to_pylist()
    assert col == [{"label": "spam", "score": 4}, {"label": "ok", "score": 2}]


def test_bad_json_becomes_null():
    def factory():
        return lambda prompts: ["not json" for _ in prompts]

    out = list(
        llm_generate(
            _batches(["x"]), factory, prompt_column="q", output_column="j", parse_json=True
        )
    )
    assert pa.Table.from_batches(out).column("j").to_pylist() == [None]


def test_http_engine_sends_prompts_concurrently_in_order(monkeypatch):
    # Patch the transport: record call order, return a per-prompt chat completion. With
    # concurrency the requests overlap, but the engine must return results in input order.
    import threading

    from batcher.ml.llm import http_engine

    seen: list[str] = []
    lock = threading.Lock()

    def fake_post_json(url, payload, *, headers, timeout, **kw):
        prompt = payload["messages"][-1]["content"]
        with lock:
            seen.append(prompt)
        return {"choices": [{"message": {"content": prompt.upper()}}]}

    monkeypatch.setattr("batcher.ml.serving.http.post_json", fake_post_json)
    engine = http_engine("http://x/v1", "m", concurrency=4)()
    prompts = [f"p{i}" for i in range(12)]
    assert engine(prompts) == [p.upper() for p in prompts]  # order preserved
    assert sorted(seen) == sorted(prompts)  # every prompt sent exactly once


def test_group_indices_by_adapter():
    from batcher.ml.llm import _group_indices_by_adapter

    prompts = [
        {"prompt": "a", "adapter": "x"},
        "b",
        {"prompt": "c", "adapter": "y"},
        {"prompt": "d", "adapter": "x"},
    ]
    assert _group_indices_by_adapter(prompts) == {"x": [0, 3], None: [1], "y": [2]}


def test_build_requests_carries_adapter_tag():
    from batcher.ml.llm import _build_requests

    batch = pa.RecordBatch.from_pydict({"q": ["a", "b"], "ad": ["lora1", None]})
    reqs = _build_requests(None, "q", None, "ad", batch)
    assert reqs == [{"prompt": "a", "adapter": "lora1"}, {"prompt": "b"}]


def test_generate_routed_groups_by_adapter_preserves_order():
    from batcher.ml.llm import _generate_routed

    class _Out:
        def __init__(self, text, pt, ct):
            self.prompt_token_ids = [0] * pt
            self.outputs = [type("O", (), {"text": text, "token_ids": [0] * ct})()]

    calls = []

    class _LLM:
        def generate(self, requests, params, lora_request=None):
            calls.append(lora_request)
            return [_Out(f"{r['prompt']}|{lora_request}", 1, 2) for r in requests]

    prompts = [{"prompt": "a", "adapter": "x"}, {"prompt": "b"}, {"prompt": "c", "adapter": "x"}]
    table = {None: "BASE", "x": "LORAX"}
    texts, usage = _generate_routed(_LLM(), None, prompts, table)
    assert texts == ["a|LORAX", "b|BASE", "c|LORAX"]  # output order preserved across groups
    assert usage == [(1, 2), (1, 2), (1, 2)]
    assert sorted(str(c) for c in calls) == ["BASE", "LORAX"]  # one generate per adapter group


def test_llm_generate_routes_adapter_per_row():
    def factory():
        def engine(reqs):
            return [f"{r['prompt']}:{r.get('adapter', 'base')}" for r in reqs]

        return engine

    batch = pa.RecordBatch.from_pydict({"q": ["a", "b"], "ad": ["x", None]})
    out = list(llm_generate([batch], factory, prompt_column="q", adapter_column="ad"))
    assert pa.Table.from_batches(out).column("response").to_pylist() == ["a:x", "b:base"]


def test_usage_columns_appended_when_engine_reports():
    # An engine that reports per-prompt token usage → prompt/completion token columns.
    def factory():
        def engine(prompts):
            engine.last_usage = [(len(p), len(p) * 2) for p in prompts]
            return [p.upper() for p in prompts]

        return engine

    out = list(llm_generate(_batches(["ab", "c"]), factory, prompt_column="q", usage=True))
    t = pa.Table.from_batches(out)
    assert t.column("response").to_pylist() == ["AB", "C"]
    assert t.column("prompt_tokens").to_pylist() == [2, 1]
    assert t.column("completion_tokens").to_pylist() == [4, 2]


def test_usage_columns_null_when_engine_silent():
    # An engine that reports no usage → null token columns (still appended on usage=True).
    def factory():
        return lambda prompts: list(prompts)

    out = list(llm_generate(_batches(["a", "b"]), factory, prompt_column="q", usage=True))
    t = pa.Table.from_batches(out)
    assert t.column("prompt_tokens").to_pylist() == [None, None]
    assert t.column("completion_tokens").to_pylist() == [None, None]


def test_http_engine_records_token_usage(monkeypatch):
    from batcher.ml.llm import http_engine

    def fake_post_json(url, payload, *, headers, timeout, **kw):
        prompt = payload["messages"][-1]["content"]
        return {
            "choices": [{"message": {"content": prompt}}],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": 1},
        }

    monkeypatch.setattr("batcher.ml.serving.http.post_json", fake_post_json)
    engine = http_engine("http://x/v1", "m", concurrency=3)()
    assert engine(["aa", "bbb"]) == ["aa", "bbb"]
    assert engine.last_usage == [(2, 1), (3, 1)]  # per-prompt, in order


def test_http_engine_serializes_when_concurrency_one(monkeypatch):
    from batcher.ml.llm import http_engine

    order: list[str] = []

    def fake_post_json(url, payload, *, headers, timeout, **kw):
        prompt = payload["messages"][-1]["content"]
        order.append(prompt)
        return {"choices": [{"message": {"content": prompt}}]}

    monkeypatch.setattr("batcher.ml.serving.http.post_json", fake_post_json)
    engine = http_engine("http://x/v1", "m", concurrency=1)()
    assert engine(["a", "b", "c"]) == ["a", "b", "c"]
    assert order == ["a", "b", "c"]  # strictly sequential
