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
