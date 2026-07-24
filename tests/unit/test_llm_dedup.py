"""Intra-batch prompt deduplication in `llm_generate` (opt-in `dedup=True`).

A corpus with repeated prompts should hit the engine once per distinct prompt and copy the
result — text, usage, finish_reason, logprob — to every row that shares it, keeping all
columns aligned to their rows. Fakes only; no model.
"""

from __future__ import annotations

import pyarrow as pa

from batcher.ml.llm import llm_generate
from batcher.ml.llm.channels import finish_reason_sink, logprob_sink, usage_sink


class _CountingEngine:
    """Reports usage/finish/logprob per request and records how many it was asked to run."""

    def __init__(self) -> None:
        self.seen: list[list[str]] = []

    def __call__(self, prompts: list) -> list[str]:
        self.seen.append([p if isinstance(p, str) else p["prompt"] for p in prompts])
        texts = [f"answer:{p}" for p in self.seen[-1]]
        usage_sink().report([(len(p), 2) for p in self.seen[-1]])
        finish_reason_sink().report(["stop"] * len(texts))
        logprob_sink().report([-1.0 * len(p) for p in self.seen[-1]])
        return texts


def _run(dedup: bool):
    engine = _CountingEngine()
    batch = pa.RecordBatch.from_pydict({"q": ["a", "b", "a", "a", "b"]})
    out = list(
        llm_generate(
            [batch],
            lambda: engine,
            prompt_column="q",
            usage=True,
            finish_reason=True,
            logprobs=True,
            dedup=dedup,
        )
    )
    return engine, out[0]


def test_dedup_runs_the_engine_once_per_distinct_prompt():
    engine, _res = _run(dedup=True)
    # Two distinct prompts, so the engine sees exactly two requests.
    assert sorted(engine.seen[0]) == ["a", "b"]
    assert len(engine.seen[0]) == 2


def test_dedup_fans_the_result_back_to_every_row_in_order():
    _engine, res = _run(dedup=True)
    d = res.to_pydict()
    assert d["response"] == ["answer:a", "answer:b", "answer:a", "answer:a", "answer:b"]
    # usage/finish/logprob stay aligned to each row's prompt.
    assert d["prompt_tokens"] == [1, 1, 1, 1, 1]  # len("a")=1, len("b")=1
    assert d["completion_tokens"] == [2, 2, 2, 2, 2]
    assert d["finish_reason"] == ["stop"] * 5
    assert d["logprob"] == [-1.0, -1.0, -1.0, -1.0, -1.0]


def test_without_dedup_the_engine_sees_every_row():
    engine, res = _run(dedup=False)
    assert len(engine.seen[0]) == 5
    assert res.to_pydict()["response"] == [
        "answer:a",
        "answer:b",
        "answer:a",
        "answer:a",
        "answer:b",
    ]


def test_dedup_and_no_dedup_agree_on_the_output():
    _e1, r1 = _run(dedup=True)
    _e2, r2 = _run(dedup=False)
    assert r1.to_pydict() == r2.to_pydict()
