"""Real-tokenizer counts and the per-batch token budget engines honour (AP-218, AP-388).

`ds.ml.token_count` and `TokenBudget` share one encoding path (`tokens.encode_texts`) and one
special-token policy, so a corpus filtered on the count column is counted the same way when
an engine groups its requests. These tests drive both with tokenizer stand-ins; the real
HuggingFace path is in `test_llm_token_budget_hf.py`.
"""

from __future__ import annotations

import threading
import time

import pytest

import batcher as bt
from batcher._internal.errors import DataQualityError, PlanError
from batcher.ml.llm.engines.base import batched_engine
from batcher.ml.llm.tokens import BudgetedBatches, TokenBudget, encode_texts

pytestmark = pytest.mark.unit


class _WordTokenizer:
    """One token per whitespace word, plus a BOS when special tokens are on (HF-shaped)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, texts, add_special_tokens=True):
        self.calls += 1
        return {"input_ids": [self.encode(t, add_special_tokens) for t in texts]}

    def encode(self, text, add_special_tokens=True):
        words = [len(w) for w in text.split()]
        return ([0] if add_special_tokens else []) + words

    def decode(self, ids):
        return " ".join("w" * i for i in ids if i)


def test_token_count_appends_int64_counts_and_keeps_nulls():
    ds = bt.from_pydict({"t": ["a b c", None, "", "d"]})
    out = ds.ml.token_count("t", tokenizer=str.split)
    assert out.to_pydict() == {"t": ["a b c", None, "", "d"], "t_tokens": [3, None, 0, 1]}
    import pyarrow as pa

    assert out.collect().schema.field("t_tokens").type == pa.int64()


def test_token_count_applies_the_special_token_policy():
    ds = bt.from_pydict({"t": ["a b", "c"]})
    tok = _WordTokenizer()
    default = ds.ml.token_count("t", tokenizer=tok, output_column="n").to_pydict()["n"]
    bare = ds.ml.token_count("t", tokenizer=tok, add_special_tokens=False).to_pydict()
    assert default == [3, 2]  # the tokenizer's own default adds a BOS
    assert bare["t_tokens"] == [2, 1]


def test_token_count_calls_a_batched_tokenizer_once_per_batch():
    tok = _WordTokenizer()
    assert encode_texts(tok, ["a", "b c", "d"]) == [[0, 1], [0, 1, 1], [0, 1]]
    assert tok.calls == 1


def test_token_count_refuses_a_non_text_column_and_a_bad_tokenizer():
    ds = bt.from_pydict({"n": [1, 2], "t": ["a", "b"]})
    with pytest.raises(PlanError, match="needs a text column"):
        ds.ml.token_count("n", tokenizer=str.split)
    with pytest.raises(PlanError, match="tokenizer"):
        ds.ml.token_count("t", tokenizer=42)


def test_budget_groups_are_consecutive_and_within_budget():
    budget = TokenBudget(str.split, max_batch_tokens=10)
    counts = [6, 4, 4, 3, 3, 1, 9]
    groups = budget.groups(counts)
    assert [i for g in groups for i in g] == list(range(len(counts)))  # order kept, all once
    assert all(sum(counts[i] for i in g) <= 10 for g in groups)
    assert groups == [[0, 1], [2, 3, 4], [5, 6]]


def test_longest_padding_charges_the_longest_prompt_times_the_group():
    budget = TokenBudget(str.split, max_batch_tokens=8, padding="longest")
    groups = budget.groups([4, 1, 1, 2, 2, 2, 2])
    assert all(max([4, 1, 1, 2, 2, 2, 2][i] for i in g) * len(g) <= 8 for g in groups)
    assert groups == [[0, 1], [2, 3, 4, 5], [6]]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_batch_tokens": 0}, "positive integer"),
        ({"max_batch_tokens": 8, "padding": "max"}, "padding"),
        ({"max_batch_tokens": 8, "oversized": "middle"}, "truncation must be"),
    ],
)
def test_budget_validation(kwargs, match):
    with pytest.raises(PlanError, match=match):
        TokenBudget(str.split, **kwargs)


def test_oversized_prompt_is_cut_to_the_budget_head_and_tail():
    tok = _WordTokenizer()
    head = BudgetedBatches(TokenBudget(tok, 3, add_special_tokens=False))
    with pytest.warns(UserWarning, match=r"TokenBudget\(oversized='head'\)"):
        fitted, groups = head.plan(["a bb ccc dddd", {"prompt": "e", "max_tokens": 4}])
    assert fitted == ["w ww www", {"prompt": "e", "max_tokens": 4}]
    assert groups == [[0], [1]]
    tail = BudgetedBatches(TokenBudget(tok, 2, oversized="tail", add_special_tokens=False))
    with pytest.warns(UserWarning, match="losing their heads"):
        fitted, _ = tail.plan(["a bb ccc dddd"])
    assert fitted == ["www wwww"]


def test_oversized_error_refuses_before_anything_is_sent():
    sent: list = []
    budget = TokenBudget(_WordTokenizer(), 2, oversized="error", add_special_tokens=False)
    engine = batched_engine(lambda p: sent.append(p), None, 1, token_budget=budget)
    with pytest.raises(DataQualityError, match=r"1 of 2 prompts exceed the 2-token budget"):
        engine(["a b c", "d"])
    assert sent == []


class _Reply:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage = (None, None)
        self.finish_reason = "stop"


def test_engine_keeps_tokens_in_flight_under_the_budget_on_ragged_text():
    """The acceptance: a batch respects the configured token budget on ragged text."""
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()
    in_flight = {"tokens": 0, "peak": 0}

    def call_one(prompt):
        n = len(prompt.split())
        with lock:
            in_flight["tokens"] += n
            in_flight["peak"] = max(in_flight["peak"], in_flight["tokens"])
        time.sleep(0.01)
        with lock:
            in_flight["tokens"] -= n
        return _Reply(prompt.upper())

    ragged = ["x " * n for n in (9, 1, 7, 2, 2, 8, 1, 1, 5, 3)]
    budget = TokenBudget(str.split, max_batch_tokens=10)
    with ThreadPoolExecutor(max_workers=16) as pool:
        engine = batched_engine(call_one, pool, 16, token_budget=budget)
        out = engine(ragged)
    assert out == [p.upper() for p in ragged]  # one reply per request, in request order
    assert in_flight["peak"] <= 10
    unbudgeted_peak = sum(len(p.split()) for p in ragged)
    assert unbudgeted_peak > 10  # without the budget the whole batch would be in flight


def test_http_engine_accepts_a_token_budget(monkeypatch):
    import batcher.ml.serving.http as http_mod

    seen: list[str] = []

    def fake(url, body, **kw):
        seen.append(body["messages"][-1]["content"])
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(http_mod, "post_json", fake)
    budget = TokenBudget(_WordTokenizer(), max_batch_tokens=2, add_special_tokens=False)
    engine = bt.ml.http_engine("http://x/v1", "m", concurrency=1, token_budget=budget)()
    with pytest.warns(UserWarning, match="truncated"):
        assert engine(["a bb ccc", "d"]) == ["ok", "ok"]
    assert seen == ["w ww", "d"]


def test_cutting_needs_a_tokenizer_that_can_decode():
    planner = BudgetedBatches(TokenBudget(str.split, max_batch_tokens=1))
    with pytest.raises(PlanError, match=r"needs a tokenizer with \.decode"):
        planner.plan(["a b"])
    assert planner.plan(["a", "b"])[1] == [[0], [1]]
