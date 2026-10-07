"""Live smoke test: `ds.ml.token_count` and `TokenBudget` with a tokenizer from the Hub.

Skipped unless ``BATCHER_LIVE_HF_TOKENIZER`` names a Hub model id (for example ``gpt2``); the
unit suite builds its tokenizer locally and never downloads one.
"""

from __future__ import annotations

import os

import pytest

import batcher as bt

pytestmark = pytest.mark.skipif(
    not os.environ.get("BATCHER_LIVE_HF_TOKENIZER"),
    reason="set BATCHER_LIVE_HF_TOKENIZER to a Hub model id to run",
)


def test_counts_match_the_tokenizer_and_the_budget():
    from transformers import AutoTokenizer

    name = os.environ["BATCHER_LIVE_HF_TOKENIZER"]
    texts = ["Hello world", "A longer sentence with several more tokens in it."]
    expected = [len(AutoTokenizer.from_pretrained(name)(t)["input_ids"]) for t in texts]
    counted = bt.from_pydict({"t": texts}).ml.token_count("t", tokenizer=name)
    assert counted.to_pydict()["t_tokens"] == expected
    from batcher.ml.llm.tokens import BudgetedBatches

    _, groups = BudgetedBatches(bt.ml.TokenBudget(name, sum(expected))).plan(texts)
    assert groups == [[0, 1]]
