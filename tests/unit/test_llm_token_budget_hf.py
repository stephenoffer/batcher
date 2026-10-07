"""`ds.ml.token_count` and `TokenBudget` over a real HuggingFace fast tokenizer (AP-218/388).

The tokenizer is built in memory from a six-word vocabulary and saved to a temporary
directory, so the by-path load goes through ``AutoTokenizer.from_pretrained`` with no
network. That is the same code a worker runs for a Hub id.
"""

from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

import batcher as bt  # noqa: E402
from batcher.ml.llm.tokens import BudgetedBatches, TokenBudget, encode_texts  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    vocab = {"[UNK]": 0, "[BOS]": 1, "hello": 2, "world": 3, "a": 4, "b": 5}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[BOS] $A", special_tokens=[("[BOS]", 1)]
    )
    hf = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", bos_token="[BOS]"
    )
    path = tmp_path_factory.mktemp("tok")
    hf.save_pretrained(str(path))
    return str(path)


def test_token_count_loads_a_tokenizer_by_path_and_counts_exactly(tokenizer_dir):
    ds = bt.from_pydict({"t": ["hello world", "a b a b", None]})
    with_bos = ds.ml.token_count("t", tokenizer=tokenizer_dir).to_pydict()["t_tokens"]
    bare = ds.ml.token_count("t", tokenizer=tokenizer_dir, add_special_tokens=False)
    assert with_bos == [3, 5, None]
    assert bare.to_pydict()["t_tokens"] == [2, 4, None]


def test_budget_counts_with_the_same_encoding_as_the_column(tokenizer_dir):
    """The acceptance for AP-218: the filter and the request batcher agree on every count."""
    hf = transformers.AutoTokenizer.from_pretrained(tokenizer_dir)
    texts = ["hello world", "a b a b", "a"]
    column = (
        bt.from_pydict({"t": texts})
        .ml.token_count("t", tokenizer=hf, add_special_tokens=False)
        .to_pydict()["t_tokens"]
    )
    batched = [len(ids) for ids in encode_texts(hf, texts, add_special_tokens=False)]
    assert column == batched == [2, 4, 1]
    planner = BudgetedBatches(TokenBudget(tokenizer_dir, 5, add_special_tokens=False))
    _, groups = planner.plan(texts)
    assert groups == [[0], [1, 2]]


def test_oversized_prompt_is_cut_with_the_real_tokenizer(tokenizer_dir):
    planner = BudgetedBatches(TokenBudget(tokenizer_dir, 2, add_special_tokens=False))
    with pytest.warns(UserWarning, match="truncated to 2 tokens"):
        fitted, _ = planner.plan(["a b hello world"])
    assert fitted == ["a b"]
