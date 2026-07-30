"""The SQuAD normalization kernel against the composition it replaced.

`normalize` was five expressions — `lower`, three `regexp_replace_all` passes, two trims — and
is now one engine kernel. Every word-level metric in the package tokenizes through it, so a
drift here silently changes `token_set_f1`, `answer_groundedness`, BLEU, ROUGE, and the
diversity monitors all at once, in a direction no test would name.

The oracle is therefore the composition itself, spelled out here and compared byte for byte.
The cases are chosen where the two could diverge: the article rule, the delete-versus-replace
asymmetry of punctuation, and the Unicode classes where "word character" is a judgement call.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.plan.expr_ir.constructors import col
from batcher.plan.functions.metrics.text._text import normalize

pytestmark = pytest.mark.differential


def _composition(expr):
    """The five-expression chain the kernel replaced, as the oracle."""
    return (
        expr.str.lower()
        .str.regexp_replace_all(r"\b(a|an|the)\b", " ")
        .str.remove_punctuation()
        .str.normalize_whitespace()
        .str.strip()
    )


_CASES = [
    "The quick brown Fox!",
    "a cat, an apple, the dog",
    "cat-dog",
    "cat, dog",
    "  leading and trailing  ",
    "",
    "   ",
    "the",
    "the the the",
    "a",
    "an_apple",
    "hello_world 42",
    "theatre and thistle",
    "Ünïcödé tëxt wîth àccents",
    "东京都 and 東京市",
    "naïve café résumé",
    "punctuation!!!...???",
    "mixed123 456abc",
    "tabs\tand\nnewlines",
    "The-The-The",
    "a.b.c",
    "e.g. the thing",
    "don't can't won't",
    "50% of $100",
    "emoji 🙂 survives?",
    None,
]


def test_the_kernel_matches_the_composition_it_replaced():
    ds = bt.from_pydict({"s": _CASES})
    got = ds.select(kernel=normalize(col("s")), chain=_composition(col("s"))).to_pydict()
    for case, kernel, chain in zip(_CASES, got["kernel"], got["chain"], strict=True):
        assert kernel == chain, f"differed on {case!r}: {kernel!r} != {chain!r}"


def test_the_documented_behaviours_hold():
    """The three rules a caller needs to know, stated as assertions rather than prose."""
    ds = bt.from_pydict({"s": ["The cat", "theatre", "cat-dog", "cat, dog", "  x  "]})
    got = ds.select(n=normalize(col("s"))).to_pydict()["n"]
    assert got[0] == "cat"  # a standalone article is dropped
    assert got[1] == "theatre"  # an article inside a word is not
    assert got[2] == "catdog"  # punctuation is deleted, joining its neighbours
    assert got[3] == "cat dog"  # unless it carried whitespace
    assert got[4] == "x"  # the ends are trimmed


def test_a_null_stays_null():
    ds = bt.from_pydict({"s": [None, "The cat"]})
    assert ds.select(n=normalize(col("s"))).to_pydict()["n"] == [None, "cat"]


def test_the_metrics_built_on_it_are_unchanged():
    """The reason the equivalence matters: these numbers are published in docs and tests."""
    pairs = bt.from_pydict({"p": ["the cat sat"], "r": ["the cat sat down"]})
    got = pairs.agg(
        clipped=bt.ngram_precision("p", "r"),
        token_set=bt.token_set_f1("p", "r"),
        grounded=bt.answer_groundedness("p", "r"),
    ).to_pydict()
    assert got["clipped"] == [1.0]
    assert got["token_set"] == [0.8]
    assert got["grounded"] == [1.0]
