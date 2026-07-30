"""Token-level text metrics over rows that normalize to nothing.

`tokens()` applies SQuAD normalization, which drops articles. Splitting the resulting
empty string yielded `[""]` — one phantom token — so a row with no real tokens read as
"one distinct token out of one", a perfect score. `distinct_token_ratio` therefore
returned 1.0 for `"the the the ..."`, the exact degeneration it exists to detect and the
example its own docstring gives.

Every metric built on `tokens()` shared the bug, so the contract is pinned here rather
than only in the diversity module's tests.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col
from batcher.plan.functions.metrics.text._text import tokens

pytestmark = pytest.mark.unit


def _tokens_of(text: str) -> list[str]:
    return bt.from_pydict({"t": [text]}).select(x=tokens(col("t"))).to_pydict()["x"][0]


def test_a_row_of_only_articles_has_no_tokens() -> None:
    """Articles are dropped by normalization, so nothing is left — not one empty token."""
    assert _tokens_of("the the the") == []
    assert _tokens_of("a an the") == []
    assert _tokens_of("") == []


def test_real_words_still_tokenize() -> None:
    """The article stripping does not disturb ordinary text."""
    assert _tokens_of("cat sat mat") == ["cat", "sat", "mat"]
    assert _tokens_of("the cat sat") == ["cat", "sat"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # A repetition loop scores zero diversity, not one.
        ("the the the the the the", 0.0),
        ("cat cat cat cat", 0.25),
        ("the cat the cat", 0.5),
        ("a b a b a b", 1 / 3),
        ("cat sat mat", 1.0),
    ],
)
def test_distinct_token_ratio(text: str, expected: float) -> None:
    """The Distinct-1 score is unique tokens over total tokens, with no phantom token."""
    got = bt.from_pydict({"o": [text]}).select(d=bt.distinct_token_ratio("o")).to_pydict()["d"][0]
    assert got == pytest.approx(expected)


def test_the_documented_corpus_example() -> None:
    """The value `distinct_token_ratio`'s own docstring promises."""
    ds = bt.from_pydict({"o": ["cat sat mat", "cat cat cat cat"]})
    assert ds.agg(d=bt.distinct_token_ratio("o")).to_pydict()["d"][0] == pytest.approx(0.625)


def test_token_overlap_metrics_ignore_an_article_only_row() -> None:
    """A row that normalizes away contributes a zero, not a spurious perfect match."""
    graded = bt.from_pydict({"p": ["the"], "r": ["a"]})
    # Both sides normalize to nothing, so there is no overlap to credit.
    assert graded.select(f=bt.token_set_f1("p", "r")).to_pydict()["f"][0] == 0.0
