"""Counting the structure of a document: words, sentences, lines, paragraphs.

These are the features a length filter actually wants. A token estimate is the one to
gate an LLM call on, because it is what the bill is denominated in — and `estimate_tokens`
is a heuristic, so treat it as a bound rather than a measurement.

    python examples/expr_text/document_shape.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    comments = tpch("customer").select("c_custkey", "c_comment").head(1_000)

    shaped = comments.select(
        "c_custkey",
        characters=col("c_comment").str.len_chars(),
        words=col("c_comment").str.word_count(),
        sentences=col("c_comment").str.sentence_count(),
        lines=col("c_comment").str.line_count(),
        tokens=col("c_comment").str.estimate_tokens(),
        avg_word=col("c_comment").str.avg_word_length(),
    )

    result = shaped.head(3).to_pydict()
    print(result)

    # A comment is one line of prose, so line count is 1 and words exceed sentences.
    assert all(value == 1 for value in result["lines"])
    assert all(
        words >= sentences
        for words, sentences in zip(result["words"], result["sentences"], strict=True)
    )

    # Characters bound words, and the average word length reconciles the two.
    assert all(
        characters >= words
        for characters, words in zip(result["characters"], result["words"], strict=True)
    )

    # A token estimate sits between word count and character count, which is the range
    # any sane tokenizer lands in for English.
    checks = shaped.filter(
        (col("tokens") < col("words")) | (col("tokens") > col("characters"))
    ).count()
    assert checks == 0

    # The filter this exists for: keep only documents that fit a budget.
    within = comments.filter(col("c_comment").str.fits_token_budget(40))
    print(f"{within.count()} of {comments.count()} fit a 40-token budget")
    assert 0 < within.count() <= comments.count()


if __name__ == "__main__":
    main()
