"""A word-frequency table over real text, entirely in the engine.

Split into a list, explode into rows, group and count. No Python touches a token, which is
what keeps this the same shape at a thousand rows and at a billion.

    python examples/text_analytics/word_frequencies.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    comments = tpch("lineitem").select("l_comment").head(20_000)

    words = (
        comments.select(word=col("l_comment").str.to_lowercase().str.split(" "))
        .explode("word")
        .filter(col("word").str.len_chars() > 3)
    )

    frequencies = (
        words.group_by("word").agg(n=bt.count()).sort("n", descending=True).limit(15).to_pydict()
    )

    for word, count in zip(frequencies["word"], frequencies["n"], strict=True):
        print(f"  {word:<16} {count:>6}")

    assert frequencies["n"] == sorted(frequencies["n"], reverse=True)
    assert all(len(word) > 3 for word in frequencies["word"])
    assert all(word == word.lower() for word in frequencies["word"])

    # The total token count reconciles with the per-row word counts.
    total_tokens = words.count()
    print("tokens over 3 characters:", total_tokens)
    assert total_tokens > comments.count()

    # Vocabulary size versus token count is the type/token ratio.
    vocabulary = words.n_unique("word")
    print(f"vocabulary {vocabulary}, ratio {vocabulary / total_tokens:.4f}")
    assert 0 < vocabulary < total_tokens


if __name__ == "__main__":
    main()
