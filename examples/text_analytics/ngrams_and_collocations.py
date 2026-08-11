"""Word pairs: the bigrams that appear together more than chance.

A bigram count is a group-by over adjacent word pairs, which a window makes easy: lag the
word column inside each document and group on the pair. The result is the cheapest phrase
detector there is.

    python examples/text_analytics/ngrams_and_collocations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    comments = (
        tpch("lineitem")
        .select("l_orderkey", "l_linenumber", "l_comment")
        .head(20_000)
        .with_row_index(name="doc")
    )

    words = (
        comments.select("doc", word=col("l_comment").str.to_lowercase().str.split(" "))
        .explode("word")
        .filter(col("word").str.len_chars() > 2)
    )

    # Lag inside each document to pair each word with the one before it.
    # A ranking window needs an explicit order: "the row before this one" is not
    # defined without one, and the engine says so rather than picking arbitrarily.
    paired = (
        words.with_columns(
            position=bt.row_number().over(partition_by=["doc"], order_by=["word"]),
        )
        .with_columns(
            previous=col("word").shift(1).over(partition_by=["doc"], order_by=["position"])
        )
        .filter(col("previous").is_not_null())
        .with_columns(bigram=bt.concat_ws(" ", col("previous"), col("word")))
    )

    top = (
        paired.group_by("bigram").agg(n=bt.count()).sort("n", descending=True).limit(10).to_pydict()
    )
    for bigram, count in zip(top["bigram"], top["n"], strict=True):
        print(f"  {count:>5}  {bigram}")

    assert top["n"] == sorted(top["n"], reverse=True)
    assert all(" " in value for value in top["bigram"])

    # Every bigram is two words, and there is one fewer bigram than word per document.
    assert paired.count() == words.count() - words.n_unique("doc")

    # The most common bigram appears more than once, which is what makes it a collocation
    # rather than a coincidence.
    assert top["n"][0] > 1


if __name__ == "__main__":
    main()
