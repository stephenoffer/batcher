"""Splitting long text into model-sized chunks.

`chunk` splits on a character budget with an overlap, which is what a retrieval index
wants: the overlap keeps a sentence that straddles a boundary retrievable from either
side. The result is a list column, so `explode` turns chunks into rows.

    python examples/expr_text/tokens_and_chunking.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    documents = bt.from_pydict(
        {
            "doc": [1, 2],
            "body": [
                "The quick brown fox jumps over the lazy dog. " * 12,
                "Short document that fits in a single chunk.",
            ],
        }
    )

    chunked = documents.select(
        "doc",
        chunks=col("body").str.chunk(120, overlap=20),
    )
    counts = chunked.select("doc", n=col("chunks").list.len()).to_pydict()
    print(counts)

    # The long document splits; the short one does not.
    assert counts["n"][0] > 1
    assert counts["n"][1] == 1

    # One row per chunk, ready for embedding.
    rows = chunked.explode("chunks").select("doc", text=col("chunks"))
    print("chunk rows:", rows.count())
    assert rows.count() == sum(counts["n"])

    lengths = rows.select(length=col("text").str.len_chars()).to_pydict()["length"]
    print("chunk lengths:", lengths[:5])
    assert all(0 < length <= 120 for length in lengths)

    # n-grams over the tokens, for a lexical index alongside the vector one.
    grams = documents.select("doc", grams=col("body").str.token_ngrams(2)).select(
        "doc", n=col("grams").list.len()
    )
    print(grams.to_pydict())
    assert all(value > 0 for value in grams.to_pydict()["n"])


if __name__ == "__main__":
    main()
