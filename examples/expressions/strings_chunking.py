"""Splitting long documents into overlapping chunks for a RAG index.

``chunk`` is the columnar version of the loop everyone writes by hand before indexing.
Overlap matters: without it, a sentence spanning a boundary is retrievable from neither
chunk, and that is exactly the passage the question was about.

    python examples/expressions/strings_chunking.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    docs = bt.from_pydict(
        {
            "doc_id": ["d1", "d2"],
            "body": [
                "The refund window is thirty days from delivery. "
                "Shipping is free above fifty dollars. "
                "Returns require the original packaging.",
                "Short document.",
            ],
        }
    )

    # A list column of fixed-size character chunks.
    chunked = docs.with_columns(
        pieces=col("body").str.chunk(size=60, overlap=10),
        n_pieces=col("body").str.chunk(size=60, overlap=10).list.len(),
    ).to_pydict()

    print(chunked["n_pieces"])
    assert chunked["n_pieces"][0] > 1
    # A document shorter than the window yields exactly one chunk.
    assert chunked["n_pieces"][1] == 1
    assert chunked["pieces"][1] == ["Short document."]

    # Overlap means consecutive chunks share their boundary text.
    first, second = chunked["pieces"][0][0], chunked["pieces"][0][1]
    assert first[-10:] == second[:10]

    # More overlap yields more chunks over the same text.
    tighter = docs.select(n=col("body").str.chunk(size=60, overlap=30).list.len()).to_pydict()
    assert tighter["n"][0] >= chunked["n_pieces"][0]

    # Chunk on a word boundary instead of mid-word, when the text is prose.
    words = docs.select(
        pieces=col("body").str.chunk(size=60, overlap=10, boundary="word")
    ).to_pydict()
    print(words["pieces"][0][:2])
    assert len(words["pieces"][0]) >= 1

    # The shape a RAG index actually wants: one row per chunk, with its source id.
    exploded = (
        docs.select(
            doc_id=col("doc_id"),
            chunk=col("body").str.chunk(size=60, overlap=10),
        )
        .explode("chunk")
        .with_row_index("chunk_no")
        .to_pydict()
    )
    print(exploded["doc_id"])
    assert len(exploded["chunk"]) == sum(chunked["n_pieces"])
    assert exploded["doc_id"][0] == "d1"
    assert exploded["chunk_no"] == list(range(len(exploded["chunk"])))


if __name__ == "__main__":
    main()
